import argparse
import math
import os
from pathlib import Path
import tiktoken
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import re

from gpt2 import GPTModel
from utils import generate_and_return_sample, calc_loss_batch, evaluate_model, create_dataloader_from_pt
from galore_torch import GaLoreAdamW


def read_tokenized_file(file_path):
    token_tensor = torch.load(file_path)
    return token_tensor


def create_dataloaders(token_tensor, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(token_tensor))

    train_loader = create_dataloader_from_pt(
        token_tensor[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_from_pt(
        token_tensor[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def estimate_training_steps(pt_file_paths, train_ratio=0.9, context_length=1024, stride=1024, batch_size=8):
    total_samples = 0
    for file_path in pt_file_paths:
        tokens = torch.load(file_path)
        num_samples = max(0, (len(tokens) - context_length) // stride)
        train_samples = int(num_samples * train_ratio)
        total_samples += train_samples

    total_steps = total_samples // batch_size
    return total_steps


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    global_step = checkpoint.get("global_step", 0)

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    track_tokens_seen = checkpoint.get("track_tokens_seen", [])
    track_lrs = checkpoint.get("track_lrs", [])
    file_index = checkpoint.get("file_index", 0)
    intra_file_step = checkpoint.get("intra_file_step", 0)

    print(f"Resumed from checkpoint at step {global_step}")
    return model, optimizer, global_step, train_losses, val_losses, track_tokens_seen, track_lrs, file_index, intra_file_step


def save_checkpoint(output_dir, global_step, model, optimizer, train_losses, val_losses, track_tokens_seen, track_lrs, current_file_index, current_file_step):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen,
        "track_lrs": track_lrs,
        "file_index": current_file_index,
        "intra_file_step": current_file_step
    }

    file_name = output_dir/f"checkpoint_step{global_step}.pt"
    torch.save(checkpoint, file_name)
    print(f"Saved checkpoint: {file_name}")


def save_array_to_txt(array, file_path):
    file_path = Path(file_path)
    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)  # Create the file if it doesn't exist
    with open(file_path, "w", encoding="utf-8") as f:
        for value in array:
            f.write(f"{value}\n")


def get_param_groups(model):
    galore_params = []
    non_galore_params = []

    for name, param in model.named_parameters():
        is_bias = ("bias" in name and param.ndim == 1)
        is_layernorm = "norm" in name
        is_embedding = "tok_emb" in name or "pos_emb" in name
        is_lm_head = "out_head" in name  # final LM head

        if is_bias or is_layernorm or is_embedding or is_lm_head:
            non_galore_params.append(param)
        else:
            galore_params.append(param)

    return [
        {"params": non_galore_params},
        {
            "params": galore_params,
            "use_galore": True,
            "rank": 16,
            "update_proj_gap": 200,
            "scale": 0.25,
            "proj_type": "std"
        }
    ]


def train_model_simple(model, optimizer, initial_lr, peak_lr, lr_increment, total_training_steps, warmup_steps,
                       device, n_epochs, eval_freq, eval_iter, print_sample_iter, start_context, output_dir, text_output_path,
                       save_ckpt_freq, tokenizer, all_files, prev_global_step=-1, prev_train_losses=None, prev_val_losses=None,
                       prev_track_tokens_seen=None, prev_track_lr=None, file_index=0, intra_file_step=0, batch_size=1024, train_ratio=0.90):

    global_step = prev_global_step
    tokens_seen = 0
    min_lr = 0.00005
    scaler = GradScaler(device=device)

    train_losses = prev_train_losses or []
    val_losses = prev_val_losses or []
    track_tokens_seen = prev_track_tokens_seen or []
    track_lrs = prev_track_lr or []

    try:
        for epoch in range(n_epochs):

            pbar = tqdm(total=global_total_steps, initial=global_step +
                        1, desc="Training Progress", unit="step")
            
            for index, file_path in enumerate(all_files[file_index:], start=file_index):
                token_tensor = read_tokenized_file(file_path)
                print(
                    f"\nOpening tokenized file {index} of {len(all_files)}: {file_path}")

                if global_step > 0 and index == file_index:
                    current_file_step = intra_file_step
                else:
                    current_file_step = 0
                print(f"Starting intra-file step: {current_file_step}")

                # Initialize new data loaders for each file
                train_loader, val_loader = create_dataloaders(
                    token_tensor,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=0
                )

                print("\nTraining ...")
                model.train()

                for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                    if global_step > 0 and index == file_index and batch_idx < intra_file_step:
                        print(
                            f"Skipping batch {batch_idx} out of {intra_file_step}")
                        continue

                    current_file_step += 1
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step < warmup_steps:
                        cur_lr = initial_lr + (global_step * lr_increment)
                    else:
                        progress = (global_step - warmup_steps) / \
                            (total_training_steps - warmup_steps)
                        cur_lr = min_lr + (peak_lr - min_lr) * \
                            0.5 * (1 + math.cos(math.pi * progress))

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = cur_lr
                    track_lrs.append(optimizer.param_groups[0]["lr"])

                    with torch.autocast(device_type="cuda"):
                        loss = calc_loss_batch(
                            input_batch, target_batch, model, device)
                    scaler.scale(loss).backward()

                    if global_step > warmup_steps:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    tokens_seen += input_batch.numel()
                    pbar.update(1)

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        print(
                            f"\nStep {global_step}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        text = generate_and_return_sample(
                            model, tokenizer, device, start_context)
                        with open(text_output_path, "a", encoding="utf-8") as f:
                            f.write(f"Step {global_step}:\n{text}\n")

                if global_step % save_ckpt_freq == 0:
                    save_checkpoint(output_dir, global_step, model, optimizer,
                                    train_losses, val_losses, track_tokens_seen, track_lrs, current_file_index=index, current_file_step=current_file_step)
            pbar.close()

    except KeyboardInterrupt:
        save_checkpoint(output_dir, global_step, model, optimizer,
                        train_losses, val_losses, track_tokens_seen, track_lrs, current_file_index=index, current_file_step=current_file_step)

    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Completed Pretraining")
    return train_losses, val_losses, track_tokens_seen, track_lrs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='GPT Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='gutenberg_tokenized',
                        help='Directory containing the training data'),
    parser.add_argument('--model_text_output_dir', type=str, default='model_output_during_training.txt',
                        help='Text file that loop will write model outputs to given input context')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=1000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for the optimizer'),
    parser.add_argument('--peak_lr', type=float, default=0.0004,
                        help='Max learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training'),
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes'),
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint (.pt) to resume training from')

    args = parser.parse_args()

    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "drop_rate": 0.0,
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG_124M)
    torch.compile(model)
    model.to(device)

    print(f"Using device: {device}")

    data_dir = args.data_dir

    def extract_number(path):
        match = re.search(r'combined_(\d+)\.pt', path)
        return int(match.group(1)) if match else float('inf')

    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".pt"))]
    all_files = sorted(all_files, key=extract_number)
    all_files = all_files[-15:]
    print(all_files)

    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    tokenized_dir = "gutenberg_tokenized"
    total_training_steps = estimate_training_steps(
        all_files, train_ratio=0.9, context_length=1024, stride=1024, batch_size=args.batch_size)
    warmup_steps = total_training_steps * 0.1

    optimizer = GaLoreAdamW(get_param_groups(
        model), lr=args.initial_lr, weight_decay=0.1, no_deprecation_warning=True)
    lr_increment = (args.peak_lr - args.initial_lr) / warmup_steps

    tokenizer = tiktoken.get_encoding("gpt2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_output_path = output_dir/f"{args.model_text_output_dir}"
    text_output_path.touch(exist_ok=True)

    train_losses, val_losses, track_tokens_seen, track_lrs, global_step, file_index, intra_file_step = None, None, None, None, -1, 0, 0

    if args.resume_from:
        model, optimizer, global_step, train_losses, val_losses, track_tokens_seen, track_lrs, file_index, intra_file_step = load_checkpoint(
            model, optimizer, args.resume_from, device)

        print(
            f"Resuming from global step {global_step}, file index {file_index}, intra-file step {intra_file_step}")

    global_total_steps = total_training_steps

    train_losses, val_losses, track_tokens_seen, track_lrs = train_model_simple(
        model=model, optimizer=optimizer, device=device,
        initial_lr=args.initial_lr, peak_lr=args.peak_lr,
        lr_increment=lr_increment, total_training_steps=total_training_steps,
        warmup_steps=warmup_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, eval_freq=args.eval_freq, eval_iter=5,
        print_sample_iter=args.print_sample_iter, output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq, start_context="Every effort moves you",
        text_output_path=text_output_path, tokenizer=tokenizer,
        all_files=all_files, prev_global_step=global_step,
        prev_train_losses=train_losses, prev_val_losses=val_losses,
        prev_track_tokens_seen=track_tokens_seen, prev_track_lr=track_lrs, file_index=file_index, intra_file_step=intra_file_step
    )

    save_array_to_txt(train_losses, output_dir / "train_losses.txt")
    save_array_to_txt(val_losses, output_dir / "val_losses.txt")
    save_array_to_txt(track_tokens_seen, output_dir / "tokens_seen.txt")
    save_array_to_txt(track_lrs, output_dir / "learning_rates.txt")

    print("Saved training curves to text files.")
