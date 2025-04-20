import argparse
import math
import os
from pathlib import Path
import time
import tiktoken
import torch
from tqdm import tqdm
import re

from gpt2 import GPTModel
from utils import generate_and_return_sample, calc_loss_batch, evaluate_model, create_dataloader_from_pt


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


def estimate_training_steps(tokenized_dir, train_ratio=0.9, context_length=1024, stride=1024):
    pt_files = [
        os.path.join(path, name)
        for path, _, files in os.walk(tokenized_dir)
        for name in files if name.endswith(".pt")
    ]

    total_steps = 0
    for file_path in pt_files:
        tokens = torch.load(file_path)
        steps = max(0, (len(tokens) - context_length) // stride)
        total_steps += steps

    return int(total_steps * train_ratio)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    global_step = checkpoint.get("global_step", 0)

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    track_tokens_seen = checkpoint.get("track_tokens_seen", [])
    track_lrs = checkpoint.get("track_lrs", [])

    print(f"Resumed from checkpoint at step {global_step}")
    return model, optimizer, global_step, train_losses, val_losses, track_tokens_seen, track_lrs


def train_model_simple(model, optimizer, initial_lr, peak_lr, lr_increment, total_training_steps, warmup_steps,
                       device, n_epochs, eval_freq, eval_iter, print_sample_iter, start_context, output_dir, text_output_path,
                       save_ckpt_freq, tokenizer, all_files, prev_global_step=-1, prev_train_losses=None, prev_val_losses=None, 
                       prev_track_tokens_seen=None, prev_track_lr=None, batch_size=1024, train_ratio=0.90):

    global_step = prev_global_step
    tokens_seen = 0
    min_lr = 0.00001
    
    train_losses = prev_train_losses or []
    val_losses = prev_val_losses or []
    track_tokens_seen = prev_track_tokens_seen or []
    track_lrs = prev_track_lr or []

    try:
        for epoch in range(n_epochs):

            pbar = tqdm(total=global_total_steps, initial=global_step+1, desc="Training Progress", unit="step")
            # Iterate over the books in the training corpus
            for index, file_path in enumerate(all_files, 1):
                token_tensor = read_tokenized_file(file_path)
                print(
                    f"\nOpening tokenized file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
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

                for input_batch, target_batch in train_loader:
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

                    loss = calc_loss_batch(
                        input_batch, target_batch, model, device)
                    loss.backward()

                    if global_step > warmup_steps:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)

                    optimizer.step()
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
                        text = generate_and_return_sample(model, tokenizer, device, start_context)
                        with open(text_output_path, "a") as f:
                            f.write(f"{global_step}\n: {text}\n")

                if global_step % save_ckpt_freq == 0:
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "track_tokens_seen": track_tokens_seen,
                        "track_lrs": track_lrs,
                    }
                    file_name = output_dir/f"checkpoint_step{global_step}.pt"
                    torch.save(checkpoint, file_name)
                    print(f"Saved checkpoint: {file_name}")

            pbar.close()
    except KeyboardInterrupt:
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
            "track_lrs": track_lrs,
        }
        file_name = output_dir/f"checkpoint_step{global_step}.pt"
        torch.save(checkpoint, file_name)
        print(f"Saved due to interrupt: {file_name}")
    
    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Completed Pretraining")
    return train_losses, val_losses, track_tokens_seen
    



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
    parser.add_argument('--print_sample_iter', type=int, default=10000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=10000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--initial_lr', type=float, default=0.00005,
                        help='Initial learning rate for the optimizer'),
    parser.add_argument('--peak_lr', type=float, default=0.0002,
                        help='Max learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
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
        "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(10)

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    tokenized_dir = "gutenberg_tokenized"
    total_training_steps = estimate_training_steps(
        tokenized_dir, train_ratio=0.9, context_length=1024, stride=1024)
    warmup_steps = total_training_steps * 0.1

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    lr_increment = (args.peak_lr - args.initial_lr) / warmup_steps

    tokenizer = tiktoken.get_encoding("gpt2")

    data_dir = args.data_dir
    def extract_number(path):
        match = re.search(r'combined_(\d+)\.pt', path)
        return int(match.group(1)) if match else float('inf')

    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".pt"))]
    all_files = sorted(all_files, key=extract_number)

    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_output_path = output_dir/f"{args.model_text_output_dir}"
    text_output_path.touch(exist_ok=True)

    train_losses, val_losses, track_tokens_seen, track_lrs, global_step = None, None, None, None, -1

    if args.resume_from:
        model, optimizer, global_step, train_losses, val_losses, track_tokens_seen, track_lrs = load_checkpoint(
            model, optimizer, args.resume_from, device)

        steps_so_far = 0
        remaining_files = []

        for file_path in all_files:
            token_tensor = torch.load(file_path)

            steps_in_file = max(0, (len(
                token_tensor) - GPT_CONFIG_124M["context_length"]) // GPT_CONFIG_124M["context_length"])
            if steps_so_far + steps_in_file <= global_step:
                steps_so_far += steps_in_file  # file was fully processed, skip it
            else:
                remaining_files.append(file_path)

        print(
            f"Resuming from global step {global_step}. Skipping {len(all_files) - len(remaining_files)} file(s).")
        all_files = remaining_files
    
    global_total_steps = 0
    for file_path in all_files:
        token_tensor = read_tokenized_file(file_path)
        global_total_steps += max(0, (len(token_tensor) - GPT_CONFIG_124M["context_length"]) // GPT_CONFIG_124M["context_length"])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model, optimizer=optimizer, device=device,
        initial_lr=args.initial_lr, peak_lr=args.peak_lr,
        lr_increment=lr_increment, total_training_steps=total_training_steps,
        warmup_steps=warmup_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, eval_freq=args.eval_freq, eval_iter=1,
        print_sample_iter=args.print_sample_iter, output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq, start_context="Every effort moves you",
        text_output_path=text_output_path, tokenizer=tokenizer, 
        all_files=all_files, prev_global_step=global_step, 
        prev_train_losses=train_losses, prev_val_losses=val_losses, 
        prev_track_tokens_seen=track_tokens_seen, prev_track_lr=track_lrs
    )