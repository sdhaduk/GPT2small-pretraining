import argparse
import os
import re
from tqdm import tqdm
from gutenberg.src.cleanup import strip_headers
import tiktoken
import torch
from langdetect import detect


def is_english(text, threshold=0.9):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold


def remove_title_block(text):
    # Look for first "real" paragraph — longer than 300 chars and ends in a period.
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        if len(para.strip()) > 300 and para.strip().endswith("."):
            return "\n\n".join(paragraphs[i:])
    return text


def clean_non_narrative_sections(text):
    # Remove Gutenberg-style Roman numeral page tags like {ix}, {x}, etc.
    text = re.sub(r"\{\s*[xivlcdm]+\s*\}", "", text, flags=re.IGNORECASE)

    # Remove lines made entirely of decorative asterisks and spaces
    text = re.sub(r"(?m)^\s*[\*\s]{5,}\s*$", "", text)

    # Remove full lines made up of unicode box characters
    text = re.sub(r"(?m)^[\u2500-\u257F\s]{3,}$", "", text)

    # Remove lines with multiple pipe-style dividers (like tables)
    text = re.sub(r"(?m)^.*[│|].*[│|].*$", "", text)

    # Remove long runs of dashes or symbols
    text = re.sub(r"(?m)^[-─━=~\s]{3,}$", "", text)

    # Remove [Illustration] and [Illustration: ...]
    text = re.sub(r"\[Illustration(?:.*?)?\]", "", text, flags=re.IGNORECASE)

    # Remove all-caps caption headings
    text = re.sub(r"(?m)^[A-Z0-9 ,.'\-()]{5,}$", "", text)

    # Remove Table of Contents style lines with page alignment
    text = re.sub(r"(?m)^.*\s{2,}\d+\s*$", "", text)
    text = re.sub(
        r"(?m)^(PLATE|FIGURE|TABLE|ILLUSTRATION)\s+\d+.*\d+\s*$", "", text)

    # Remove chapter-style headings: "CHAPTER I. ....... 1"
    text = re.sub(r"(?m)^ *(CHAPTER|[IVXLC]+)\.?.*?(\.+ +\d+)?$", "", text)

    # Remove blocks with a high ALL-CAPS word ratio
    paragraphs = text.split("\n\n")
    filtered_paragraphs = []
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        upper_words = sum(1 for w in words if w.isupper())
        if upper_words / len(words) < 0.7:
            filtered_paragraphs.append(para)

    cleaned_text = "\n\n".join(filtered_paragraphs)

    # Normalize spacing
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    return cleaned_text.strip()


def has_minimum_tokens_after_cleaning(text, encoder, min_tokens=1000):
    tokens = encoder.encode(text)
    token_count = len(tokens)

    if token_count < min_tokens:
        return False
    else:
        return True


def is_modern_english(text, encoder, max_token_to_word_ratio=3.0):
    try:
        lang = detect(text)
    except:
        return False

    if lang != "en":
        return False

    words = text.split()
    if not words:
        return False

    tokens = encoder.encode(text)
    token_to_word_ratio = len(tokens) / len(words)

    return token_to_word_ratio < max_token_to_word_ratio


def mark_file_as_processed(file_path, log_file):
    with open(log_file, "a") as f:
        f.write(file_path + "\n")


def load_processed_file_paths(log_file):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        return set(line.strip() for line in f.readlines())


def cleaning_pipeline(text, encoder, file_path):
    if not is_english(text):
        tqdm.write(f"Skipping {file_path} not english")
        return False, None

    text = strip_headers(text)
    text = remove_title_block(text)
    text = clean_non_narrative_sections(text)

    if not has_minimum_tokens_after_cleaning(text, encoder):
        tqdm.write(f"Skipping {file_path} not enough tokens")
        return False, None

    if not is_modern_english(text, encoder):
        tqdm.write(f"Skipping {file_path} token to word ratio too high")
        return False, None

    return True, text


def combine_files(file_paths, target_dir, encoder, file_counter, log_file="processed_files.txt", max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    
    # create dir for tokenized input
    os.makedirs(target_dir, exist_ok=True)

    used_files = []
    current_content = []
    current_size = 0
    file_counter = file_counter
    processed_files = load_processed_file_paths(log_file)
    
    for file_path in tqdm(file_paths):

        if file_path in processed_files:
                tqdm.write(f"Skipping already processed: {file_path}")
                continue
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:

            tqdm.write(
                f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()
        
        print(file_path)
        valid, content = cleaning_pipeline(content, encoder, file_path)
        if not valid:
            mark_file_as_processed(file_path, log_file)
            continue

        estimated_size = len(content.encode("utf-8"))

        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            # Save tokenized version
            tokens = encoder.encode(separator.join(current_content), allowed_special={"<|endoftext|>"})
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            token_path = os.path.join(target_dir, f"combined_{file_counter}.pt")
            torch.save(token_tensor, token_path)

            for used in used_files:
                mark_file_as_processed(used, log_file)
            used_files = []

            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            used_files.append(file_path)
            current_size += estimated_size
    
    # Save remaining content
    if current_content:
        tokens = encoder.encode(separator.join(current_content), allowed_special={"<|endoftext|>"})
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        token_path = os.path.join(target_dir, f"combined_{file_counter}.pt")
        torch.save(token_tensor, token_path)
    
    return file_counter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and combine text files for pretraining")

    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved"),
    parser.add_argument("--file_counter", type=int, default="1", help="What number to start file_counter at if continuing file processing after stopping previously")
    

    args = parser.parse_args()
    gpt2_encoder = tiktoken.get_encoding("gpt2")

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    print(f"{len(all_files)} file(s) to process.")
    file_counter = combine_files(
        all_files, args.output_dir, max_size_mb=args.max_size_mb, encoder=gpt2_encoder, file_counter=args.file_counter)
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")
