#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate input_ids.npy and attention_mask.npy")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--output-dir", default=".", help="Output directory for npy files")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    enc = tokenizer.encode(args.text)

    input_ids = list(enc.ids)
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > args.max_length:
        input_ids = input_ids[: args.max_length]
        attention_mask = attention_mask[: args.max_length]

    pad_token_id = tokenizer.token_to_id("<|end_of_text|>") or 0
    while len(input_ids) < args.max_length:
        input_ids.append(pad_token_id)
        attention_mask.append(0)

    input_ids_np = np.array([input_ids], dtype=np.int32)
    attention_mask_np = np.array([attention_mask], dtype=np.int32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input_ids.npy", input_ids_np)
    np.save(output_dir / "attention_mask.npy", attention_mask_np)

    print(f"Saved input_ids.npy and attention_mask.npy to {output_dir}")
    print(f"input_ids shape: {input_ids_np.shape} dtype: {input_ids_np.dtype}")
    print(f"attention_mask shape: {attention_mask_np.shape} dtype: {attention_mask_np.dtype}")


if __name__ == "__main__":
    main()

