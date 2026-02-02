"""
End-to-end inference: take raw text, run exported model, return anonymized text.
Uses ONNX export by default; supports TorchScript via --backend pt.
PII spans are replaced with [MASKED].
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_EXPORT_DIR = os.path.join(BASE_DIR, "model_export")

MODEL_NAME = "tanaos/tanaos-text-anonymizer-v1"
SAFE_NAME = MODEL_NAME.replace("/", "_")
MAX_LENGTH = 128


def get_id2label():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    return config.id2label


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


def load_onnx():
    import onnxruntime as ort

    path = os.path.join(MODEL_EXPORT_DIR, f"{SAFE_NAME}.onnx")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def load_torchscript():
    path = os.path.join(MODEL_EXPORT_DIR, f"{SAFE_NAME}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TorchScript model not found: {path}")
    return torch.jit.load(path)


def run_onnx(session, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    feeds = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64),
    }
    out = session.run(None, feeds)
    return out[0]


def run_torchscript(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits.numpy()


def _get_label(id2label, pred_id):
    pid = int(pred_id)
    return id2label.get(pid, id2label.get(str(pid), "O"))


def predict_and_anonymize(
    text: str,
    tokenizer,
    id2label: dict,
    backend: str = "onnx",
) -> str:
    """Tokenize text, run model, merge B-I spans and replace PII with [MASKED].
    Uses token-level masking then convert_tokens_to_string (works with slow tokenizer).
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if backend == "onnx":
        session = load_onnx()
        logits = run_onnx(
            session,
            input_ids.numpy(),
            attention_mask.numpy(),
        )
    else:
        model = load_torchscript()
        logits = run_torchscript(model, input_ids, attention_mask)

    # (1, seq_len, num_labels) -> (seq_len,)
    pred_ids = np.argmax(logits[0], axis=-1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Only consider tokens up to last real token (exclude padding)
    seq_len = int(attention_mask[0].sum().item())

    # Build masked token list: for each PII span (B-I consecutive), replace with single [MASKED]
    masked_tokens = []
    i = 0
    while i < seq_len:
        if attention_mask[0, i].item() == 0:
            masked_tokens.append(tokens[i])
            i += 1
            continue
        label = _get_label(id2label, pred_ids[i])
        if label == "O":
            masked_tokens.append(tokens[i])
            i += 1
            continue
        entity_type = (
            label[2:] if (label.startswith("B-") or label.startswith("I-")) else label
        )
        # One [MASKED] per entity: skip I-X of same type, or consecutive B-X (some models predict B per token)
        masked_tokens.append("[MASKED]")
        i += 1
        while i < seq_len and attention_mask[0, i].item() == 1:
            next_label = _get_label(id2label, pred_ids[i])
            if next_label == "I-" + entity_type or next_label == "B-" + entity_type:
                i += 1  # same entity (I- continuation or mistaken B-)
            else:
                break

    out = tokenizer.convert_tokens_to_string(masked_tokens)
    # Strip any trailing padding artifacts (e.g. RoBERTa adds space before </s>)
    return out.replace("<pad>", "").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize text using exported tanaos model."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Input text to anonymize (or use -f/--file).",
    )
    parser.add_argument("-f", "--file", help="Read input text from file.")
    parser.add_argument(
        "--backend",
        choices=["onnx", "pt"],
        default="onnx",
        help="Exported model backend (default: onnx).",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        # Default test sentence
        text = "John Doe lives in Paris. Call me at 555-1234 or visit 123 Main St. Born on 1990-01-15."

    tokenizer = load_tokenizer()
    id2label = get_id2label()

    anonymized = predict_and_anonymize(text, tokenizer, id2label, backend=args.backend)

    print("Input:     ", repr(text))
    print("Anonymized:", repr(anonymized))


if __name__ == "__main__":
    main()
