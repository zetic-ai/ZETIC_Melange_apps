#!/usr/bin/env python3
"""
Export tokenizer and config for meta-llama/Llama-Prompt-Guard-2-86M for use in the iOS app.
Run from repo root or prepare/: pip install transformers huggingface_hub, set HF_TOKEN if model is gated.
Outputs: tokenizer.json, labels.json (into prepare/; copy into iOS app Resources).
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get("OVERRIDE_MODEL_ID", "meta-llama/Llama-Prompt-Guard-2-86M")


def _get_hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from huggingface_hub import get_token
        t = get_token()
        if t:
            return t
    except Exception:
        pass
    return None


def main():
    from transformers import AutoConfig, AutoTokenizer

    token_kw = {}
    hf_token = _get_hf_token()
    if hf_token:
        token_kw["token"] = hf_token

    print(f"Loading tokenizer and config: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_kw)
    config = AutoConfig.from_pretrained(MODEL_ID, **token_kw)

    # 1. Export id2label (class index -> label name) for optional use in app
    labels_path = os.path.join(SCRIPT_DIR, "labels.json")
    id2label = getattr(config, "id2label", None)
    if id2label is not None:
        # id2label can be dict with int keys; JSON keys must be strings
        out = {str(k): v for k, v in id2label.items()}
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Exported labels to {labels_path} (num_labels={len(out)})")
    else:
        print("No id2label in config; skipping labels.json")

    # 2. Export tokenizer.json (same format as TextAnonymizer: vocab in model.vocab or vocab)
    tokenizer_export_dir = os.path.join(SCRIPT_DIR, "tokenizer_export")
    tokenizer.save_pretrained(tokenizer_export_dir)
    src_json = os.path.join(tokenizer_export_dir, "tokenizer.json")
    dst_json = os.path.join(SCRIPT_DIR, "tokenizer.json")
    if os.path.exists(src_json):
        with open(src_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Exported tokenizer to {dst_json}")
    else:
        print("Warning: tokenizer.json not found after save_pretrained")

    print("Done. Copy tokenizer.json and labels.json into iOS app Resources (e.g. ZeticMLangePromptGuard-iOS/Resources).")


if __name__ == "__main__":
    main()
