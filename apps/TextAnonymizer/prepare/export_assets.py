
import json
import os
from transformers import AutoTokenizer, AutoConfig

MODEL_NAME = "tanaos/tanaos-text-anonymizer-v1"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print(f"Loading model info: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # 1. Export labels.json
    # The model config has id2label: {0: 'O', 1: 'B-EMAIL', ...}
    labels_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(config.id2label, f, indent=2)
    print(f"Exported labels to {labels_path}")

    # 2. Export tokenizer.json
    # We want the backend tokenizer format which contains the mapping
    # Note: Not all tokenizers support saving the fast format directory to a single JSON easily
    # easily readable by custom logic, but `tokenizer.save_pretrained` saves `tokenizer.json`
    # if it is a fast tokenizer.
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer_export")
    tokenizer.save_pretrained(tokenizer_path)
    
    # We specifically want the 'tokenizer.json' from that directory
    src_json = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(src_json):
        dst_json = os.path.join(OUTPUT_DIR, "tokenizer.json")
        with open(src_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Exported tokenizer to {dst_json}")
    else:
        print("Warning: tokenizer.json not found. This might be a slow tokenizer.")

if __name__ == "__main__":
    main()
