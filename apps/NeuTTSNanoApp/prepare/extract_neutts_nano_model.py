#!/usr/bin/env python3
"""
Script to trace neuphonic/neutts-nano to TorchScript format.

This script:
1. Sets up the directory structure: model_zoo/neutts_nano/{model, inputs}
2. Loads the model from Hugging Face
3. Creates a wrapper for Causal LM
4. Traces it to TorchScript format (.pt)
5. Saves sample input tokens as .npy files
"""

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def main():
    # ---------------------------------------------------------
    # 1. Setup Directories
    # ---------------------------------------------------------
    project_name = "neutts_nano"
    
    # Create structure relative to current working directory
    # Default to a local path to avoid the model_zoo symlinked disk.
    model_zoo_root = os.environ.get(
        "MODEL_ZOO_DIR", "/home/jsn/zetic_mentat/model_zoo_local"
    )
    base_dir = os.path.join(model_zoo_root, project_name)
    model_dir = os.path.join(base_dir, "model")
    inputs_dir = os.path.join(base_dir, "inputs")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)
    
    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Model directory: {model_dir}")
    print(f"[INFO] Inputs directory: {inputs_dir}")

    # ---------------------------------------------------------
    # 2. Load Model and Tokenizer
    # ---------------------------------------------------------
    model_id = "neuphonic/neutts-nano"
    print(f"\n[INFO] Loading model: {model_id}...")

    try:
        # Load Config with trust_remote_code=True
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Force eager implementation for easier tracing
        config._attn_implementation = "eager"
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load Model
        # Using float32 for stable tracing, though the model might be float16/bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print(f"[INFO] ✓ Model loaded: {type(model).__name__}")
        print(f"[INFO] Config: {model.config}")

    except Exception as e:
        print(f"[ERROR] Failed to load model. Error: {e}")
        exit(1)

    # ---------------------------------------------------------
    # 3. Model Wrapper
    # ---------------------------------------------------------
    class NeuttsNanoWrapper(nn.Module):
        """
        Wraps the Causal LM model to return raw logits.
        """
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            """
            Args:
                input_ids: [batch, seq_len]
                attention_mask: [batch, seq_len]
            Returns:
                logits: [batch, seq_len, vocab_size]
            """
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True
            )
            return outputs.logits

    # ---------------------------------------------------------
    # 4. Prepare Inputs
    # ---------------------------------------------------------
    print(f"\n[INFO] Preparing input tokens...")
    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog."

    FIXED_SEQ_LEN = 128
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length", 
        max_length=FIXED_SEQ_LEN,
        truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"[INFO] Input shapes: ids={input_ids.shape}, mask={attention_mask.shape}")

    # Save inputs as .npy
    input_ids_path = os.path.join(inputs_dir, "input_ids.npy")
    attention_mask_path = os.path.join(inputs_dir, "attention_mask.npy")
    
    # Save as Int32 (Compatible with most mobile backends)
    np.save(input_ids_path, input_ids.numpy().astype(np.int32))
    np.save(attention_mask_path, attention_mask.numpy().astype(np.int32))
    print(f"[INFO] ✓ Inputs saved to {inputs_dir}")

    # ---------------------------------------------------------
    # 5. Trace and Save (.pt)
    # ---------------------------------------------------------
    print(f"\n[INFO] Tracing model to TorchScript...")
    wrapped_model = NeuttsNanoWrapper(model).eval()

    try:
        with torch.no_grad():
            # Trace the model
            # We use a tuple for multiple inputs
            traced_model = torch.jit.trace(
                wrapped_model,
                (input_ids, attention_mask),
                strict=False
            )
        
        # Save the traced model
        pt_filename = f"{project_name}.pt"
        pt_path = os.path.join(model_dir, pt_filename)
        
        torch.jit.save(traced_model, pt_path)
        print(f"[INFO] ✓ TorchScript model saved to {pt_path}")
        
        # Verify reload
        print(f"[INFO] Verifying reload...")
        test_model = torch.jit.load(pt_path)
        test_output = test_model(input_ids, attention_mask)
        print(f"[INFO] ✓ Inference successful. Output shape: {test_output.shape}")
        
        # ---------------------------------------------------------
        # 6. Print Run Command
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("READY TO RUN CONVERSION")
        print("="*60)
        print("Copy and run this command:")
        print(f"python tools/run_convert_all.py \\")
        print(f"  {pt_path} \\")
        print(f"  {input_ids_path},{attention_mask_path} \\")
        print(f"  {base_dir}/output_results")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] Failed to trace model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()



