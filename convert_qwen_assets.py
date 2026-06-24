import os
import struct
import sys

# Try imports
try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback or exit if crucial
    pass

def inspect_npy(path):
    print(f"--- Inspecting {path} ---")
    try:
        data = np.load(path)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {data.min()}, Max: {data.max()}")
        return data
    except Exception as e:
        print(f"Error loading npy: {e}")
        return None

def convert_pt_to_bin(pt_path, bin_path):
    print(f"--- Converting {pt_path} to {bin_path} ---")
    try:
        # Load PT
        # Assuming mapped or simple tensor
        if not os.path.exists(pt_path):
            print("File not found.")
            return

        # Attempt to load with torch
        tensor = torch.load(pt_path, map_location='cpu')
        
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().numpy()
        elif isinstance(tensor, dict):
            # Maybe it's a state dict? find the weight
            print(f"Keys: {tensor.keys()}")
            # heuristic
            key = next((k for k in tensor.keys() if 'weight' in k or 'embed' in k), None)
            if key:
                data = tensor[key].numpy()
            else:
                print("Could not find tensor in dict.")
                return
        else:
            print(f"Unknown type: {type(tensor)}")
            return

        print(f"Tensor Shape: {data.shape}, Dtype: {data.dtype}")
        
        # Save as raw binary (C-order)
        # Ensure float32 for simplicity in Swift unless fp16 is requested
        if data.dtype != np.float32:
            print("Converting to float32...")
            data = data.astype(np.float32)
            
        data.tofile(bin_path)
        print(f"Saved {bin_path}")
        
    except Exception as e:
        print(f"Error converting pt: {e}")

def main():
    base_dir = "apps/QwenTextToSpeech/models"
    
    # 1. Inspect NPYs
    npy_files = [f for f in os.listdir(base_dir) if f.endswith(".npy")]
    for f in npy_files:
        inspect_npy(os.path.join(base_dir, f))
        
    # 2. Convert PTs
    pt_files = [f for f in os.listdir(base_dir) if f.endswith(".pt")]
    for f in pt_files:
        bin_name = f.replace(".pt", ".bin")
        convert_pt_to_bin(os.path.join(base_dir, f), os.path.join(base_dir, bin_name))

if __name__ == "__main__":
    main()
