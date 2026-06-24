import torch
import os
import struct

def convert_pt_to_bin(pt_path, bin_path):
    print(f"Loading {pt_path}...")
    try:
        # Try loading as standard pytorch model
        try:
            tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
             # Try loading as jit script
            print(f"Standard load failed ({e}), trying jit.load...")
            tensor = torch.jit.load(pt_path, map_location='cpu')
            # Extract state dict or tensor if it's a module
            if hasattr(tensor, 'state_dict'):
                # Heuristic: look for the biggest tensor
                sd = tensor.state_dict()
                max_size = 0
                max_key = None
                for k, v in sd.items():
                    if v.numel() > max_size:
                        max_size = v.numel()
                        max_key = k
                if max_key:
                    tensor = sd[max_key]
                    print(f"Extracted tensor '{max_key}' from state_dict")
        
        # If it's still a module (e.g. from jit.load), try to find a parameter
        if isinstance(tensor, torch.jit.ScriptModule) or isinstance(tensor, torch.nn.Module):
             # Try to find the parameter that looks like an embedding
             for name, param in tensor.named_parameters():
                 print(f"Found param: {name} {param.shape}")
                 tensor = param # Just take the last one or the one that matches?
                 # ideally we want the weight
                 break
        
        # Ensure it's a float32 tensor
        if not isinstance(tensor, torch.Tensor):
            print(f"Error: Loaded object is {type(tensor)}, not a Tensor.")
            return

        if tensor.dtype != torch.float32:
            print(f"Converting {tensor.dtype} to float32...")
            tensor = tensor.to(torch.float32)
            
        print(f"Tensor shape: {tensor.shape}")
        
        # Flatten and convert to numpy
        data = tensor.detach().numpy().flatten()
        
        # Write to binary file
        print(f"Writing to {bin_path}...")
        with open(bin_path, 'wb') as f:
            f.write(data.tobytes())
            
        print(f"Successfully converted {pt_path} to {bin_path}")
        
    except Exception as e:
        print(f"Error converting {pt_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_dir = "/Users/jsn/Desktop/open_source/zetic_apps/ZETIC_MLange_apps/apps/QwenTextToSpeech/models"
    
    files_to_convert = [
        ("qwen3_tts_text_embedding.pt", "text_embedding.bin"),
        ("qwen3_tts_codec_embedding.pt", "codec_embedding.bin")
    ]
    
    for pt_file, bin_file in files_to_convert:
        pt_path = os.path.join(base_dir, pt_file)
        bin_path = os.path.join(base_dir, bin_file)
        
        if os.path.exists(pt_path):
            convert_pt_to_bin(pt_path, bin_path)
        else:
            print(f"File not found: {pt_path}")
