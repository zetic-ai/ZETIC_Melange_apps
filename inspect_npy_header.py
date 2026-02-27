import os
import struct
import ast

def read_npy_header(path):
    print(f"--- {os.path.basename(path)} ---")
    try:
        with open(path, 'rb') as f:
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                print("Not a NPY file")
                return
            
            major = f.read(1)
            minor = f.read(1)
            header_len_bytes = f.read(2)
            header_len = struct.unpack('<H', header_len_bytes)[0]
            
            header_str = f.read(header_len).decode('ascii').strip()
            # The header is an eval-able dictionary string
            header = ast.literal_eval(header_str)
            
            print(f"Shape: {header.get('shape')}")
            print(f"Descr: {header.get('descr')}")
            print(f"Fortran: {header.get('fortran_order')}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    base_dir = "apps/QwenTextToSpeech/models"
    files = [f for f in os.listdir(base_dir) if f.endswith(".npy")]
    for f in files:
        read_npy_header(os.path.join(base_dir, f))

if __name__ == "__main__":
    main()
