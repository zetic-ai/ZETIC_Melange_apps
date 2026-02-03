# Melange Application Structure

Every app in this `apps/` folder follows the same structure, making it easy to navigate and understand how each one works.

## Directory Structure

Each app directory (e.g., `apps/<ModelName>/`) has this structure:

```
.
├── prepare/      # Scripts to prepare models and sample inputs
├── Android/      # Android app (Kotlin)
├── iOS/          # iOS app (Swift)
├── Flutter/      # Flutter implementation (Coming Soon)
└── ReactNative/  # React Native implementation (Coming Soon)
```

## Prepare Directory

The `prepare/` folder contains scripts to get your model ready for Melange. Usually you'll find:
- `prepare_model.py` or similar scripts to download models from HuggingFace, convert them if needed, and create sample inputs for testing

### Model Preparation Guide

Before you can use a model with **Melange**, you'll need to prepare the model file and sample inputs. For the full guide, check out the [official documentation](https://docs.zetic.ai/prepare-model).

#### 1. Supported Formats

Melange supports these model formats:
- **PyTorch Exported Program (`.pt2`)** ⭐ Recommended
- **ONNX (`.onnx`)**
- **TorchScript (`.pt`)** ⚠️ Deprecated

#### 2. Saving Model & Inputs

Save your model in one of the supported formats above, and save your sample inputs as NumPy arrays (`.npy`).

**Example: PyTorch Exported Program (.pt2)**
```python
import torch
import numpy as np

# (1) Export the model
torch_model = ...
dummy_input = torch.randn(...)
exported_program = torch.export.export(torch_model, (dummy_input,))
torch.export.save(exported_program, "model.pt2")

# (2) Save sample inputs
np.save("input.npy", dummy_input.detach().numpy())
```

**Example: ONNX (.onnx)**
```python
import torch
import numpy as np

# (1) Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", ...)

# (2) Save sample inputs
np.save("input.npy", dummy_input.detach().numpy())
```

#### 3. Verify Inputs

Melange compiles models into a static hardware graph, so **input order** and **input shapes** need to be consistent and fixed. Always double-check your `.onnx` or `.pt2` model structure using tools like [Netron](https://netron.app/).

## Platform Directories

- **Android/** - Full Gradle-based Android project ready to open in Android Studio
- **iOS/** - Complete Xcode project ready to build and run
- **Flutter/** / **ReactNative/** - Cross-platform implementations (coming soon)

Each platform directory contains a complete, runnable app that uses the Melange SDK to run the model with NPU acceleration.
