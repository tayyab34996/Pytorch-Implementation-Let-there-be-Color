# colorization_pytorch

PyTorch port of `colorization_tensorflow` (aims to match original implementation
and behavior). Files are placed in `colorization_pytorch/SOURCE` and mirror the
original `SOURCE` API and filenames.

How to use

1. Install dependencies:

```powershell
cd colorization_pytorch
pip install -r requirements.txt
```

2. Run training (same flow as original):

```powershell
python SOURCE\main.py
```

Notes
- This port keeps the same architecture, activations and training loop structure
  as the original TensorFlow implementation. It shares the original repo's
  `DATASET`, `RESULT`, `MODEL`, and `LOGS` directories so no dataset reorganization
  is required.
