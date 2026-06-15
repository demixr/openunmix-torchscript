# Executorch (.pte) Export Notes

## The Problem

The original OpenUnmix model cannot be exported to ExecuTorch (`.pte`) format
because it uses **data-dependent control flow** that `torch.export` cannot trace.

Specifically, in `openunmix/model.py` line ~301:

```python
pos = int(cur_frame[-1]) + 1  # ← data-dependent integer
```

This creates a `SymInt` (symbolic integer) that `torch.export` cannot
specialize, causing the export to fail with:

```
GuardOnDataDependentSymNode: Could not extract specialized integer
from data-dependent expression u0 (unhinted: u0).
```

## The Fix

To export to `.pte`, you need to modify the OpenUnmix `Separator` class to
use **fixed-size tensor operations** instead of dynamic frame indexing.

### Option 1: Use a fully traceable model architecture

Replace the frame-based processing with a version that:
1. Uses fixed-size STFT windows (no dynamic frame boundaries)
2. Uses tensor slicing instead of index-based frame access
3. Processes the full audio as a single batch (no chunked frame iteration)

Example pattern:
```python
# Instead of:
pos = int(cur_frame[-1]) + 1  # data-dependent

# Use:
pos = torch.tensor(current_frame_idx, device=audio.device)  # static
```

### Option 2: Use TorchScript (.ptl) — works out of the box

The `.ptl` (TorchScript) export works without modification:

```bash
python openunmix_executorch.py --umxhq --ptl
python openunmix_executorch.py --umxl --ptl
```

This is the recommended approach for now. The `.ptl` format:
- ✅ Works with the original OpenUnmix model
- ✅ Dynamic quantization (8-bit) reduces size
- ✅ `optimize_for_mobile()` applies mobile-specific optimizations

The trade-off is:
- ❌ No MPS/CoreML GPU acceleration
- ❌ Larger binary size
- ❌ No Executorch Flutter plugin (`executorch_flutter`)

## Model Size Comparison

| Model | Format | Size | GPU | Flutter Plugin |
|-------|--------|------|-----|----------------|
| umxhq | .ptl | ~140 MB | No | None (custom) |
| umxhq | .pte | ~140 MB | MPS/CoreML | `executorch_flutter` |
| umxl  | .ptl | ~290 MB | No | None (custom) |
| umxl  | .pte | ~290 MB | MPS/CoreML | `executorch_flutter` |

## Recommendation

For your Flutter app (`demixr-app`), the `.ptl` path is the pragmatic choice:
- Your existing `DemixingPlugin.swift` would need a TorchScript backend
- No model modification required
- Works on all devices (including iOS simulator, Intel Macs)

The `.pte` path requires:
1. Modifying the OpenUnmix model (non-trivial)
2. Using `executorch_flutter` (already in your pubspec)
3. Apple Silicon / iOS device only
