#!/usr/bin/env python3
"""
Export OpenUnmix models to TorchScript (.ptl) and/or ExecuTorch (.pte) format.

TorchScript (.ptl):
  - Works out of the box with the original OpenUnmix model
  - Uses torch.jit.script + mobile optimizer
  - Larger binary, limited GPU support

ExecuTorch (.pte):
  - Requires a modified model (no data-dependent control flow)
  - The original OpenUnmix model uses dynamic frame indexing
    (int(cur_frame[-1]) + 1) that torch.export cannot trace
  - See NOTES.md for how to modify the model for .pte export

Usage:
    python openunmix_executorch.py --all     # Generate both .ptl and .pte
    python openunmix_executorch.py --ptl     # TorchScript only (default, works)
    python openunmix_executorch.py --pte     # ExecuTorch (needs model modification)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile

import openunmix


def create_separator(model_fn, device="cpu"):
    """Create separator from built-in OpenUnmix model."""
    return model_fn(device=device, pretrained=True).eval().to(device)


def quantize_model(model):
    """Quantize model dynamically (8-bit quantization for mobile).

    Returns the original model if no quantized engine is available
    (e.g., Apple Silicon without QNNPACK).
    """
    try:
        model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        print("  (dynamic 8-bit quantization applied)")
    except RuntimeError as e:
        if "QEngine" in str(e):
            print(f"  (skipped quantization: {e})")
        else:
            raise
    return model


def fix_numpy_attrs(module):
    """Recursively convert numpy scalar attributes to Python ints for TorchScript.

    Handles nested modules (e.g., Linear layers inside LSTM) by using
    named_modules() which recurses into all submodules.
    """
    for name, child in module.named_modules():
        for attr_name in dir(child):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(child, attr_name)
                # Check if it's a numpy scalar (not a valid TorchScript constant)
                if hasattr(val, "item") and not isinstance(
                    val, (int, float, str, bool, type(None))
                ):
                    try:
                        setattr(child, attr_name, int(val))
                    except (ValueError, TypeError):
                        pass  # Skip non-convertible attributes
            except (AttributeError, RuntimeError):
                pass  # Skip properties that raise on access
    return module


def create_script(model_name, separator):
    """Create the TorchScript model (.ptl) from a separator."""
    # Fix numpy scalar attributes that TorchScript can't handle
    separator = fix_numpy_attrs(separator)
    jit_script = torch.jit.script(separator)

    # optimize_for_mobile was removed in PyTorch 2.x;
    # mobile optimization is now automatic.
    try:
        optimized = torch.utils.mobile_optimizer.optimize_for_mobile(jit_script)
    except AttributeError:
        optimized = jit_script  # PyTorch 2.x — no-op

    output_path = f"dist/{model_name}.ptl"
    optimized._save_for_lite_interpreter(output_path)
    print(f"  ✅ Saved .ptl: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def create_executorch_model(model_name, separator):
    """
    Export the Separator model to ExecuTorch .pte format.

    NOTE: The original OpenUnmix model uses dynamic frame indexing
    (int(cur_frame[-1]) + 1) that torch.export cannot trace.
    To generate .pte files, you must modify the model to:
      1. Use fixed-size tensor operations instead of dynamic indexing
      2. Replace the frame-based processing with a fully traceable version

    See NOTES.md for the required modifications.

    Pipeline:
      1. torch.export.export()  → ExportedProgram
      2. to_edge()              → EdgeProgram (IR)
      3. to_executorch()        → ExecutorchProgram (runtime binary)
      4. Write .pte file
    """
    try:
        from executorch.exir import to_edge
    except ImportError:
        print("  ❌ executorch not installed. Install with:")
        print("      pip install 'openunmix-torchscript[executorch]'")
        print("  Or:  pip install executorch")
        return

    device = "cpu"
    separator = separator.to(device)

    # The Separator expects (batch, channels, time) — stereo audio
    nb_channels = 2
    sample_rate = 44100.0
    duration = 3.0  # 3 seconds
    num_samples = int(sample_rate * duration)

    example_input = torch.randn(1, nb_channels, num_samples)

    print(f"  📐 Example input shape: {tuple(example_input.shape)} "
          f"(batch=1, channels={nb_channels}, samples={num_samples})")

    print("  🔧 Step 1: torch.export.export() ...")
    try:
        exported_program = torch.export.export(separator.eval(), (example_input,))
    except Exception as e:
        print(f"  ❌ Export failed: {type(e).__name__}")
        print()
        print("  The original OpenUnmix model uses dynamic frame indexing")
        print("  (int(cur_frame[-1]) + 1) that torch.export cannot trace.")
        print()
        print("  To generate .pte files, modify the model to use fixed-size")
        print("  tensor operations. See NOTES.md for details.")
        print()
        print("  The .ptl (TorchScript) export works without modification.")
        return

    print("  🔧 Step 2: to_edge() ...")
    edge_program = to_edge(exported_program)

    print("  🔧 Step 3: to_executorch() ...")
    et_program = edge_program.to_executorch()

    output_path = f"dist/{model_name}.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    file_size = os.path.getsize(output_path)
    print(f"  ✅ Saved .pte: {output_path} ({file_size / 1e6:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenUnmix models to .ptl and/or .pte format"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate both .ptl and .pte (default: .ptl only for unmodified model)"
    )
    parser.add_argument("--ptl", action="store_true",
                        help="Generate only .ptl (TorchScript, works with original model)")
    parser.add_argument("--pte", action="store_true",
                        help="Generate only .pte (ExecuTorch, needs modified model)")
    parser.add_argument(
        "--umxhq", action="store_true", default=True,
        help="Generate umxhq model (default)"
    )
    parser.add_argument(
        "--umxl", action="store_true", default=True,
        help="Generate umxl model (default)"
    )
    parser.add_argument(
        "--output-dir", default="dist",
        help="Output directory (default: dist/)"
    )
    args = parser.parse_args()

    # Default: generate .ptl only (works with original model)
    if not args.ptl and not args.pte and not args.all:
        args.ptl = True  # default: .ptl only

    generate_ptl = args.all or args.ptl
    generate_pte = args.all or args.pte

    if not generate_ptl and not generate_pte:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  OpenUnmix → Mobile Model Export")
    print("=" * 60)

    models = {
        "umxhq": lambda: create_separator(openunmix.umxhq),
        "umxl": lambda: create_separator(openunmix.umxl),
    }

    for name, model_fn in models.items():
        if not (args.umxhq and name == "umxhq") and not (args.umxl and name == "umxl"):
            continue

        print(f"\n{'─' * 40}")
        print(f"  Model: {name}")
        print(f"{'─' * 40}")

        separator = model_fn()

        if generate_ptl:
            print(f"\n  📦 Generating .ptl (TorchScript)...")
            separator_quantized = quantize_model(separator)
            create_script(name, separator_quantized)

        if generate_pte:
            print(f"\n  📦 Generating .pte (ExecuTorch)...")
            create_executorch_model(name, separator)

    print(f"\n{'=' * 60}")
    print(f"  Done! Models saved to: dist/")
    print(f"{'=' * 60}")

    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, f)
        size = os.path.getsize(path)
        print(f"  {f:20s} {size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
