# openunmix-torchscript

Open-Unmix models for **mobile on-device inference** — TorchScript (`.ptl`) and ExecuTorch (`.pte`) export pipeline.

## Install dependencies

```sh
uv sync
```

To also generate Executorch models:

```sh
uv sync --extra executorch
```

## Create the PyTorch Mobile model

```sh
uv run python openunmix_executorch.py --umxhq --ptl
uv run python openunmix_executorch.py --umxl --ptl
```

Output: `dist/umxhq.ptl` (~140 MB) and `dist/umxl.ptl` (~290 MB).

## Create the ExecuTorch model (`.pte`)

```sh
uv run python openunmix_executorch.py --umxhq --pte
```

> ⚠️ The original OpenUnmix model uses dynamic frame indexing that
> `torch.export` cannot trace. See [NOTES.md](NOTES.md) for how to
> modify the model for `.pte` export. The `.ptl` export works without
> modification and is the recommended path for now.

### Options

```
uv run python openunmix_executorch.py --all           # .ptl + .pte
uv run python openunmix_executorch.py --ptl           # TorchScript only (default, works)
uv run python openunmix_executorch.py --pte           # ExecuTorch (needs model modification)
uv run python openunmix_executorch.py --umxhq         # Only umxhq model
uv run python openunmix_executorch.py --umxl          # Only umxl model
```

## Download pre-converted models

You can find the latest releases in the [GitHub releases](https://github.com/demixr/openunmix-torchscript/releases/latest).

## Model formats comparison

| Feature | `.ptl` (TorchScript) | `.pte` (ExecuTorch) |
|---------|---------------------|---------------------|
| Model modification | ❌ Not needed | ✅ Required (see NOTES.md) |
| GPU acceleration | Limited | MPS (Metal), CoreML |
| Quantization | 8-bit dynamic | 8-bit, 4-bit, dynamic |
| Flutter plugin | None (custom) | `executorch_flutter` |
| iOS simulator | ✅ | ❌ (device only) |
| macOS Intel | ✅ | ❌ (Apple Silicon) |

## License

MIT — same as [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch).
