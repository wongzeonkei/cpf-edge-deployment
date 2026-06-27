# CPF Edge Deployment

This repository records my learning and engineering practice for deploying a PET/CT lung tumor segmentation model from PyTorch to ONNX, ONNX Runtime, TensorRT, and edge devices.

## Goal

- PyTorch inference
- ONNX export
- ONNX Runtime inference
- TensorRT FP32/FP16/INT8 deployment
- C++ inference
- Jetson / RK3588 edge deployment
- Latency / FPS / memory / Dice / IoU / HD95 evaluation

## Current Stage

Stage 0: Environment setup and PyTorch inference.

## Project Structure

- `configs/`: inference configuration files.
- `docs/`: environment setup notes and learning logs.
- `models/`: CPF model structure.
- `scripts/`: PyTorch environment check and inference scripts.
- `deploy/onnx/`: ONNX export, ONNX Runtime Python inference, and comparison scripts.
- `deploy/cpp/`: ONNX Runtime C++ inference demo.
- `tools/`: utility scripts.
- `assets/`: project assets.
- `tests/`: test scripts.

## Stage 1: PyTorch to ONNX Runtime

### ONNX Export

The CPF model was exported from PyTorch to ONNX with two inputs.

| Name | Shape | Description |
|---|---:|---|
| `ct` | `[batch, 3, 512, 512]` | CT image repeated to 3 channels |
| `pet` | `[batch, 1, 512, 512]` | PET image as metabolic prior input |
| `logits` | `[batch, 1, 512, 512]` | Segmentation logits |

The exported model was checked by `onnx.checker` and inspected with Netron.

### Preprocessing

CPF uses the same preprocessing as the original training pipeline:

- Image preprocessing: `img = img / 255.0 * 3.2 - 1.6`
- Probability map: `prob = sigmoid(logits)`
- Binary mask threshold: `mask = prob >= 0.35`

### ONNX Runtime Python Inference

ONNX Runtime was tested with `CUDAExecutionProvider`.

Small positive-sample validation was conducted on 5 PET/CT slices.

| Metric | PyTorch | ONNX Runtime |
|---|---:|---:|
| Dice vs GT | 0.9132 | 0.8751 |
| IoU vs GT | 0.8409 | 0.7784 |
| Mean latency | 76.75 ms | 50.43 ms |

Torch/ONNX consistency:

| Metric | Value |
|---|---:|
| Torch/ONNX Dice | 0.9186 |
| Torch/ONNX IoU | 0.8501 |
| Mean mask difference ratio | 0.000611 |
| ONNX speedup | 1.52x |

Notes:

- The validation above is a small positive-sample deployment sanity check, not a full test-set evaluation.
- ONNX Runtime produces highly similar binary masks, while logits show non-negligible numerical differences.
- Large artifacts such as `.pth`, `.onnx`, `.engine`, `.so`, medical images and inference outputs are excluded from Git.

### ONNX Runtime C++ Inference

A minimal C++ ONNX Runtime inference demo was added.

Current C++ pipeline:

1. Python preprocessing.
2. Save `ct.bin` and `pet.bin`.
3. C++ ONNX Runtime inference.
4. Save `mask_cpp.pgm`.

The first C++ version uses preprocessed binary tensors as input.

| Input | Shape | File |
|---|---:|---|
| CT | `[1, 3, 512, 512]` | `outputs/cpp_input/ct.bin` |
| PET | `[1, 1, 512, 512]` | `outputs/cpp_input/pet.bin` |

C++ inference output:

| Output | Shape | Description |
|---|---:|---|
| `mask_cpp.pgm` | `[512, 512]` | Binary segmentation mask |

Current C++ result:

| Runtime | Provider | Latency |
|---|---|---:|
| ONNX Runtime Python | CUDAExecutionProvider | ~50 ms |
| ONNX Runtime C++ | CPU package | ~4686 ms |

Python ONNX and C++ ONNX generated identical binary masks on the tested sample.

| Metric | Value |
|---|---:|
| Python ONNX foreground pixels | 408 |
| C++ ONNX foreground pixels | 408 |
| Difference pixels | 0 |
| Difference ratio | 0.0 |

Notes:

- The current C++ demo uses the CPU ONNX Runtime release package, so its latency is not comparable to Python CUDA inference.
- This step verifies the C++ deployment workflow.
- CUDA/TensorRT C++ acceleration will be added in later stages.

## Stage 1 Completion Status

Stage 1 is completed.

Completed items:

- PyTorch model exported to ONNX.
- ONNX graph checked and inspected.
- ONNX Runtime Python inference completed.
- PyTorch and ONNX outputs compared with Dice, IoU, mask difference ratio and latency.
- Multi-sample deployment sanity check completed.
- ONNX Runtime C++ inference demo completed.
- Python ONNX and C++ ONNX binary masks matched on the tested sample.

