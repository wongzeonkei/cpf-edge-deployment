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


## Stage 2: TensorRT FP32 / FP16 Deployment

The CPF ONNX model was converted to TensorRT FP32 and FP16 engines and evaluated on an RTX 3090.

### TensorRT Engine Benchmark

| Backend | Precision | Source | Throughput | Mean Latency | GPU Compute | Engine Size | Context GPU Memory |
|---|---:|---|---:|---:|---:|---:|---:|
| TensorRT | FP32 | trtexec | 49.0207 qps | 20.6405 ms | 19.7640 ms | 216 MiB | 1905 MiB |
| TensorRT | FP16 | trtexec | 85.4206 qps | 11.9730 ms | 11.0956 ms | 104 MiB | 961 MiB |

### TensorRT Real Input Inference

The TensorRT engines were also tested with real CT/PET sample inputs.

| Backend | Mean Latency | Foreground Pixels |
|---|---:|---:|
| TensorRT FP32 | 21.621 ms | 408 |
| TensorRT FP16 | 13.064 ms | 408 |

Compared with FP32, the FP16 TensorRT engine reduced latency, engine size and context GPU memory usage.

### Generated Validation Files

- `docs/benchmarks/trt_benchmark_summary.csv`
- `docs/benchmarks/stage2_backend_summary.csv`

Large artifacts such as `.engine`, `.onnx`, `.pth`, inference outputs and TensorRT logs are excluded from Git.

## Stage 3: Jetson Nano B01 Edge Feasibility Validation

CPF was deployed to Jetson Nano B01 for edge feasibility validation.

### Jetson Environment

| Item | Version |
|---|---|
| Device | Jetson Nano B01 |
| L4T | R32.7.6 |
| CUDA | 10.2 |
| TensorRT | 8.2.1 |
| Python | 3.6.9 |

### Compatibility Issue

The original ONNX model failed to build on Jetson Nano B01 because TensorRT 8.2.1 does not support the exported `LayerNormalization` operator.

The original ONNX contained 105 `LayerNormalization` nodes. Re-exporting the model with opset 13 produced a Jetson-compatible ONNX model and enabled TensorRT FP16 engine construction.

### Jetson TensorRT FP16 Result

| Metric | Value |
|---|---:|
| Input size | 512×512 |
| Engine size | ~355 MiB |
| Mean latency | 1852.65 ms |
| Median latency | 1852.42 ms |
| P99 latency | 1869.86 ms |
| Throughput | 0.539481 qps |
| GPU compute mean | 1852.14 ms |
| Real-input foreground pixels | 408 |

### Resource Observation

During steady inference, RAM usage was about `2455 / 3964 MB`, SWAP stayed around `50 / 8126 MB`, and `GR3D_FREQ` reached 99%. This indicates that CPF inference on Jetson Nano B01 is mainly GPU-compute-bound.

### Conclusion

CPF can run on Jetson Nano B01 after ONNX opset13 re-export and TensorRT FP16 engine rebuilding. However, the 512×512 model is not suitable for real-time inference on Nano B01 due to about 1.85 seconds latency per image. This stage is therefore reported as complex-model edge feasibility validation rather than real-time deployment.

Generated validation file:

- `docs/benchmarks/jetson_nano_b01_cpf_benchmark.csv`

