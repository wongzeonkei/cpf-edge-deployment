# Jetson Nano B01 CPF Deployment Notes

## Environment

- Device: Jetson Nano B01
- L4T: R32.7.6
- CUDA: 10.2
- TensorRT: 8.2.1
- Python: 3.6.9
- Storage: project placed under `/home/nano/sdcard`

## Original ONNX Failure

The original `cpf_static.onnx` failed to build a TensorRT engine on Jetson Nano B01.

Main error:

```text
No importer registered for op: LayerNormalization.
Plugin not found.
Failed to parse onnx file.
Failed to create engine from model.

The original ONNX model contained 105 LayerNormalization nodes. TensorRT 8.2.1 on Jetson Nano B01 does not support this operator directly.

Fix

A Jetson-compatible ONNX model was re-exported with opset 13:

cpf_static_opset13.onnx

This resolved the LayerNormalization compatibility issue and allowed TensorRT FP16 engine construction on Jetson Nano B01.

TensorRT Engine
Engine: cpf_fp16_512_opset13_jetson.engine
Input shape:
CT: [1, 3, 512, 512]
PET: [1, 1, 512, 512]
Engine size: about 355 MiB
Build status: success
trtexec Benchmark
Mean latency: 1852.65 ms
Median latency: 1852.42 ms
P99 latency: 1869.86 ms
Throughput: 0.539481 qps
GPU compute mean: 1852.14 ms
H2D latency mean: 0.407202 ms
D2H latency mean: 0.10752 ms
tegrastats Observation

During steady inference:

RAM usage: about 2455 / 3964 MB
SWAP usage: about 50 / 8126 MB
GR3D_FREQ: about 99%
GPU temperature: about 31–32.5°C
CPU usage: low after initialization

The model is mainly GPU-compute-bound on Jetson Nano B01.

Real CT/PET Input Inference

Real CT/PET input was tested through trtexec --loadInputs.

Output:

Logits shape: [1, 1, 512, 512]
Logits min/max: -12.5312 / 7.5
Binary mask shape: [512, 512]
Foreground pixels: 408
Conclusion

CPF can be deployed and executed on Jetson Nano B01 with TensorRT FP16 after ONNX opset13 re-export. However, the 512×512 model has about 1.85 seconds latency per image, so it is not suitable for real-time inference on Jetson Nano B01.

This result should be described as a complex medical segmentation model edge feasibility analysis rather than a real-time deployment.
