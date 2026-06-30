import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_gray_image(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = img / 255.0 * 3.2 - 1.6
    img = img[None, None, :, :]  # [1, 1, H, W]
    return img


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

    return engine


def get_binding_index(engine, name):
    for i in range(engine.num_bindings):
        if engine.get_binding_name(i) == name:
            return i
    raise KeyError(f"Binding name not found: {name}")


def allocate_buffers(engine, context, input_arrays):
    bindings = [None] * engine.num_bindings
    host_inputs = {}
    device_inputs = {}
    host_outputs = {}
    device_outputs = {}
    output_shapes = {}

    # Set dynamic shapes if needed.
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            shape = tuple(input_arrays[name].shape)
            engine_shape = tuple(engine.get_binding_shape(i))
            if any(dim < 0 for dim in engine_shape):
                context.set_binding_shape(i, shape)

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))

        if engine.binding_is_input(i):
            arr = np.ascontiguousarray(input_arrays[name].astype(dtype))
            host_inputs[name] = arr
            device_mem = cuda.mem_alloc(arr.nbytes)
            device_inputs[name] = device_mem
            bindings[i] = int(device_mem)
        else:
            shape = tuple(context.get_binding_shape(i))
            output_shapes[name] = shape
            size = int(np.prod(shape))
            host_mem = np.empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            host_outputs[name] = host_mem
            device_outputs[name] = device_mem
            bindings[i] = int(device_mem)

    return bindings, host_inputs, device_inputs, host_outputs, device_outputs, output_shapes


def infer_once(context, bindings, host_inputs, device_inputs, host_outputs, device_outputs, stream):
    for name, host_arr in host_inputs.items():
        cuda.memcpy_htod_async(device_inputs[name], host_arr, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    for name, host_arr in host_outputs.items():
        cuda.memcpy_dtoh_async(host_arr, device_outputs[name], stream)

    stream.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpf_infer.yaml")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logits-output", default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_size = cfg["model"]["input_size"]
    if isinstance(input_size, (list, tuple)):
        if len(input_size) != 2:
            raise ValueError(f"input_size list must have length 2, got: {input_size}")
        h, w = int(input_size[0]), int(input_size[1])
        if h != w:
            raise ValueError(f"Current TensorRT static engine expects square input, got: {input_size}")
        size = h
    else:
        size = int(input_size)

    ct_1ch = load_gray_image(cfg["data"]["sample_ct"], size)
    pet = load_gray_image(cfg["data"]["sample_pet"], size)

    ct = np.repeat(ct_1ch, 3, axis=1).astype(np.float32)
    pet = pet.astype(np.float32)

    input_arrays = {
        "ct": ct,
        "pet": pet,
    }

    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    bindings, host_inputs, device_inputs, host_outputs, device_outputs, output_shapes = allocate_buffers(
        engine, context, input_arrays
    )

    stream = cuda.Stream()

    for _ in range(args.warmup):
        infer_once(context, bindings, host_inputs, device_inputs, host_outputs, device_outputs, stream)

    times = []
    for _ in range(args.repeat):
        start = time.time()
        infer_once(context, bindings, host_inputs, device_inputs, host_outputs, device_outputs, stream)
        end = time.time()
        times.append((end - start) * 1000.0)

    # The model has one output named logits.
    if "logits" in host_outputs:
        output_name = "logits"
    else:
        output_name = list(host_outputs.keys())[0]

    logits = host_outputs[output_name].reshape(output_shapes[output_name])
    prob = sigmoid(logits)
    mask = (prob >= args.threshold).astype(np.uint8) * 255
    mask_2d = mask.squeeze()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, mask_2d)

    if args.logits_output:
        os.makedirs(os.path.dirname(args.logits_output), exist_ok=True)
        np.save(args.logits_output, logits.astype(np.float32))

    print("TensorRT inference done.")
    print(f"Engine: {args.engine}")
    print(f"CT shape: {ct.shape}")
    print(f"PET shape: {pet.shape}")
    print(f"Output name: {output_name}")
    print(f"Logits shape: {logits.shape}")
    print(f"Threshold: {args.threshold}")
    print(f"Latency mean: {np.mean(times):.3f} ms")
    print(f"Latency median: {np.median(times):.3f} ms")
    print(f"Latency min: {np.min(times):.3f} ms")
    print(f"Latency max: {np.max(times):.3f} ms")
    print(f"Foreground pixels: {int((mask_2d > 0).sum())}")
    print(f"Saved mask to: {args.output}")
    if args.logits_output:
        print(f"Saved logits to: {args.logits_output}")


if __name__ == "__main__":
    main()
