import os
import time
import cv2
import yaml
import numpy as np
import onnxruntime as ort


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gray_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.resize(img, (size[1], size[0]))
    img = img.astype(np.float32)

    # TODO: keep consistent with PyTorch preprocessing
    img = img / 255.0 * 3.2 - 1.6

    img = img[None, None, :, :]  # [1, 1, H, W]
    return img


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    cfg = load_config("configs/cpf_infer.yaml")

    onnx_path = "deploy/onnx/models/cpf_static.onnx"
    save_dir = "outputs/onnx_infer"
    os.makedirs(save_dir, exist_ok=True)

    size = cfg["model"]["input_size"]
    threshold = float(cfg["model"]["threshold"])

    ct = load_gray_image(cfg["data"]["sample_ct"], size)
    pet = load_gray_image(cfg["data"]["sample_pet"], size)

    ct = np.repeat(ct, 3, axis=1).astype(np.float32)  # [1, 3, H, W]
    pet = pet.astype(np.float32)                      # [1, 1, H, W]

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    sess = ort.InferenceSession(onnx_path, providers=providers)

    print("ONNX Runtime providers:", sess.get_providers())
    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}, {inp.type}")

    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape}, {out.type}")

    # warmup
    for _ in range(10):
        _ = sess.run(["logits"], {"ct": ct, "pet": pet})

    times = []
    for _ in range(50):
        start = time.time()
        logits = sess.run(["logits"], {"ct": ct, "pet": pet})[0]
        end = time.time()
        times.append((end - start) * 1000)

    prob = sigmoid(logits)
    mask = (prob >= threshold).astype(np.uint8)

    mask_png = mask[0, 0] * 255
    save_path = os.path.join(save_dir, "pred_mask_onnx.png")
    cv2.imwrite(save_path, mask_png)

    print("ONNX inference done.")
    print(f"CT shape: {ct.shape}")
    print(f"PET shape: {pet.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Threshold: {threshold}")
    print(f"Latency mean: {np.mean(times):.3f} ms")
    print(f"Latency median: {np.median(times):.3f} ms")
    print(f"Latency min: {np.min(times):.3f} ms")
    print(f"Latency max: {np.max(times):.3f} ms")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
