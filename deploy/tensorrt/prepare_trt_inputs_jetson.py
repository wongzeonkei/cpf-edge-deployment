import os
import cv2
import numpy as np


def load_gray(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = img / 255.0 * 3.2 - 1.6
    img = img[None, None, :, :]  # [1, 1, H, W]
    return img


def main():
    out_dir = "outputs/jetson_trt/input_bins"
    os.makedirs(out_dir, exist_ok=True)

    ct_1ch = load_gray("data/samples/ct.png", 512)
    pet = load_gray("data/samples/pet.png", 512)

    ct = np.repeat(ct_1ch, 3, axis=1).astype(np.float32)
    pet = pet.astype(np.float32)

    ct_path = os.path.join(out_dir, "ct_fp32_1x3x512x512.bin")
    pet_path = os.path.join(out_dir, "pet_fp32_1x1x512x512.bin")

    ct.tofile(ct_path)
    pet.tofile(pet_path)

    print("Saved TensorRT input bins:")
    print(f"  {ct_path}  shape={ct.shape}, dtype={ct.dtype}, bytes={ct.nbytes}")
    print(f"  {pet_path} shape={pet.shape}, dtype={pet.dtype}, bytes={pet.nbytes}")


if __name__ == "__main__":
    main()
