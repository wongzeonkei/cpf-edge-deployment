import os
import sys
import yaml
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infer_torch import load_gray_image


def main():
    with open("configs/cpf_infer.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    size = cfg["model"]["input_size"]

    save_dir = "outputs/cpp_input"
    os.makedirs(save_dir, exist_ok=True)

    ct_1ch = load_gray_image(cfg["data"]["sample_ct"], size)
    pet = load_gray_image(cfg["data"]["sample_pet"], size)

    ct = np.repeat(ct_1ch, 3, axis=1).astype(np.float32)
    pet = pet.astype(np.float32)

    ct.tofile(os.path.join(save_dir, "ct.bin"))
    pet.tofile(os.path.join(save_dir, "pet.bin"))

    print("Saved C++ input tensors:")
    print(f"  {save_dir}/ct.bin  shape={ct.shape}, dtype=float32")
    print(f"  {save_dir}/pet.bin shape={pet.shape}, dtype=float32")


if __name__ == "__main__":
    main()
