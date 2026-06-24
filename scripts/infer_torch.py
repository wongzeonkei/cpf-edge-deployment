import os
import time
import yaml
import torch
import numpy as np
import cv2


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.resize(img, tuple(size[::-1]))
    img = img.astype(np.float32)

    # TODO: replace with your real normalization
    img = img / 255.0

    img = img[None, None, :, :]  # [1, 1, H, W]
    return img


def build_model():
    # TODO: import your CPF model here
    # from models.cpf.build_model import build_cpf
    # model = build_cpf()
    raise NotImplementedError("Please implement CPF model loading.")


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    cfg = load_config("configs/cpf_infer.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["model"]["input_size"]
    threshold = cfg["model"]["threshold"]

    ct = load_image(cfg["data"]["sample_ct"], size)
    pet = load_image(cfg["data"]["sample_pet"], size)

    x = np.concatenate([ct, pet], axis=1)  # [1, 2, H, W]
    x = torch.from_numpy(x).to(device)

    model = build_model()
    model = load_checkpoint(model, cfg["model"]["checkpoint"])
    model = model.to(device)
    model.eval()

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        pred = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    if isinstance(pred, (tuple, list)):
        pred = pred[-1]

    prob = torch.sigmoid(pred)
    mask = (prob > threshold).float()

    mask_np = mask[0, 0].detach().cpu().numpy()
    mask_png = (mask_np * 255).astype(np.uint8)

    save_path = os.path.join(cfg["output"]["save_dir"], "pred_mask.png")
    cv2.imwrite(save_path, mask_png)

    print("Inference done.")
    print(f"Device: {device}")
    print(f"Latency: {(end - start) * 1000:.3f} ms")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
