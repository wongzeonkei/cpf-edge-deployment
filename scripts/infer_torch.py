import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gray_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # size: [H, W], cv2.resize needs (W, H)
    img = cv2.resize(img, (size[1], size[0]))
    img = img.astype(np.float32)

    # TODO: replace with the exact normalization used in training.
    # Current placeholder assumes png range [0, 255].
    img = img / 255.0

    # [H, W] -> [1, 1, H, W]
    img = img[None, None, :, :]
    return img


def build_model(cfg):
    from models.builder2 import EncoderDecoder as CPFModel

    mcfg = cfg["model"]

    model_cfg = SimpleNamespace()
    model_cfg.backbone = "Swin_transformer"
    model_cfg.pretrained_model = None
    model_cfg.decoder = "MambaDecoder"
    model_cfg.decoder_embed_dim = 512

    model_cfg.image_height = int(mcfg["input_size"][0])
    model_cfg.image_width = int(mcfg["input_size"][1])

    model_cfg.bn_eps = 1e-3
    model_cfg.bn_momentum = 0.1
    model_cfg.num_classes = int(mcfg.get("num_classes", 1))

    model_cfg.ablation_mode = int(mcfg.get("ablation_mode", 2))
    model_cfg.fixed_tau = float(mcfg.get("fixed_tau", 0.005))

    model = CPFModel(cfg=model_cfg, norm_layer=nn.BatchNorm2d)
    return model


def clean_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and ckpt.get("model_ema") is not None:
        state_dict = ckpt["model_ema"]
        print("Loaded checkpoint key: model_ema")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("Loaded checkpoint key: model")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("Loaded checkpoint key: state_dict")
    else:
        state_dict = ckpt
        print("Loaded raw state_dict")

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])

    return model


def main():
    cfg = load_config("configs/cpf_infer.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["model"]["input_size"]
    threshold = float(cfg["model"]["threshold"])
    ablation_mode = int(cfg["model"].get("ablation_mode", 2))

    ct_np = load_gray_image(cfg["data"]["sample_ct"], size)
    pet_np = load_gray_image(cfg["data"]["sample_pet"], size)

    ct = torch.from_numpy(ct_np).to(device)     # [1, 1, H, W]
    pet = torch.from_numpy(pet_np).to(device)   # [1, 1, H, W]

    # CPF training/eval uses CT as 3-channel input and PET as 1-channel prior input.
    ct = ct.repeat(1, 3, 1, 1)                  # [1, 3, H, W]

    model = build_model(cfg)
    model = load_checkpoint(model, cfg["model"]["checkpoint"])
    model = model.to(device)
    model.eval()

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        if ablation_mode == 0:
            pred = model(ct, None)
        else:
            pred = model(ct, pet)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    # CPF with deep supervision returns (x_last, x_output_0, x_output_1, x_output_2).
    # The original evaluation code uses output[0].
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    prob = torch.sigmoid(pred)
    mask = (prob >= threshold).float()

    mask_np = mask[0, 0].detach().cpu().numpy()
    mask_png = (mask_np * 255).astype(np.uint8)

    save_path = os.path.join(cfg["output"]["save_dir"], "pred_mask.png")
    cv2.imwrite(save_path, mask_png)

    print("Inference done.")
    print(f"Device: {device}")
    print(f"CT shape: {tuple(ct.shape)}")
    print(f"PET shape: {tuple(pet.shape)}")
    print(f"Pred shape: {tuple(pred.shape)}")
    print(f"Threshold: {threshold}")
    print(f"Latency: {(end - start) * 1000:.3f} ms")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
