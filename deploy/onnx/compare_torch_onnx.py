import os
import sys
import time
import csv
import yaml
import cv2
import torch
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infer_torch import build_model, load_checkpoint, load_gray_image


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def dice_iou(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    inter = np.sum(pred * target)
    pred_sum = np.sum(pred)
    target_sum = np.sum(target)

    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (pred_sum + target_sum - inter + eps)

    return float(dice), float(iou)


def load_mask(path, size):
    if not path or not os.path.exists(path):
        return None

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    mask = mask[None, None, :, :]
    return mask


def benchmark_torch(model, ct, pet, ablation_mode, warmup=10, repeat=50):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(warmup):
            if ablation_mode == 0:
                out = model(ct, None)
            else:
                out = model(ct, pet)
            if isinstance(out, (list, tuple)):
                out = out[0]

        torch.cuda.synchronize()

        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.time()

            if ablation_mode == 0:
                out = model(ct, None)
            else:
                out = model(ct, pet)

            if isinstance(out, (list, tuple)):
                out = out[0]

            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

    return out, times


def benchmark_onnx(sess, ct_np, pet_np, warmup=10, repeat=50):
    times = []

    for _ in range(warmup):
        _ = sess.run(["logits"], {"ct": ct_np, "pet": pet_np})

    for _ in range(repeat):
        start = time.time()
        out = sess.run(["logits"], {"ct": ct_np, "pet": pet_np})[0]
        end = time.time()
        times.append((end - start) * 1000)

    return out, times


def main():
    cfg = load_config("configs/cpf_infer.yaml")

    onnx_path = "deploy/onnx/models/cpf_static.onnx"
    save_dir = "outputs/compare"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["model"]["input_size"]
    threshold = float(cfg["model"]["threshold"])
    ablation_mode = int(cfg["model"].get("ablation_mode", 2))

    ct_np_1ch = load_gray_image(cfg["data"]["sample_ct"], size)
    pet_np = load_gray_image(cfg["data"]["sample_pet"], size)

    ct_np = np.repeat(ct_np_1ch, 3, axis=1).astype(np.float32)
    pet_np = pet_np.astype(np.float32)

    ct_torch = torch.from_numpy(ct_np).to(device)
    pet_torch = torch.from_numpy(pet_np).to(device)

    model = build_model(cfg)
    model = load_checkpoint(model, cfg["model"]["checkpoint"])
    model = model.to(device)
    model.eval()

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    torch_logits, torch_times = benchmark_torch(
        model, ct_torch, pet_torch, ablation_mode=ablation_mode
    )
    onnx_logits, onnx_times = benchmark_onnx(sess, ct_np, pet_np)

    torch_logits_np = torch_logits.detach().cpu().numpy()

    abs_err = np.abs(torch_logits_np - onnx_logits)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))
    median_abs_err = float(np.median(abs_err))

    torch_prob = sigmoid_np(torch_logits_np)
    onnx_prob = sigmoid_np(onnx_logits)

    torch_pred = (torch_prob >= threshold).astype(np.uint8)
    onnx_pred = (onnx_prob >= threshold).astype(np.uint8)

    pred_diff = np.abs(torch_pred.astype(np.int32) - onnx_pred.astype(np.int32)).astype(np.uint8)

    total_pixels = int(np.prod(torch_pred.shape))
    diff_pixels = int(np.sum(pred_diff))
    diff_ratio = diff_pixels / max(total_pixels, 1)

    torch_fg = int(torch_pred.sum())
    onnx_fg = int(onnx_pred.sum())

    torch_onnx_dice, torch_onnx_iou = dice_iou(torch_pred, onnx_pred)

    mask = load_mask(cfg["data"].get("sample_mask", ""), size)
    if mask is not None:
        gt_fg = int(mask.sum())
        torch_dice, torch_iou = dice_iou(torch_pred, mask)
        onnx_dice, onnx_iou = dice_iou(onnx_pred, mask)
    else:
        gt_fg = -1
        torch_dice, torch_iou = None, None
        onnx_dice, onnx_iou = None, None

    cv2.imwrite(
        os.path.join(save_dir, "pred_mask_torch.png"),
        (torch_pred[0, 0] * 255).astype(np.uint8)
    )
    cv2.imwrite(
        os.path.join(save_dir, "pred_mask_onnx.png"),
        (onnx_pred[0, 0] * 255).astype(np.uint8)
    )
    cv2.imwrite(
        os.path.join(save_dir, "pred_diff.png"),
        (pred_diff[0, 0] * 255).astype(np.uint8)
    )

    csv_path = os.path.join(save_dir, "torch_onnx_compare.csv")
    rows = [
        ["metric", "value"],
        ["max_abs_error_logits", max_abs_err],
        ["mean_abs_error_logits", mean_abs_err],
        ["median_abs_error_logits", median_abs_err],
        ["torch_onnx_pred_dice", torch_onnx_dice],
        ["torch_onnx_pred_iou", torch_onnx_iou],
        ["gt_foreground_pixels", gt_fg],
        ["torch_foreground_pixels", torch_fg],
        ["onnx_foreground_pixels", onnx_fg],
        ["diff_pixels", diff_pixels],
        ["diff_ratio", diff_ratio],
        ["torch_latency_mean_ms", float(np.mean(torch_times))],
        ["torch_latency_median_ms", float(np.median(torch_times))],
        ["torch_latency_min_ms", float(np.min(torch_times))],
        ["torch_latency_max_ms", float(np.max(torch_times))],
        ["onnx_latency_mean_ms", float(np.mean(onnx_times))],
        ["onnx_latency_median_ms", float(np.median(onnx_times))],
        ["onnx_latency_min_ms", float(np.min(onnx_times))],
        ["onnx_latency_max_ms", float(np.max(onnx_times))],
    ]

    if mask is not None:
        rows.extend([
            ["torch_dice_vs_gt", torch_dice],
            ["torch_iou_vs_gt", torch_iou],
            ["onnx_dice_vs_gt", onnx_dice],
            ["onnx_iou_vs_gt", onnx_iou],
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("=" * 60)
    print("PyTorch vs ONNX Runtime Compare")
    print("=" * 60)
    print(f"ONNX providers: {sess.get_providers()}")
    print(f"CT shape: {ct_np.shape}")
    print(f"PET shape: {pet_np.shape}")
    print(f"Torch logits shape: {torch_logits_np.shape}")
    print(f"ONNX logits shape: {onnx_logits.shape}")
    print("-" * 60)
    print(f"Max abs error logits:    {max_abs_err:.8f}")
    print(f"Mean abs error logits:   {mean_abs_err:.8f}")
    print(f"Median abs error logits: {median_abs_err:.8f}")
    print(f"Pred Dice Torch/ONNX:    {torch_onnx_dice:.8f}")
    print(f"Pred IoU Torch/ONNX:     {torch_onnx_iou:.8f}")
    print(f"GT foreground pixels:    {gt_fg}")
    print(f"PyTorch foreground:      {torch_fg}")
    print(f"ONNX foreground:         {onnx_fg}")
    print(f"Diff pixels:             {diff_pixels} / {total_pixels}")
    print(f"Diff ratio:              {diff_ratio:.8f}")
    print("-" * 60)
    print(f"PyTorch latency mean:    {np.mean(torch_times):.3f} ms")
    print(f"ONNX latency mean:       {np.mean(onnx_times):.3f} ms")
    if mask is not None:
        print("-" * 60)
        print(f"PyTorch Dice vs GT:      {torch_dice:.8f}")
        print(f"ONNX Dice vs GT:         {onnx_dice:.8f}")
        print(f"PyTorch IoU vs GT:       {torch_iou:.8f}")
        print(f"ONNX IoU vs GT:          {onnx_iou:.8f}")
    print("-" * 60)
    print(f"Saved compare CSV to: {csv_path}")
    print(f"Saved masks to: {save_dir}")


if __name__ == "__main__":
    main()
