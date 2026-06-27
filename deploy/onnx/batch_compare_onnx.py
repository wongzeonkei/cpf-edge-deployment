import os
import sys
import time
import csv
import glob
import argparse
import yaml
import cv2
import torch
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infer_torch import build_model, load_checkpoint, load_gray_image


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
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")

    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    mask = mask[None, None, :, :]
    return mask


def collect_positive_samples(data_root, size, min_gt_fg, max_samples):
    ct_files = sorted(glob.glob(os.path.join(data_root, "*", "*_CT.png")))
    candidates = []

    for ct_path in ct_files:
        pet_path = ct_path.replace("_CT.png", "_PET.png")
        mask_path = ct_path.replace("_CT.png", "_mask.png")

        if not os.path.exists(pet_path) or not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        gt = (mask > 0).astype(np.uint8)
        gt_fg = int(gt.sum())

        if gt_fg >= min_gt_fg:
            candidates.append((gt_fg, ct_path, pet_path, mask_path))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:max_samples]


def run_torch_once(model, ct, pet, ablation_mode):
    with torch.no_grad():
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

    return out, (end - start) * 1000


def run_onnx_once(sess, ct_np, pet_np):
    start = time.time()
    out = sess.run(["logits"], {"ct": ct_np, "pet": pet_np})[0]
    end = time.time()
    return out, (end - start) * 1000


def summarize(rows, summary_path):
    metrics = [
        "torch_dice",
        "onnx_dice",
        "torch_iou",
        "onnx_iou",
        "torch_onnx_dice",
        "torch_onnx_iou",
        "diff_ratio",
        "gt_fg",
        "torch_fg",
        "onnx_fg",
        "max_abs_error_logits",
        "mean_abs_error_logits",
        "torch_latency_mean_ms",
        "onnx_latency_mean_ms",
    ]

    summary_rows = [["metric", "mean", "std", "min", "max"]]

    for metric in metrics:
        vals = []
        for row in rows:
            if metric not in row:
                continue
            try:
                vals.append(float(row[metric]))
            except Exception:
                continue

        if len(vals) == 0:
            print(f"[Warning] skip empty metric: {metric}")
            continue

        vals = np.array(vals, dtype=np.float64)
        summary_rows.append([
            metric,
            float(vals.mean()),
            float(vals.std()),
            float(vals.min()),
            float(vals.max()),
        ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/g/1/hjq/CIPA/data/PCLT20K")
    parser.add_argument("--onnx-path", type=str, default="deploy/onnx/models/cpf_static.onnx")
    parser.add_argument("--config", type=str, default="configs/cpf_infer.yaml")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--min-gt-fg", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="outputs/compare")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    size = cfg["model"]["input_size"]
    threshold = float(cfg["model"]["threshold"])
    ablation_mode = int(cfg["model"].get("ablation_mode", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Collecting positive samples...")
    samples = collect_positive_samples(
        data_root=args.data_root,
        size=size,
        min_gt_fg=args.min_gt_fg,
        max_samples=args.max_samples,
    )

    if len(samples) == 0:
        raise RuntimeError("No positive samples found. Try lowering --min-gt-fg.")

    print(f"Collected {len(samples)} samples.")

    model = build_model(cfg)
    model = load_checkpoint(model, cfg["model"]["checkpoint"])
    model = model.to(device).eval()

    sess = ort.InferenceSession(
        args.onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    first_gt_fg, first_ct, first_pet, first_mask = samples[0]
    ct_1ch = load_gray_image(first_ct, size)
    pet_np = load_gray_image(first_pet, size).astype(np.float32)
    ct_np = np.repeat(ct_1ch, 3, axis=1).astype(np.float32)

    ct = torch.from_numpy(ct_np).to(device)
    pet = torch.from_numpy(pet_np).to(device)

    print("Warmup...")
    for _ in range(args.warmup):
        _ = run_torch_once(model, ct, pet, ablation_mode)
        _ = sess.run(["logits"], {"ct": ct_np, "pet": pet_np})

    rows = []

    for idx, (gt_fg, ct_path, pet_path, mask_path) in enumerate(samples):
        image_id = os.path.basename(ct_path).replace("_CT.png", "")

        ct_1ch = load_gray_image(ct_path, size)
        pet_np = load_gray_image(pet_path, size).astype(np.float32)
        ct_np = np.repeat(ct_1ch, 3, axis=1).astype(np.float32)

        gt = load_mask(mask_path, size)

        ct = torch.from_numpy(ct_np).to(device)
        pet = torch.from_numpy(pet_np).to(device)

        torch_times = []
        onnx_times = []

        torch_logits = None
        onnx_logits = None

        for _ in range(args.repeat):
            torch_logits, t_ms = run_torch_once(model, ct, pet, ablation_mode)
            onnx_logits, o_ms = run_onnx_once(sess, ct_np, pet_np)
            torch_times.append(t_ms)
            onnx_times.append(o_ms)

        torch_logits_np = torch_logits.detach().cpu().numpy()

        torch_prob = sigmoid_np(torch_logits_np)
        onnx_prob = sigmoid_np(onnx_logits)

        torch_pred = (torch_prob >= threshold).astype(np.uint8)
        onnx_pred = (onnx_prob >= threshold).astype(np.uint8)

        torch_dice, torch_iou = dice_iou(torch_pred, gt)
        onnx_dice, onnx_iou = dice_iou(onnx_pred, gt)
        torch_onnx_dice, torch_onnx_iou = dice_iou(torch_pred, onnx_pred)

        diff = np.abs(torch_pred.astype(np.int32) - onnx_pred.astype(np.int32))
        diff_pixels = int(diff.sum())
        total_pixels = int(np.prod(torch_pred.shape))
        diff_ratio = diff_pixels / max(total_pixels, 1)

        torch_fg = int(torch_pred.sum())
        onnx_fg = int(onnx_pred.sum())

        abs_err = np.abs(torch_logits_np - onnx_logits)
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))

        row = {
            "index": idx,
            "image_id": image_id,
            "ct_path": ct_path,
            "pet_path": pet_path,
            "mask_path": mask_path,
            "gt_fg": int(gt.sum()),
            "torch_fg": torch_fg,
            "onnx_fg": onnx_fg,
            "torch_dice": torch_dice,
            "onnx_dice": onnx_dice,
            "torch_iou": torch_iou,
            "onnx_iou": onnx_iou,
            "torch_onnx_dice": torch_onnx_dice,
            "torch_onnx_iou": torch_onnx_iou,
            "diff_pixels": diff_pixels,
            "diff_ratio": diff_ratio,
            "max_abs_error_logits": max_abs_err,
            "mean_abs_error_logits": mean_abs_err,
            "torch_latency_mean_ms": float(np.mean(torch_times)),
            "onnx_latency_mean_ms": float(np.mean(onnx_times)),
        }

        rows.append(row)

        print(
            f"[{idx + 1}/{len(samples)}] {image_id} | "
            f"Torch Dice={torch_dice:.4f}, ONNX Dice={onnx_dice:.4f}, "
            f"Torch IoU={torch_iou:.4f}, ONNX IoU={onnx_iou:.4f}, "
            f"Torch/ONNX Dice={torch_onnx_dice:.4f}, "
            f"Diff={diff_ratio:.6f}, "
            f"Torch={np.mean(torch_times):.2f} ms, "
            f"ONNX={np.mean(onnx_times):.2f} ms"
        )

    csv_path = os.path.join(args.save_dir, "batch_compare_onnx.csv")
    summary_path = os.path.join(args.save_dir, "batch_compare_summary.csv")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows, summary_path)

    print("=" * 80)
    print("Batch compare finished.")
    print(f"Saved detailed CSV to: {csv_path}")
    print(f"Saved summary CSV to:  {summary_path}")
    print("=" * 80)

    for row in summary_rows[1:]:
        print(
            f"{row[0]}: "
            f"mean={float(row[1]):.6f}, "
            f"std={float(row[2]):.6f}, "
            f"min={float(row[3]):.6f}, "
            f"max={float(row[4]):.6f}"
        )


if __name__ == "__main__":
    main()
