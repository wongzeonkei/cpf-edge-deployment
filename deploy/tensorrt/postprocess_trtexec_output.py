import os
import json
import cv2
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def collect_numeric_lists(obj, out):
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            out.append(obj)
        else:
            for x in obj:
                collect_numeric_lists(x, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numeric_lists(v, out)


def main():
    json_path = "outputs/jetson_trt/trt_fp16_512_opset13_output.json"
    mask_path = "outputs/jetson_trt/pred_mask_fp16_512_opset13_trtexec.png"
    logits_path = "outputs/jetson_trt/logits_fp16_512_opset13_trtexec.npy"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    numeric_lists = []
    collect_numeric_lists(data, numeric_lists)

    target_size = 512 * 512
    candidates = [x for x in numeric_lists if len(x) == target_size]

    if not candidates:
        lengths = sorted(set(len(x) for x in numeric_lists))
        raise RuntimeError(
            "Cannot find logits array with length 262144. "
            f"Found numeric list lengths: {lengths[:20]}"
        )

    logits = np.array(candidates[0], dtype=np.float32).reshape(1, 1, 512, 512)
    prob = sigmoid(logits)
    mask = (prob >= 0.35).astype(np.uint8) * 255
    mask_2d = mask.squeeze()

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask_2d)
    np.save(logits_path, logits)

    print("Postprocess done.")
    print("Logits shape:", logits.shape)
    print("Logits min/max:", float(logits.min()), float(logits.max()))
    print("Mask foreground pixels:", int((mask_2d > 0).sum()))
    print("Saved mask:", mask_path)
    print("Saved logits:", logits_path)


if __name__ == "__main__":
    main()
