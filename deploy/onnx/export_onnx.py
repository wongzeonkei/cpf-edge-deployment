import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infer_torch import build_model, load_checkpoint


class CPFOnnxWrapper(nn.Module):
    """
    Wrap CPF model for ONNX export.

    Original CPF may return multiple outputs due to deep supervision.
    This wrapper keeps only the main output: output[0].
    """

    def __init__(self, model, ablation_mode=2):
        super().__init__()
        self.model = model
        self.ablation_mode = ablation_mode

    def forward(self, ct, pet):
        if self.ablation_mode == 0:
            out = self.model(ct, None)
        else:
            out = self.model(ct, pet)

        if isinstance(out, (list, tuple)):
            out = out[0]

        return out


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/cpf_infer.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = cfg["model"]["input_size"]
    h, w = int(input_size[0]), int(input_size[1])

    ablation_mode = int(cfg["model"].get("ablation_mode", 2))

    onnx_path = "deploy/onnx/models/cpf_static.onnx"

    model = build_model(cfg)
    model = load_checkpoint(model, cfg["model"]["checkpoint"])
    model = model.to(device)
    model.eval()

    wrapper = CPFOnnxWrapper(model, ablation_mode=ablation_mode)
    wrapper = wrapper.to(device)
    wrapper.eval()

    ct = torch.randn(1, 3, h, w, dtype=torch.float32, device=device)
    pet = torch.randn(1, 1, h, w, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print("Exporting ONNX...")
    print(f"CT input shape:  {tuple(ct.shape)}")
    print(f"PET input shape: {tuple(pet.shape)}")
    print(f"Output path: {onnx_path}")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        torch.onnx.export(
            wrapper,
            (ct, pet),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["ct", "pet"],
            output_names=["logits"],
            dynamic_axes={
                "ct": {0: "batch"},
                "pet": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    print("ONNX export finished.")
    print(f"Time: {(end - start):.2f} s")
    print(f"Saved to: {onnx_path}")


if __name__ == "__main__":
    main()
