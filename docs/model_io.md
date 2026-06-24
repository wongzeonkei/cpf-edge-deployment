# CPF Model Input / Output Specification

## Model Name

CPF: Controlled Prior Fusion for PET/CT Lung Tumor Segmentation

## Checkpoint

- Path: weights/cpf_best.pth
- Epoch: TODO
- Dataset: PCLT20K
- Threshold: 0.35

## Input

- Modalities: CT + PET

## Input Tensor Shape

### Option A: Separate CT and PET inputs

- CT: [B, 1, H, W]
- PET: [B, 1, H, W]

### Option B: Concatenated input

- Input: [B, 2, H, W]

## Image Size

- H: 640
- W: 640

## Data Type

- float32

## Normalization

- CT: TODO
- PET: TODO

## Output

- Output tensor shape: [B, 1, H, W]
- Output meaning: logits or probability
- Activation: sigmoid
- Threshold: 0.35

## Postprocess

1. Apply sigmoid if output is logits.
2. Apply threshold = 0.35.
3. Resize back to original size if needed.
4. Save binary mask.
5. Calculate metrics if GT is available.

## Metrics

- Dice
- IoU
- HD95
- Precision
- Recall
