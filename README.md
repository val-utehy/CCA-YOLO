# CCA-YOLOv12

## Overview

CCA-YOLOv12 is a modified YOLOv12 object detector built on the local Ultralytics codebase. Its backbone integrates Coordinate Attention through three `C2f_CA` blocks at the P3, P4, and P5 stages to explore spatially aware feature extraction for object detection.

- **Code:** [GitHub repository](https://github.com/mluu59990-collab/Yolov12_backbone_modify)
- **Dataset:** [Google Drive folder](https://drive.google.com/drive/folders/15Kb_uhkGJDsftOMYpr5kEPHsRrTFoeQW?hl=vi)

The model configuration is [`yolov12s_3cca.yaml`](ultralytics/cfg/models/v12/yolov12s_3cca.yaml), which selects the small (`s`) scale and uses a three-scale detection head. The attention modules are implemented in [`block.py`](ultralytics/nn/modules/block.py) and registered in the model parser in [`tasks.py`](ultralytics/nn/tasks.py).

## Setup

Clone the repository and create a Python 3.11 environment:

```bash
git clone https://github.com/mluu59990-collab/Yolov12_backbone_modify.git
cd Yolov12_backbone_modify
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4" "torch==2.2.2" "torchvision==0.17.2"
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`. The editable installation makes the `yolo` command use this repository's custom modules.

The commands above install the core training and prediction dependencies. [`requirements.txt`](requirements.txt) also lists optional demo and export packages, but references a local FlashAttention wheel for Linux x86_64, Python 3.11, CUDA 11, and PyTorch 2.2 that is not included in this repository. Installing that file directly requires supplying the matching wheel. FlashAttention is optional: the implementation falls back to PyTorch scaled dot-product attention when it is unavailable.

## Training

Prepare your dataset in YOLO detection format, with one label file per image. Each label row must contain `class_id x_center y_center width height`, with coordinates normalized to the image dimensions and class IDs starting at zero.

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Create a `data.yaml` file with your dataset path and class names. This single-class example is a template; the repository does not include a custom training dataset:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: object
```

Train the modified model from its architecture configuration:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v12/yolov12s_3cca.yaml \
  data=data.yaml \
  epochs=50 imgsz=640 batch=16 device=0 \
  project=runs/detect name=yolov12s_3cca
```

These hyperparameters are example settings. Adjust the batch size for your available memory. `device=0` selects the first CUDA GPU; use `device=cpu` for CPU execution. Training takes the class count from your dataset configuration.

For the first run with this name, checkpoints are saved to:

```text
runs/detect/yolov12s_3cca/weights/best.pt
runs/detect/yolov12s_3cca/weights/last.pt
```

Repeated runs may create a numbered output directory. Use the actual path printed by the training command.

## Prediction

Run inference with your trained checkpoint on an image directory:

```bash
yolo detect predict \
  model=runs/detect/yolov12s_3cca/weights/best.pt \
  source=/absolute/path/to/images \
  imgsz=640 conf=0.25 save=True
```

Replace `source` with an image or video path as needed. Annotated predictions are saved under `runs/detect/predict`, or a numbered directory if it already exists.

To evaluate the checkpoint on your validation split:

```bash
yolo detect val \
  model=runs/detect/yolov12s_3cca/weights/best.pt \
  data=data.yaml imgsz=640
```

## Trained models

No trained `.pt` checkpoints are included in this checkout. Train the custom architecture using the command above to generate your own weights.

| Checkpoint | Purpose |
| --- | --- |
| `best.pt` | Best checkpoint selected during training; use for evaluation and prediction. |
| `last.pt` | Latest training checkpoint; use to resume an interrupted run. |

Resume training with:

```bash
yolo detect train model=runs/detect/yolov12s_3cca/weights/last.pt resume=True
```

## License

This repository includes Ultralytics code distributed under the [AGPL-3.0 license](LICENSE).
