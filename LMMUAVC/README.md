# Multi-Modal Fusion for UAV Classification

This repository provides the implementation of our multi-modal fusion framework for drone type classification using RGB images and multiple LiDAR modalities (Livox Avia, 360° LiDAR, radar-enhanced point clouds) from the paper:

**"Multi-Modal Fusion for UAV Classification Using Image and Multi-Type LiDAR Sensors with Explicit Missing Modality Modeling"**

## Highlights

- **Four modalities**: RGB (YOLO11), Livox Avia, 360° LiDAR, radar-enhanced point clouds
- **Lightweight fusion**: Tiny MLP (~0.1M params), Compact Transformer (~0.4M), Efficient Fusion (~3M)
- **Robust to missing modalities**: Learnable embeddings instead of zero-padding
- **Dynamic weighting**: Confidence + point cloud density based
- **96.88% test accuracy** on MMAUD dataset with only 0.1M parameters (Tiny MLP)

## Dataset

This work is evaluated on the **MMAUD** dataset from the [CVPR 2024 UG2+ Challenge](https://cvpr2024ug2.github.io/). Please follow the official instructions to download the dataset.

Expected data structure:
```
data/
├── seq0001/
│   ├── Image/              # RGB images
│   ├── livox_avia/         # Livox point clouds (.npy)
│   ├── lidar_360/          # 360° LiDAR point clouds
│   ├── radar_enhance_pcl/  # Radar-enhanced point clouds
│   ├── class/              # Class labels (0-3)
│   └── ground_truth/
├── seq0002/
└── ...
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/MMUAD.git
cd MMUAD
pip install -r requirements.txt
```

### 2. PointNeXt (for point cloud features)

Clone [PointNeXt](https://github.com/guochengqian/PointNeXt) **inside the repository root**:

```bash
cd MMUAD_github
git clone https://github.com/guochengqian/PointNeXt.git
cd PointNeXt && pip install -e . && cd ..
```

### 3. Download pretrained weights

Place the following files in `visual_processing/preprocessing/`:

| File | Description | Source |
|------|-------------|--------|
| `yolo11s.pt` | YOLO11-S for image detection | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| `pointnext-s.pth` | PointNeXt-S for point cloud features | [PointNeXt](https://github.com/guochengqian/PointNeXt) |
| `lstm_livox_avia_model.pth` | Livox Avia detector | Train with `train_livox_detector.py` |
| `lstm_lidar_360_model.pth` | Lidar 360 detector | Train with `train_lidara_detector.py` |
| `lstm_radar_enhance_model.pth` | Radar detector | Train with `train_radar_detector.py` |

## Quick Start

**Important**: Run all commands from the repository root (`MMUAD_github/`). The scripts add the repo root to `PYTHONPATH` automatically when using the shell scripts, or set it manually:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Step 1: Feature extraction

```bash
cd visual_processing/preprocessing
python extract_and_align_features.py \
    --data-root /path/to/data \
    --output-dir ./out \
    --convnext-output ./features/image_features \
    --pointnext-output ./features/livox_features \
    --lidar360-pointnext-output ./features/lidar360_features \
    --radar-pointnext-output ./features/radar_features \
    --yolo-weights yolo11s.pt \
    --window-size 0.4 \
    --livox-detector-model ./lstm_livox_avia_model.pth \
    --lidar360-detector-model ./lstm_lidar_360_model.pth \
    --radar-detector-model ./lstm_radar_enhance_model.pth \
    --pointnext-cfg ../../PointNeXt/cfgs/scanobjectnn/pointnext-s.yaml \
    --pointnext-pretrained ./pointnext-s.pth
```

### Step 2: Merge and split dataset

```bash
python merge_and_split_features.py \
    --features-dir ./features \
    --base-dir /path/to/data \
    --output-dir ./features \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --split-by-time \
    --modalities image livox lidar360 radar
```

### Step 3: Train (Tiny MLP recommended)

```bash
python train_lightweight.py \
    --model-type tiny \
    --base-dir /path/to/data \
    --timeline-dir ./out \
    --features-dir ./features \
    --output-dir ./checkpoints_tiny \
    --batch-size 32 \
    --num-epochs 100 \
    --lr 1e-4 \
    --device cuda
```

### Step 4: Evaluate

```bash
python evaluate_multimodal_classifier.py \
    --model-path ./checkpoints_tiny/best_model.pth \
    --base-dir /path/to/data \
    --timeline-dir ./out \
    --features-dir ./features \
    --split test \
    --class-names Class_0 Class_1 Class_2 Class_3
```

## Model variants

| Model | Params | Use case |
|-------|--------|----------|
| Tiny MLP | ~0.1M | Small datasets, best generalization |
| Compact Transformer | ~0.4M | Medium datasets |
| Efficient Fusion | ~3M | Larger datasets |

## Citation

If you use this code, please cite our paper:

```bibtex
@article{mmuad2025,
  title={Multi-Modal Fusion for UAV Classification Using Image and Multi-Type LiDAR Sensors with Explicit Missing Modality Modeling},
  author={Feng, Yibo and Li, Ning and Wu, Di},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2025}
}
```

## License

This project is for research purposes. See the MMAUD dataset license for data usage terms.

## Acknowledgments

- [MMAUD Dataset](https://github.com/dtc111111/Multi-Modal-UAV) - CVPR 2024 UG2+ Challenge
- [PointNeXt](https://github.com/guochengqian/PointNeXt) - Point cloud feature extraction
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) - Image detection
