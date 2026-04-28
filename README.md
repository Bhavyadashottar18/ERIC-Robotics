# 🤖 Bhavya Dashottar - Object Detection + Distance Estimation for Robotics Navigation

A complete pipeline for detecting navigation-relevant objects (cones, barriers, stop signs) and estimating their real-world distance from a camera using geometry-based methods. Optimized for edge deployment.

---

## 📁 Project Structure

```
robotics_nav/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── dataset.py               # BDD100K dataset loader & preprocessing
│   ├── model.py                 # YOLOv8 model wrapper + fine-tuning
│   ├── distance.py              # Geometry-based distance estimation
│   ├── annotator.py             # Bounding box drawing & annotation
│   ├── optimize.py              # Quantization & pruning utilities
│   └── inference.py             # Full inference pipeline
├── scripts/
│   ├── train.py                 # Fine-tune model on BDD100K
│   ├── evaluate.py              # Evaluate mAP, FPS benchmarks
│   ├── benchmark.py             # CPU vs GPU speed comparison
│   └── demo.py                  # Run inference on images/video
├── notebooks/
│   └── exploration.ipynb        # EDA + results visualization
└── results/                     # Output images, logs, metrics
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd robotics_nav
pip install -r requirements.txt
```

### 2. Download Dataset

Register at [Kaggle](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) and download BDD100K. Place it as:

```
data/
└── bdd100k/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Or use the Kaggle CLI:
```bash
kaggle datasets download -d solesensei/solesensei_bdd100k
unzip solesensei_bdd100k.zip -d data/
```

### 3. Configure

Edit `configs/config.yaml` to set your dataset path and camera intrinsics.

---

## 🚀 Usage

### Train
```bash
python scripts/train.py --config configs/config.yaml
```

### Run Demo on an Image
```bash
python scripts/demo.py --source path/to/image.jpg --weights runs/best.pt
```

### Run Demo on Video
```bash
python scripts/demo.py --source path/to/video.mp4 --weights runs/best.pt
```

### Benchmark FPS (CPU vs GPU)
```bash
python scripts/benchmark.py --weights runs/best.pt
```

### Optimize (Quantize / Prune)
```bash
python scripts/evaluate.py --weights runs/best.pt --optimize all
```

---

## 📐 Distance Estimation Method

Distance is estimated using the **pinhole camera model**:

```
Distance = (Real Object Height × Focal Length) / Pixel Height
```

| Object       | Real Height |
|-------------|------------|
| Traffic Cone | 0.75 m     |
| Stop Sign    | 0.75 m     |
| Barrier      | 1.0 m      |

Camera intrinsics (focal length) can be calibrated or approximated from EXIF/dataset metadata.

---

## ⚡ Optimization Results

| Model Variant         | CPU FPS | GPU FPS | mAP@50 |
|----------------------|---------|---------|--------|
| YOLOv8n (baseline)   | ~12     | ~85     | 0.XX   |
| YOLOv8n INT8 Quant   | ~18     | ~110    | 0.XX   |
| YOLOv8n Pruned 30%   | ~15     | ~95     | 0.XX   |

*(Fill in your actual numbers after running `benchmark.py`)*

---

## 🎯 Extra Credit Implemented

- ✅ Geometry-based distance (focal length + known object height)
- ✅ Bird's-eye view via Homography/Perspective Transform
- ✅ Optical Flow for cone tracking across frames
- ✅ Quantization (INT8) and structured pruning comparison

---

## 📤 Submission

1. Ensure all code is committed:
   ```bash
   git add .
   git commit -m "Complete robotics navigation assignment"
   git push origin main
   ```
2. Push to the **designated repository** (not your personal repo).
3. Include `results/` folder with sample annotated outputs.

---

## Contact Information

Name - Bhavya Dashottar

Contact Number - 7976047375

Email Address - bhavyadashottar18@gmail.com
