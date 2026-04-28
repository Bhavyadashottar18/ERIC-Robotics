# Bhavya Dashottar — Object Detection + Distance Estimation for Robotics Navigation

A pipeline for detecting navigation-relevant objects (cones, barriers, stop signs) and estimating their real-world distance using geometry-based methods.

## Project Structure

```
robotics_nav/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── distance.py
│   ├── annotator.py
│   ├── optimize.py
│   └── inference.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── demo.py
├── notebooks/
│   └── exploration.ipynb
└── results/
```


Download BDD100K from Kaggle and place it as:

```
data/bdd100k/
├── images/train/ & val/
└── labels/train/ & val/
```

## Usage

```bash
# Train
python scripts/train.py --config configs/config.yaml

# Demo on image or video
python scripts/demo.py --source path/to/file --weights runs/best.pt

# Benchmark
python scripts/benchmark.py --weights runs/best.pt
```

## Distance Estimation

Uses the pinhole camera model:

```
Distance = (Real Object Height × Focal Length) / Pixel Height
```

| Object | Real Height |
|--------|-------------|
| Traffic Cone | 0.75 m |
| Stop Sign | 0.75 m |
| Barrier | 1.0 m |

---

## Contact

**Bhavya Dashottar**  
bhavyadashottar18@gmail.com
