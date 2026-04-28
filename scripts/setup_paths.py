import os, sys
from pathlib import Path
import yaml

HOME = Path.home()
print(f"Detected home: {HOME}")

SEARCH_DIRS = [HOME / "Downloads", HOME / "Desktop", HOME / "Documents", Path("/Volumes")]

def find_folder(name):
    for base in SEARCH_DIRS:
        c = base / name
        if c.exists(): return c
    return None

bdd100k_root = find_folder("bdd100k")
labels_root  = find_folder("bdd100k_labels_release")
print(f"Found bdd100k: {bdd100k_root}")
print(f"Found labels:  {labels_root}")

if not bdd100k_root: print("ERROR: bdd100k not found"); sys.exit(1)
if not labels_root:  print("ERROR: bdd100k_labels_release not found"); sys.exit(1)

images_train = bdd100k_root / "bdd100k" / "images" / "100k" / "train"
images_val   = bdd100k_root / "bdd100k" / "images" / "100k" / "val"
json_train   = labels_root  / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
json_val     = labels_root  / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
yolo_train   = bdd100k_root / "yolo_labels" / "train"
yolo_val     = bdd100k_root / "yolo_labels" / "val"

for lbl, p in [("images/train", images_train),("images/val",images_val),("json_train",json_train),("json_val",json_val)]:
    print(f"  [{chr(79) if p.exists() else chr(88)}] {lbl}: {p}")

config_path = Path("configs/config.yaml")
cfg = yaml.safe_load(open(config_path))
cfg["dataset"]["root"]              = str(bdd100k_root)
cfg["dataset"]["images_train"]      = str(images_train)
cfg["dataset"]["images_val"]        = str(images_val)
cfg["dataset"]["labels_json_train"] = str(json_train)
cfg["dataset"]["labels_json_val"]   = str(json_val)
cfg["dataset"]["labels_train"]      = str(yolo_train)
cfg["dataset"]["labels_val"]        = str(yolo_val)
yaml.dump(cfg, open(config_path,"w"), default_flow_style=False, allow_unicode=True)
print("config.yaml updated! Now run: python scripts/train.py")
