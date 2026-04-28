"""
HOW TO SUBMIT
=============
Follow these steps exactly to submit to the designated repository.

──────────────────────────────────────────────
STEP 1 — Clone the designated repository
──────────────────────────────────────────────
    git clone <DESIGNATED_REPO_URL>
    cd <repo-folder>

  ⚠ Do NOT push to your personal GitHub repo.
    Use the repository Eric (or the hiring team) provided.

──────────────────────────────────────────────
STEP 2 — Copy this project into the repo
──────────────────────────────────────────────
    cp -r /path/to/robotics_nav/* .

  Or if you developed directly in the repo folder, skip this step.

──────────────────────────────────────────────
STEP 3 — Add sample results (important!)
──────────────────────────────────────────────
After training and running the demo, ensure the results/ folder contains:
  • results/sample_detections.png   — annotated output images
  • results/optimization_fps.png    — before/after benchmark chart
  • results/bird_eye_view.png       — BEV extra credit
  • results/distance_geometry.png   — geometry plot

  git add results/

──────────────────────────────────────────────
STEP 4 — Commit everything
──────────────────────────────────────────────
    git add .
    git commit -m "feat: complete robotics navigation detection assignment

    - YOLOv8n fine-tuned on BDD100K (cone/barrier/stop-sign)
    - Geometry-based distance estimation (pinhole model)
    - INT8 quantization + L1 pruning with FPS benchmarks
    - Bird's-eye view via homography (extra credit)
    - Optical flow tracking (extra credit)
    "

──────────────────────────────────────────────
STEP 5 — Push to designated repo
──────────────────────────────────────────────
    git push origin main

  (Use 'master' instead of 'main' if that's the default branch.)

──────────────────────────────────────────────
CHECKLIST before pushing
──────────────────────────────────────────────
  [ ] All src/*.py files committed
  [ ] scripts/train.py, demo.py, evaluate.py, benchmark.py committed
  [ ] configs/config.yaml committed
  [ ] requirements.txt committed
  [ ] notebooks/exploration.ipynb committed
  [ ] results/ folder with at least 2-3 output images
  [ ] README.md is clear and complete
  [ ] No large dataset files committed (add data/ to .gitignore)

──────────────────────────────────────────────
.gitignore recommendation
──────────────────────────────────────────────
Add this to .gitignore to avoid committing huge files:
  data/
  runs/train/weights/*.pt   # large model files
  __pycache__/
  *.egg-info/
  .env

  Keep: runs/train/weights/best.pt  if < 50MB (YOLOv8n is ~6MB — safe to include)
"""
