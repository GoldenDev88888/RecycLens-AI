# EcoVision AI — Waste Detection & Analytics

A local-first extension of [YOLOWaste](https://github.com/mysticalayushi/YOLOWaste) by Ayushi Rai. The original MIT notice, trained YOLOv8 weights, notebook, results and screenshots are preserved. The original README is archived in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). This extension does not claim the upstream training results as new work.

## What is new

- Shared YOLO inference layer with input validation, EXIF-aware RGB decoding, consistent color conversion, confidence/IoU controls, bounding boxes, class counts and measured inference-call latency.
- Responsive FastAPI-hosted HTML/CSS/JavaScript dashboard: image upload, consent-based webcam frame capture, original/annotated comparison, JPEG download and inspection history. No Node.js build needed.
- SQLite history and aggregate analytics; CSV export. Original uploaded images are not stored. Local video paths can appear in detailed records.
- CLI image, video and webcam processing; optional ByteTrack IDs, annotated MP4 output, FPS and frame-level counts. Videos are silent and processed up to a configurable frame limit.
- Non-destructive dataset audit and stratified splitting with content hashes, orphan/invalid-label checks and a generated dataset manifest.
- Configurable training, test-set evaluation and optional ONNX export. Updated optional Streamlit interface. Original Colab notebook retained as historical material; its paths may need adjustment.
- Regression tests, CI definition and Dockerfile.

## Quick start (Windows PowerShell)

Use Python 3.11, from the extracted project root. If using a GPU, install a compatible PyTorch build first following its official installation instructions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ecovision.cli serve
```

Open http://127.0.0.1:8000 (dashboard) or http://127.0.0.1:8000/docs (API). Upload a photo containing cardboard, metal, paper or plastic. A checkpoint is bundled at `models/best.pt`; only load trusted model files, since PyTorch checkpoints can contain serialized executable objects. No automatic weight download occurs in the application. Use environment variables `ECOVISION_WEIGHTS`, `ECOVISION_DEVICE` (e.g. `cpu` or `0`) and `ECOVISION_DB` to change server settings. The first request loads the model. Health reports process status, not model readiness.

Linux/macOS: use `source .venv/bin/activate` instead. No public authentication is supplied; bind to localhost. Do not publish the service without adding auth, HTTPS, reverse-proxy request limits, rate limiting, quotas and data retention controls. Camera access requires HTTPS or localhost plus browser permission.

## Image and video CLI

```powershell
python -m ecovision.cli image input.jpg --output output/detection.jpg --device cpu
python -m ecovision.cli video input.mp4 --output output/detection.mp4 --max-frames 300
python -m ecovision.cli video 0 --camera --track --max-frames 300 --output output/webcam.mp4
```

Add `--preview` to the video command to open an OpenCV preview (desktop GUI required; Q to stop). ByteTrack may require an additional `lap` installation (`pip install lap>=0.5.12`, quote the requirement on shells where `>` redirects). Tracker state is isolated per CLI run. Unique track IDs are approximate and can change after occlusion. Video frame totals are not unique waste counts. Output uses MP4V and has no audio; if a player cannot open it, use a compatible player or transcode to H.264. The CLI refuses to overwrite existing output. Processing speed depends on hardware and input; no real-time speed guarantee.

API endpoints: `POST /api/predict` multipart `file`, `conf`, `iou`, `imgsz`; `GET /api/history`, `/api/summary`, `/api/history.csv`, `/api/health`. Video and webcam continuous tracking are CLI features; the browser webcam mode captures a single frame per inspection.

## Dataset preparation and training

Download the original [Garbage Classification 4 dataset](https://universe.roboflow.com/recycling-vs-waste/garbage-classification-4-oklrj) separately and check its license. Put YOLO images and labels under `data/images/{train,val,test}` and `data/labels/{train,val,test}`. The original dataset is not included. Empty label files are valid negative examples.

```powershell
python scripts/split_dataset.py --source data --audit-only
python scripts/split_dataset.py --source data --output data_balanced
python scripts/train.py --data data_balanced/data.yaml --epochs 50 --batch 16 --device 0
python scripts/train.py --data data_balanced/data.yaml --weights models/best.pt --evaluate-only --device cpu
```

The audit rejects exact duplicate image bytes (including across splits), invalid YOLO boxes and missing pairs. It does not identify near-duplicates or semantic leakage; group related images by acquisition session before evaluation. The new 70/20/10 split uses dominant class (not multilabel balancing) and fails safely when the data is too small for stratification. Source files are never deleted. To use a frozen test set, do not resplit it; keep a separately held-out evaluation dataset. `--export-onnx` enables optional export (additional ONNX packages may be needed). Training downloads a pretrained base checkpoint when not supplied locally.

## Optional Streamlit and Docker

```powershell
python -m pip install -r requirements-streamlit.txt
python -m streamlit run app/app.py
```

```sh
docker build -t ecovision-ai .
docker run --rm -p 127.0.0.1:8000:8000 -v ecovision-data:/app/runtime ecovision-ai
```

Docker is provided as an untested deployment recipe. CPU inference is the default when no GPU is available. For GPU containers, configure NVIDIA Container Toolkit and appropriate PyTorch/CUDA dependencies separately.

## Testing and results

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [evidence/TEST_REPORT.md](evidence/TEST_REPORT.md) and [evidence/pytest.txt](evidence/pytest.txt). The automated tests inject deterministic fixture detections, not actual model outputs. The uploaded trained weights were preserved byte-for-byte, but Ultralytics could not be installed in the editing environment. Live model inference, retraining, ONNX export, Streamlit, browser camera and Docker execution are not validated here. Original reported results (86.0% mAP50, 90.5% precision, 77.1% recall) belong to the upstream project, not this enhancement. No new model-accuracy results are claimed.

## Licensing and attribution

Original repository: Ayushi Rai, [mysticalayushi/YOLOWaste](https://github.com/mysticalayushi/YOLOWaste), MIT, see LICENSE. Retain this notice and attribution. Ultralytics and its checkpoints, datasets, images and other dependencies may be subject to separate terms (including Ultralytics AGPL-3.0 or commercial licensing). Review these before distributing, commercial deployment or model redistribution. This repository is an enhanced derivative, not an independently trained replacement.
