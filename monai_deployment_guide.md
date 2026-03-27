# MONAI Inference Server — Deployment Guide

A containerized medical imaging inference server using **MONAI** and **EfficientNet-B0** for multi-label chest X-ray classification with **Grad-CAM** visualization.

---

## Architecture Diagrams

### High-Level Architecture

![High-Level Architecture](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/highlevel_architecture_1772174546621.png)

### Simplified Architecture

![Simplified Architecture](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/monai_architecture_simple_1772117041269.png)

### Low-Level Architecture

![Low-Level Architecture](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/lowlevel_architecture_1772174895102.png)

### Infrastructure Architecture

![Infrastructure Architecture](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/architecture_diagram_1772173681951.png)

| Component | Technology | Container |
|-----------|-----------|-----------|
| **Backend** | FastAPI + MONAI + PyTorch (CUDA 12.1) | `monai-backend` |
| **Frontend** | React (Vite) + Nginx | `monai-frontend` |
| **Model** | EfficientNet-B0 (14-class multi-label) | Loaded at startup |
| **GPU** | NVIDIA GPU via Container Toolkit | Required |

---

## Prerequisites

- **Docker** ≥ 24.0 with Docker Compose v2
- **NVIDIA Container Toolkit** ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **NVIDIA GPU** with CUDA 12.1+ support
- Available ports: `8001` (backend), `3001` (frontend)

---

## Project Structure

```
MonAi/
├── backend/
│   ├── Dockerfile            # CUDA 12.1 + Python 3.11
│   ├── main.py               # FastAPI server
│   ├── requirements.txt      # Python dependencies
│   └── utils/
│       └── helper.py          # Model loading, inference, Grad-CAM
├── frontend/
│   ├── Dockerfile            # Node build → Nginx serve
│   ├── nginx.conf            # Reverse proxy config
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   └── index.css         # Styles
│   └── package.json
├── outputs/                   # Volume-mounted data
│   ├── checkpoints/          # Model checkpoints (.pt)
│   ├── val_samples/          # 200 validation X-ray images
│   └── val_samples.pkl       # Pickled sample metadata
└── docker-compose.yml
```

---

## Deployment Steps

### 1. Build the Docker Images

```bash
cd MonAi
docker compose build
```

> [!NOTE]
> First build pulls the NVIDIA CUDA 12.1 base image (~1.3GB) and installs PyTorch + MONAI. Subsequent builds are cached and fast.

### 2. Start the Services

```bash
docker compose up -d
```

### 3. Verify Both Containers Are Running

```bash
docker compose ps
```

Expected output:
```
NAME             IMAGE            STATUS          PORTS
monai-backend    monai-backend    Up              0.0.0.0:8001->8000/tcp
monai-frontend   monai-frontend   Up              0.0.0.0:3001->80/tcp
```

### 4. Verify Backend Health

```bash
# Check model loaded successfully
docker logs monai-backend --tail 5

# Expected: "Application startup complete."
# Expected: "Uvicorn running on http://0.0.0.0:8000"
```

### 5. Test API Endpoints

```bash
# Get available classes
curl http://localhost:8001/classes

# Get a random sample
curl http://localhost:8001/sample

# Run inference
curl -X POST http://localhost:8001/infer \
  -H "Content-Type: application/json" \
  -d '{"image_path": "00000091_003.png", "class_name": "Cardiomegaly"}'
```

### 6. Access the Application

Open **http://localhost:3001** in your browser.

---

## Docker Compose Configuration

```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: monai-backend
    ports:
      - "8001:8000"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - CKPT_PATH=/app/outputs/checkpoints/best_cxr_multilabel_13.pt
      - VAL_LOADER_PATH=/app/outputs/val_samples.pkl
      - TEMP_DIR=/app/static/heatmaps
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: monai-frontend
    ports:
      - "3001:80"
    depends_on:
      - backend
    restart: unless-stopped
```

> [!IMPORTANT]
> The `deploy.resources.reservations.devices` section requires the NVIDIA Container Toolkit. Without it, the backend container will fail to start.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/classes` | Returns 14 CXR class names |
| `GET` | `/sample` | Returns a random validation sample with ground truth |
| `POST` | `/infer` | Runs inference on a pre-loaded sample image |
| `POST` | `/infer_upload` | Runs inference on a user-uploaded image |
| `GET` | `/images/{filename}` | Serves validation sample images |
| `GET` | `/static/heatmaps/{filename}` | Serves generated Grad-CAM heatmaps |

### Supported CXR Classes (14)

Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema, Consolidation

---

## Application Walkthrough

### Landing Page
The dashboard opens with two panels — an image panel (left) and results panel (right). Use **"Auto Fill (Sample)"** to load a random validation X-ray, or **"Upload Image"** to use your own.

![Landing Page](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/monai_landing_1772115261697.png)

### Sample Loaded
After clicking "Auto Fill", a chest X-ray loads with its **ground truth labels** displayed as badges. Select the **target class** for Grad-CAM from the dropdown, then click **"Submit for Inference"**.

![Sample Loaded](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/monai_sample_loaded_1772115301986.png)

### Inference Results
The results panel shows the **model prediction**, a **Grad-CAM heatmap** highlighting regions of interest, and **top-5 class probabilities** with visual progress bars.

![Inference Results](/home/administrator/.gemini/antigravity/brain/2e155ccb-ea1d-42c4-9951-6e18cf76c69f/monai_inference_results_1772115329696.png)

---

## Fixes Applied During Deployment

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Port conflicts (8000, 3000) | Existing containers using those ports | Changed to ports `8001` and `3001` |
| Missing `python-multipart` | Not in `requirements.txt` | Added `python-multipart>=0.0.6` |
| Wrong checkpoint file | `_3.pt` (256KB) was corrupted | Switched to `best_cxr_multilabel_13.pt` (EfficientNet) |
| `val_samples` path resolution | `PROJECT_ROOT` resolved to `/` in container | Use `BASE_DIR` directly for path construction |
| 72 corrupted sample images | Incomplete tar extraction | Copied valid files from `val_samples_1/` |
| Upload format errors | MONAI can't load all image formats | PIL pre-conversion to grayscale PNG |
| Truncated image uploads | PIL strict mode rejects truncated files | Set `ImageFile.LOAD_TRUNCATED_IMAGES = True` |

---

## Troubleshooting Guide

If you encounter issues during deployment or inference, refer to these common problems and solutions:

### 1. Containers Fail to Start (Port Conflicts)
**Symptom:** Docker Compose shows an error like `bind: address already in use` for ports 8000 or 3000.
**Fix:**
- Find conflicting containers: `docker ps | grep 8000`
- Change the host port mappings in `docker-compose.yml`:
  - Backend: Change `"8000:8000"` to `"8001:8000"`
  - Frontend: Change `"3000:80"` to `"3001:80"`
- Rebuild/Restart: `docker compose up -d`

### 2. Backend Fails to Start (nvidia container runtime missing)
**Symptom:** Error `could not select device driver "" with capabilities: [[gpu]]`.
**Fix:** 
- The NVIDIA Container Toolkit is not installed on the host. 
- Follow NVIDIA's official instructions to install the toolkit and restart Docker.

### 3. "RuntimeError: Form data requires python-multipart"
**Symptom:** FastAPI backend crashes instantly when attempting to upload an image.
**Fix:**
- Verify `python-multipart>=0.0.6` is in `backend/requirements.txt`.
- Force rebuild the backend container: `docker compose build --no-cache backend`.

### 4. "RuntimeError: Error(s) in loading state_dict"
**Symptom:** Backend container crashes at startup; logs show size mismatch errors for tensors (e.g., `Expected size [32, 1, 3, 3], but got [24, 1, 3, 3]`).
**Fix:**
- This means the loaded `.pt` checkpoint file contains weights for a different model architecture (e.g., DenseNet instead of EfficientNet).
- Update the `CKPT_PATH` environment variable in `docker-compose.yml` to point to the correct `best_cxr_multilabel_13.pt` file.

### 5. "Exception: Cannot load image ... image file is truncated"
**Symptom:** Inference fails for specifically uploaded user images, often JPEGs downloaded from the web.
**Fix:**
- Ensure `backend/main.py` explicitly tells the PIL image processing library to tolerate incomplete files. 
- Ensure this line is present at the top of `main.py`: `ImageFile.LOAD_TRUNCATED_IMAGES = True`

### 6. MONAI LoadImageD Transform Failure (0-byte Images)
**Symptom:** Selecting "Auto Fill" on the frontend causes a 500 Server Error. Backend logs show MONAI choking on image files.
**Fix:**
- Verify the physical files mapped into the container aren't corrupted or 0 bytes.
- Exec into the backend: `docker exec -it monai-backend bash`
- Check file sizes: `ls -la /app/outputs/val_samples/`
- If files are 0 bytes, replace them on the host system from a fresh backup (e.g., `val_samples_1/`) and restart the container.

### 7. High-Resolution JPEGs Fail Inference
**Symptom:** User uploads work for PNGs but fail for normal phone photos or WebP images.
**Fix:**
- MONAI's native generic loader can struggle with complex image headers or multiple channels (like RGBA WebP).
- Check `backend/main.py`’s `/infer_upload` route. Ensure the uploaded file is intercepted, opened with PIL, stripped of color data `.convert("L")`, and saved locally as a simple PNG *before* passing the path to the `infer_with_gradcam` function.

---

## Stopping / Restarting

```bash
# Stop all services
docker compose down

# Restart
docker compose up -d

# Rebuild after code changes
docker compose build && docker compose up -d

# View logs
docker logs monai-backend -f
docker logs monai-frontend -f
```
