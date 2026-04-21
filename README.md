# AlbaID — Albanian ID/Passport Renewal System

**Sindi Gugalli & Inviona Hoxha | Introduction to Computer Graphics**

A complete Digital Image Processing (DIP) pipeline for Albanian National ID card identity verification, using **OpenCV ORB descriptors** for facial similarity scoring.

---

## Project Structure

```text
albaid/
├── app.py               # Flask Web Application (UI + API + PDF Generation)
├── analysis.ipynb       # Academic notebook (EDA + filters + ORB threshold derivation)
├── utils.py             # Shared image processing utilities
├── dataset_id/          # Place your dataset here
├── templates/
│   ├── index.html       # Step 1: Choose Service
│   ├── upload.html      # Step 2: Upload ID Card
│   ├── scan.html        # Step 3: Live Selfie Scan
│   └── result.html      # Step 4: Verification & PDF Form
├── static/
│   ├── uploads/         # Temporary upload storage
│   └── css/             # Stylesheets (if any)
├── flask_app/           # (Legacy API reference)
├── pipeline/            # (Legacy CLI pipeline)  
├── requirements.txt
└── README.md
```

---

## ORB Threshold

The similarity threshold (`ORB_THRESHOLD = 72`) is derived in `notebook/analysis.ipynb` Section 3 from ROC analysis on 40+ face pairs from the AxonLabs dataset. This exact value is imported by `app.py` via `utils.py`.

| ORB Score | Verdict |
|-----------|---------|
| ≥ 72      | **MATCH** |
| < 72      | **MISMATCH** |

---

## Quick Start

### 1. Install dependencies

```bash
cd albaid
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Set up dataset

Download the AxonLabs Selfie & ID Photo Dataset from Kaggle and extract to `dataset_id/` at the project root, so the notebook can load it:

```
albaid/
  dataset_id/
    AxonLabs_Diverse Selfie & ID Photo Dat.../
    Selfie & id data - public sample/
    metadata_images.csv
    meta_public.json
```

> Replace with real Albanian ID photos and selfies (with subject consent) for production use.

### 3. Run the Jupyter Notebook (recommended for analysis)

```bash
jupyter notebook analysis.ipynb
```

The notebook walks through:
- **Section 1** — Dataset EDA (image size, aspect ratio, class distribution)
- **Section 2** — Manual NumPy image filters (grayscale, histogram equalization, CLAHE, Gaussian blur, Sobel, FFT, morphology)
- **Section 3** — ORB threshold derivation (ROC on genuine + impostor pairs)
- **Section 4** — ID card pipeline (Canny → boundary → warp → CLAHE → face extraction)

### 4. Run the Flask web server

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|-------|----------|-------------|
| GET   | `/`      | Web interface Step 1 (Service Selection) |
| GET   | `/upload`| Web interface Step 2 (ID Upload) |
| GET   | `/scan`  | Web interface Step 3 (Live Selfie Scan) |
| GET   | `/result`| Web interface Step 4 (Verification Result & Form) |
| POST  | `/api/process-id` | Upload and process the ID card image |
| POST  | `/api/liveness` | Webcam head-turn detection |
| POST  | `/api/capture-selfie` | Upload and process selfie frame |
| POST  | `/api/verify` | ORB face comparison API |
| POST  | `/api/generate-pdf`| Generates a verification report PDF using reportlab |
| GET   | `/health` | Application health and threshold check |

### 4-Step Verification Flow

1. **Service Selection:** Choose between ID Card or Passport renewal (mock payment included).
2. **ID Upload:** Upload a scan of an ID. The system detects the card boundary, automatically warps it flat, applies CLAHE, and extracts the face.
3. **Live Scan:** Through the webcam, the system performs liveness detection (verifying a face is present) and automatically captures a selfie crop.
4. **Verification & PDF:** The selfies are compared against the ID face using the derived ORB threshold. The UI generates a full PDF report detailing the biometric metrics and applicant information.

---

## Core Functions (`utils.py`)

- `extract_face(img_bgr, target_size=220)` — Haar cascade face detection → 220x220 crop
- `compute_orb_score(face1, face2)` — ORB descriptor matching → 0-100 score
- `verify_faces(face1, face2)` — Returns verdict dict with score and threshold
- `detect_card_boundary(img)` — Canny + contour → quadrilateral → perspective warp
- `apply_clahe(img)` — CLAHE on L* channel (LAB colorspace)

---

## Pipeline Overview (academic notebook)

### Section 2 Filters (manual NumPy)
| Filter | Formula |
|--------|---------|
| Grayscale | L = 0.299R + 0.587G + 0.114B |
| Histogram Equalization | s_k = (L-1)·CDF(r_k) |
| CLAHE | Adaptive, clipLimit=2, 8×8 tiles |
| Gaussian Blur | G(i,j) = (1/2πσ²)·exp(-(i²+j²)/2σ²) |
| Sobel Edge | G = √(Gx²+Gy²) |
| FFT2 | Low-pass / High-pass ideal filters |
| Morphology | Erosion, Dilation, Opening, Closing |

### Section 3 Threshold Derivation
- **Genuine pairs** — same person (ID card vs selfie)
- **Impostor pairs** — different people
- **Threshold = 72** — derived from ROC analysis: TAR=40%, FAR=25%, accuracy=57.5%

### Section 4 ID Card Pipeline
```
Original → Canny Edges → Card Boundary Detection
       → Perspective Warp → CLAHE Enhancement
       → Face Extraction → ORB Scoring
```

---

## References

- Sindi Gugalli & Inviona Hoxha, *AlbaID — Albanian ID/Passport Renewal System*, Introduction to Computer Graphics course report
- OpenCV documentation: https://docs.opencv.org
- AxonLabs Selfie & ID Photo Dataset (Kaggle)
- Albanian Smart Nation strategy / e-Albania platform