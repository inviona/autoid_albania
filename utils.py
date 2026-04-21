"""
utils.py — Shared image processing functions for AlbaID.
Used by both analysis.ipynb and app.py to ensure the notebook
analysis directly informs the production pipeline.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

# ─────────────────────────────────────────────
# THRESHOLD — derived from local dataset analysis
# ORB match score: genuine pairs mean ~67, impostor pairs mean ~60
# Optimal separation point determined by ROC analysis on 20+ pairs
# Best balance at threshold 72: TAR=40%, FAR=20%, accuracy=60%
# ─────────────────────────────────────────────
ORB_THRESHOLD = 72  # matches/100 keypoints — optimized from local dataset


# ─────────────────────────────────────────────
# GRAYSCALE  (manual numpy — matches notebook Section 2.1)
# ─────────────────────────────────────────────
def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """L = 0.299R + 0.587G + 0.114B"""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


# ─────────────────────────────────────────────
# CLAHE enhancement
# ─────────────────────────────────────────────
def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L* channel of LAB colorspace."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.pkl"
_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def detect_faces(img_bgr: np.ndarray):
    """Return list of (x, y, w, h) face bounding boxes."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return faces if len(faces) > 0 else []


def extract_face(img_bgr: np.ndarray, target_size=(220, 220)):
    """
    Detect largest face, crop with 20% padding, resize to target_size.
    Returns (face_img, bbox) or (None, None) if no face found.
    """
    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        return None, None

    # pick largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # 20% padding
    pad_x = int(w * 0.20)
    pad_y = int(h * 0.20)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_bgr.shape[1], x + w + pad_x)
    y2 = min(img_bgr.shape[0], y + h + pad_y)

    face = img_bgr[y1:y2, x1:x2]
    face_resized = cv2.resize(face, target_size)
    face_enhanced = apply_clahe(face_resized)

    return face_enhanced, (x1, y1, x2, y2)


# ─────────────────────────────────────────────
# CARD BOUNDARY DETECTION
# ─────────────────────────────────────────────
def detect_card_boundary(img_bgr: np.ndarray):
    """
    Canny → contours → find largest quadrilateral → perspective warp.
    Returns (warped_img, corners, annotated_img) or (None, None, img) if not found.
    """
    annotated = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # dilate to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    card_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = approx
            break

    if card_contour is None:
        return None, None, annotated

    cv2.drawContours(annotated, [card_contour], -1, (0, 255, 0), 3)

    # perspective warp — standard ID card ratio 85.6×54mm → ~1.585
    pts = card_contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)

    w_top = np.linalg.norm(rect[1] - rect[0])
    w_bot = np.linalg.norm(rect[2] - rect[3])
    h_left = np.linalg.norm(rect[3] - rect[0])
    h_right = np.linalg.norm(rect[2] - rect[1])

    W = int(max(w_top, w_bot))
    H = int(max(h_left, h_right))
    if W == 0 or H == 0:
        return None, None, annotated

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H))

    return warped, rect, annotated


def _order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ─────────────────────────────────────────────
# ORB FACE VERIFICATION
# ─────────────────────────────────────────────
def compute_orb_score(face1_bgr: np.ndarray, face2_bgr: np.ndarray) -> float:
    """
    Compare two face images using ORB descriptors + BFMatcher.
    Returns similarity score (0–100). Threshold from notebook: ORB_THRESHOLD.
    """
    orb = cv2.ORB_create(nfeatures=500)
    gray1 = cv2.cvtColor(face1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Good matches: distance < 60 (tight threshold for face similarity)
    good = [m for m in matches if m.distance < 60]
    n_kp = min(len(kp1), len(kp2), 100)
    score = (len(good) / n_kp) * 100 if n_kp > 0 else 0.0
    return round(min(score, 100.0), 2)


def verify_faces(face_id: np.ndarray, face_selfie: np.ndarray):
    """
    Returns (verified: bool, score: float, message: str).
    Uses ORB_THRESHOLD from notebook analysis.
    """
    score = compute_orb_score(face_id, face_selfie)
    verified = score >= ORB_THRESHOLD
    if verified:
        msg = f"Identity verified — {score:.1f}% feature match"
    else:
        msg = f"Verification failed — only {score:.1f}% feature match (need {ORB_THRESHOLD}%)"
    return verified, score, msg


# ─────────────────────────────────────────────
# LIVENESS DETECTION
# ─────────────────────────────────────────────
def detect_face_center(img_bgr: np.ndarray):
    """Return (cx, cy) of largest detected face, or None."""
    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return (x + w // 2, y + h // 2)


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────
def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode base64 JPEG/PNG → BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image_b64(img_bgr: np.ndarray, ext=".jpg") -> str:
    """Encode BGR numpy array → base64 string."""
    _, buf = cv2.imencode(ext, img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def img_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def save_temp(img_bgr: np.ndarray, filename: str, folder="uploads") -> str:
    """Save image to uploads folder, return path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, img_bgr)
    return path
