"""
AlbaID — Shared image processing utilities.
Used by both analysis.ipynb and app.py.
"""

import cv2
import numpy as np

ORB_THRESHOLD = 72

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def extract_face(img_bgr: np.ndarray, target_size: int = 220):
    """
    Detect face via Haar cascade and crop to target_size x target_size.
    Returns (face_crop, bbox) — bbox is (x, y, w, h) or None if no face found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
        )
    if len(faces) == 0:
        return None, None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face = img_bgr[y : y + h, x : x + w]
    face_resized = cv2.resize(face, (target_size, target_size))
    return face_resized, (x, y, w, h)


def compute_orb_score(face1: np.ndarray, face2: np.ndarray) -> float:
    """
    Compute ORB-based similarity between two face crops.
    Returns a score 0-100 (higher = more similar).
    """
    if face1 is None or face2 is None:
        return 0.0

    f1 = cv2.resize(face1, (220, 220))
    f2 = cv2.resize(face2, (220, 220))

    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
        if matches is None or len(matches) == 0:
            return 0.0
        matches = sorted(matches, key=lambda m: m.distance)
        num_good = sum(1 for m in matches if m.distance < 40)
        score = (num_good / min(len(kp1), len(kp2), 1)) * 100
        return min(score, 100.0)
    except cv2.error:
        return 0.0


def verify_faces(face1: np.ndarray, face2: np.ndarray) -> dict:
    """
    Full verification result for a face pair.
    Returns dict with verdict, score, and ORB_THRESHOLD used.
    """
    score = compute_orb_score(face1, face2)
    if score >= ORB_THRESHOLD:
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"
    return {
        "verdict": verdict,
        "score": round(score, 1),
        "threshold": ORB_THRESHOLD,
        "match": verdict == "MATCH",
    }


def detect_card_boundary(img: np.ndarray):
    """
    Find the ID card quadrilateral boundary via Canny + contours.
    Returns (warped, corners, annotated) — warped is perspective-corrected card,
    corners is the 4x2 point array (or None), annotated is the input image
    with boundary drawn on.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = approx
            break

    annotated = img.copy()
    corners = None

    if card_contour is not None:
        cv2.drawContours(annotated, [card_contour], -1, (0, 255, 0), 2)
        pts = card_contour.reshape(4, 2).astype(np.float32)
        pts_ordered = _order_points(pts)
        w = int(max(np.linalg.norm(pts_ordered[0] - pts_ordered[1]),
                    np.linalg.norm(pts_ordered[3] - pts_ordered[2])))
        h = int(max(np.linalg.norm(pts_ordered[1] - pts_ordered[2]),
                    np.linalg.norm(pts_ordered[0] - pts_ordered[3])))
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_ordered, dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        corners = pts_ordered
        return warped, corners, annotated

    return img.copy(), None, annotated


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8):
    """
    Apply CLAHE to the L* channel of LAB-converted image.
    Preserves color while boosting local contrast.
    Returns the enhanced BGR image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced