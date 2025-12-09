import math
import numpy as np
import pandas as pd
from src.pyceph.Landmarks import Landmarks

def get_point(landmarks, name_enum):
    """Helper to find coordinate (x, y) by Landmark Enum."""
    # Handle list of dicts (from Canvas) or list of tuples (from AI)
    if isinstance(landmarks, list) and len(landmarks) > 0:
        # Check if it's a list of tuples/lists or dicts
        if isinstance(landmarks[0], (tuple, list, np.ndarray)):
             idx = name_enum.value
             if idx < len(landmarks):
                 return np.array(landmarks[idx])
        elif isinstance(landmarks[0], dict):
            # Search by ID or Label in the dictionary list
            for obj in landmarks:
                if obj.get('id') == name_enum.value:
                    return np.array([obj['x'], obj['y']])
    return None

def calculate_distance(p1, p2, scale_factor=1.0):
    """Euclidean distance in mm."""
    dist_px = np.linalg.norm(p1 - p2)
    return dist_px * scale_factor

def calculate_angle(p1, p2, p3):
    """Angle at p2 (p1-p2-p3) in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def perform_cephalometric_analysis(landmarks_data, calibration_scale=1.0):
    """
    Runs analysis and returns a pandas DataFrame for display.
    landmarks_data: list of tuples (x,y) OR list of dicts {'id':0, 'x':.., 'y':..}
    """
    results = []

    # Map landmarks for easy access
    pts = {}
    for lm in Landmarks:
        pt = get_point(landmarks_data, lm)
        if pt is not None:
            pts[lm] = pt

    # --- ANGLES ---
    # SNA
    if all(k in pts for k in [Landmarks.SELLA, Landmarks.NASION, Landmarks.SUBSPINALE]):
        val = calculate_angle(pts[Landmarks.SELLA], pts[Landmarks.NASION], pts[Landmarks.SUBSPINALE])
        results.append({"Measurement": "SNA", "Value": f"{val:.2f}°", "Norm": "82° ± 2°"})

    # SNB
    if all(k in pts for k in [Landmarks.SELLA, Landmarks.NASION, Landmarks.SUPRAMENTALE]):
        val = calculate_angle(pts[Landmarks.SELLA], pts[Landmarks.NASION], pts[Landmarks.SUPRAMENTALE])
        results.append({"Measurement": "SNB", "Value": f"{val:.2f}°", "Norm": "80° ± 2°"})

    # ANB (Calculated from values above if possible, else directly)
    if all(k in pts for k in [Landmarks.SUBSPINALE, Landmarks.NASION, Landmarks.SUPRAMENTALE]):
        val = calculate_angle(pts[Landmarks.SUBSPINALE], pts[Landmarks.NASION], pts[Landmarks.SUPRAMENTALE])
        # ANB is typically SNA - SNB, but geometric angle works too. Let's use simple subtraction if avail.
        sna = [r for r in results if r["Measurement"] == "SNA"]
        snb = [r for r in results if r["Measurement"] == "SNB"]
        if sna and snb:
            anb = float(sna[0]["Value"][:-1]) - float(snb[0]["Value"][:-1])
            results.append({"Measurement": "ANB", "Value": f"{anb:.2f}°", "Norm": "2° ± 2°"})

    # --- DISTANCES ---
    # Mandibular Length (Co-Gn). Using Articulare as proxy for Condylion if missing.
    if all(k in pts for k in [Landmarks.ARTICULARE, Landmarks.GNATHION]):
        val = calculate_distance(pts[Landmarks.ARTICULARE], pts[Landmarks.GNATHION], calibration_scale)
        results.append({"Measurement": "Mandibular Length (Ar-Gn)", "Value": f"{val:.2f} mm", "Norm": "105-115 mm"})

    # Anterior Facial Height (N-Me)
    if all(k in pts for k in [Landmarks.NASION, Landmarks.MENTON]):
        val = calculate_distance(pts[Landmarks.NASION], pts[Landmarks.MENTON], calibration_scale)
        results.append({"Measurement": "Ant. Face Hgt (N-Me)", "Value": f"{val:.2f} mm", "Norm": "105-120 mm"})

    return pd.DataFrame(results)