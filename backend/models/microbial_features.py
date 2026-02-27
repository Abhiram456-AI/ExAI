# backend/models/microbial_features.py

import cv2
import numpy as np
from skimage.filters import threshold_local
from scipy.spatial import ConvexHull


def extract_microbial_features(image_path: str) -> dict:

    image = cv2.imread(image_path)

    if image is None:
        return {"error": "Image could not be loaded."}

    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: Mask Petri Dish (remove background) ---
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, plate_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep largest connected component (assume plate)
    contours_plate, _ = cv2.findContours(
        plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours_plate, key=cv2.contourArea)
    plate_only = np.zeros_like(gray)
    cv2.drawContours(plate_only, [largest], -1, 255, -1)

    masked = cv2.bitwise_and(gray, gray, mask=plate_only)

    # --- STEP 2: Improved Colony Segmentation (Adaptive + Edge Preservation) ---
    blur2 = cv2.GaussianBlur(masked, (5, 5), 0)

    # Adaptive threshold to capture faint colonies
    adaptive = threshold_local(blur2, block_size=61, offset=8)
    thresh = (blur2 > adaptive).astype("uint8") * 255

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.55 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    image_color = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_color, markers)

    # Extract contours from watershed regions
    contours = []
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        region = np.uint8(markers == marker)
        cnts, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours.extend(cnts)

    features = []
    centroids = []
    total_colony_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

        total_colony_area += area

        features.append({
            "area": area,
            "circularity": circularity,
            "solidity": solidity
        })

    if not features:
        return {
            "colony_count": 0,
            "colony_density": 0,
            "coverage_ratio": 0,
            "clustering_index": 0,
            "dominance": "Low"
        }

    colony_count = len(features)
    plate_area = np.sum(plate_only > 0)
    raw_coverage = total_colony_area / plate_area if plate_area != 0 else 0
    coverage_ratio = min(raw_coverage, 1.0)

    # Density weighted by area instead of raw count
    colony_density = min((total_colony_area / plate_area) * 1.2, 1.0)

    mean_circularity = np.mean([f["circularity"] for f in features])
    mean_solidity = np.mean([f["solidity"] for f in features])
    areas = [f["area"] for f in features]
    area_variation = np.std(areas) / (np.mean(areas) + 1e-5)

    # --- Clustering Index (average nearest neighbor distance inverse normalized) ---
    if len(centroids) > 1:
        distances = []
        for i in range(len(centroids)):
            dists = [
                np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                for j in range(len(centroids)) if i != j
            ]
            distances.append(min(dists))
        mean_nn = np.mean(distances)
        clustering_index = 1 / (mean_nn + 1e-5)
        clustering_index = min(clustering_index * 25, 1.0)  # normalize
    else:
        clustering_index = 0

    # --- Dominance Logic (Calibrated Morphology Thresholds - Phase 1) ---
    # Slightly relaxed bacterial conditions to avoid over-classifying as fungal
    if (
        mean_circularity > 0.65 and
        mean_solidity > 0.75 and
        area_variation < 0.75
    ):
        dominance = "Bacterial"

    # More conservative fungal classification (avoid triggering too easily)
    elif (
        mean_circularity < 0.55 and
        mean_solidity < 0.70 and
        area_variation > 0.85
    ):
        dominance = "Fungal"

    # Everything between becomes Mixed
    else:
        dominance = "Mixed"

    return {
        "colony_count": colony_count,
        "colony_density": round(float(colony_density), 3),
        "coverage_ratio": round(float(coverage_ratio), 3),
        "clustering_index": round(float(clustering_index), 3),
        "mean_circularity": round(float(mean_circularity), 3),
        "mean_solidity": round(float(mean_solidity), 3),
        "area_variation": round(float(area_variation), 3),
        "dominance": dominance
    }