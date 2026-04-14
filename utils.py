import math
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize
from scipy.ndimage import label
from scipy.signal import convolve2d

IS_456_CRACK_LIMITS_MM = {
    "Mild": 0.30,
    "Moderate": 0.20,
    "Severe": 0.10,
    "Very Severe": 0.10,
    "Extreme": 0.10,
}

def _filter_components(binary: np.ndarray) -> np.ndarray:
    labeled, num = label(binary)
    filtered = np.zeros_like(binary, dtype=bool)

    for i in range(1, num + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < 25:
            continue

        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue

        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        major = max(w, h)
        minor = max(1, min(w, h))
        aspect = major / minor
        fill_ratio = area / float(max(1, w * h))

        if aspect >= 2.0 or (aspect >= 1.4 and fill_ratio <= 0.45):
            filtered |= comp

    return filtered.astype(np.uint8)

def segment_crack(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)
    binary = cv2.medianBlur(binary, 3)

    binary = (binary > 0).astype(bool)
    binary = remove_small_objects(binary, min_size=25)
    binary = _filter_components(binary.astype(np.uint8)).astype(bool)
    return binary.astype(np.uint8)

def get_skeleton(binary: np.ndarray) -> np.ndarray:
    return skeletonize(binary > 0)

def total_crack_length_pixels(binary: np.ndarray) -> float:
    skel = get_skeleton(binary)
    ys, xs = np.where(skel)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    total = 0.0

    neighbors = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
    ]

    for y, x in pixels:
        for dy, dx, w in neighbors:
            if (y + dy, x + dx) in pixels:
                total += w

    return total

def connected_crack_components(binary: np.ndarray):
    labeled, num = label(binary > 0)
    components = []

    for i in range(1, num + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area == 0:
            continue
        length_px = total_crack_length_pixels(comp.astype(np.uint8))
        if length_px <= 0:
            continue
        components.append(
            {
                "label": i,
                "mask": comp.astype(np.uint8),
                "area_px": area,
                "length_px": float(length_px),
            }
        )

    components.sort(key=lambda c: c["length_px"], reverse=True)
    return components

def classify_crack_general(max_width: float) -> str:
    if max_width <= 0.10:
        return "Very Fine / Hairline"
    if max_width <= 0.20:
        return "Fine"
    if max_width <= 0.30:
        return "Visible"
    if max_width <= 1.00:
        return "Wide"
    return "Very Wide / Severe"

def classify_crack_is456(max_width: float, exposure_class: str) -> tuple[str, float, bool]:
    limit = IS_456_CRACK_LIMITS_MM.get(exposure_class, 0.30)
    within_limit = max_width <= limit

    if within_limit:
        status = f"Within IS 456 limit for {exposure_class} exposure"
    else:
        status = f"Exceeds IS 456 limit for {exposure_class} exposure"

    return status, float(limit), bool(within_limit)

def measure_crack(binary: np.ndarray, mm_per_pixel: float, exposure_class: str = "Mild") -> dict:
    skel = get_skeleton(binary)
    if not np.any(skel):
        raise ValueError("Could not derive a crack skeleton.")

    length_px = total_crack_length_pixels(binary)
    length_mm = length_px * mm_per_pixel

    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    widths = dist[skel] * 2.0 * mm_per_pixel
    widths = widths[np.isfinite(widths)]
    widths = widths[widths > 0]

    if widths.size == 0:
        raise ValueError("Could not estimate crack width.")

    general_classification = classify_crack_general(float(widths.max()))
    is456_status, is456_limit_mm, is456_within_limit = classify_crack_is456(
        float(widths.max()), exposure_class
    )

    return {
        "skeleton": skel,
        "length_px": float(length_px),
        "length_mm": float(length_mm),
        "widths_mm": widths,
        "min_width_mm": float(widths.min()),
        "avg_width_mm": float(widths.mean()),
        "max_width_mm": float(widths.max()),
        "general_classification": general_classification,
        "is456_status": is456_status,
        "is456_limit_mm": is456_limit_mm,
        "is456_within_limit": is456_within_limit,
        "pixel_count": int((binary > 0).sum()),
    }

def crack_density_map(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((35, 35), dtype=np.float32)
    density = convolve2d(binary.astype(np.float32), kernel, mode="same", boundary="symm")
    density = density / (density.max() + 1e-6)
    img = (density * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)
