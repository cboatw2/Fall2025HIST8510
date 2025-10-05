"""
Magazine Layout Demo (OpenCV for Historians)
============================================

This demo shows how to use classic computer vision to analyze historical
magazine pages that typically contain multi-column text and advertisements.

It performs:
  1) Preprocessing (grayscale, adaptive/Otsu threshold)
  2) Vertical line detection for columns
  3) Horizontal line detection for section separators
  4) Text block detection via morphology + contours
  5) Simple heuristics to flag candidate advertisements vs text blocks
  6) Exports an annotated image and a CSV/JSON of detected regions

Run:
  python3 magazine_layout_demo.py

Inputs:
  Week7-ComputerVision/img/*.png|jpg|jpeg|tif

Outputs:
  Week7-ComputerVision/mag_output/
    - {image}_annotated.png
    - {image}_regions.csv
    - {image}_regions.json

Heuristics (simple, transparent):
  - A region is a candidate ad if it is relatively large, near page edges,
    or has aspect ratio suggestive of rectangular blocks, or very low fill
    ratio after closing (lots of whitespace inside).
  - Otherwise it is assumed to be text.

Notes for educators:
  - Parameters are grouped up top for easy experimentation in class.
  - Emphasize that these are heuristic, non-ML methods; robust solutions for
    production may involve ML or learning-based layout analysis.
"""

from pathlib import Path
import csv
import json
import cv2
import numpy as np

# ====== Parameters to experiment with in class ======
ADAPTIVE_BLOCK_SIZE = 25     # must be odd
ADAPTIVE_C = 11
VERT_KERNEL = (1, 60)
HORIZ_KERNEL = (60, 1)
TEXT_CLOSE_H = (17, 3)
TEXT_CLOSE_V = (3, 9)
MIN_TEXT_W = 30
MIN_TEXT_H = 20
MAX_ASPECT_RATIO = 14.0
AD_CANDIDATE_MIN_AREA_RATIO = 0.08   # % of page area
EDGE_MARGIN_RATIO = 0.10             # near edges -> more likely ad

# ====================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_images(images_dir: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

def binarize(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu as default for magazines; adaptive also works
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def detect_lines(binary: np.ndarray):
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, VERT_KERNEL)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HORIZ_KERNEL)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    return vertical_lines, horizontal_lines

def close_for_text(binary: np.ndarray) -> np.ndarray:
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, TEXT_CLOSE_H)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, TEXT_CLOSE_V)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_v, iterations=1)
    return closed

def find_regions(binary_closed: np.ndarray):
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_regions(contours, page_w, page_h):
    regions = []
    page_area = page_w * page_h
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_TEXT_W or h < MIN_TEXT_H:
            continue
        ratio = max(w / float(h), h / float(w))
        if ratio > MAX_ASPECT_RATIO:
            continue
        area = w * h
        regions.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "area_ratio": area / float(page_area)
        })
    return regions

def classify_region(region, page_w, page_h, binary_closed: np.ndarray):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    area_ratio = region["area_ratio"]
    # Edge proximity
    margin = EDGE_MARGIN_RATIO
    near_edge = x < page_w * margin or (x + w) > page_w * (1 - margin) or \
                y < page_h * margin or (y + h) > page_h * (1 - margin)
    # Fill ratio inside box (after closing)
    roi = binary_closed[y:y+h, x:x+w]
    fill_ratio = float(cv2.countNonZero(roi)) / (w * h)

    # Heuristic decision
    is_large = area_ratio >= AD_CANDIDATE_MIN_AREA_RATIO
    looks_blocky = 0.25 < fill_ratio < 0.75  # lots of whitespace inside
    if is_large and (near_edge or looks_blocky):
        label = "ad_candidate"
    else:
        label = "text_block"
    region.update({"fill_ratio": round(fill_ratio, 4), "label": label})
    return region

def annotate(image_bgr: np.ndarray, regions, vertical_lines, horizontal_lines):
    vis = image_bgr.copy()
    # Overlay lines
    blue = vis.copy()
    blue[vertical_lines > 0] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 0.85, blue, 0.15, 0)
    red = vis.copy()
    red[horizontal_lines > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.85, red, 0.15, 0)
    # Draw boxes
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        if r["label"] == "ad_candidate":
            color = (0, 165, 255)  # orange
        else:
            color = (0, 255, 0)    # green
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, r["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def export_csv_json(regions, csv_path: Path, json_path: Path):
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x","y","w","h","area","area_ratio","fill_ratio","label"]) 
        writer.writeheader()
        for r in regions:
            writer.writerow(r)
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2)

def process_image(img_path: Path, out_dir: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Could not read: {img_path.name}")
        return
    h, w = image.shape[:2]

    binary = binarize(image)
    vertical_lines, horizontal_lines = detect_lines(binary)
    closed = close_for_text(binary)

    contours = find_regions(closed)
    regions = filter_regions(contours, w, h)
    regions = [classify_region(r, w, h, closed) for r in regions]

    annotated = annotate(image, regions, vertical_lines, horizontal_lines)

    base = img_path.stem
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / f"{base}_annotated.png"), annotated)
    export_csv_json(regions, out_dir / f"{base}_regions.csv", out_dir / f"{base}_regions.json")
    print(f"✅ {img_path.name}: {len(regions)} regions → annotated + CSV/JSON")

def main():
    root = Path(__file__).parent
    images_dir = root / "img"
    out_dir = root / "mag_output"
    ensure_dir(out_dir)

    images = load_images(images_dir)
    if not images:
        print(f"❌ No images found in: {images_dir}")
        return 1

    for img in images:
        process_image(img, out_dir)

    print("\nDone. Explore:")
    print(f"  {out_dir}/<image>_annotated.png")
    print(f"  {out_dir}/<image>_regions.csv (spreadsheet-friendly)")
    print(f"  {out_dir}/<image>_regions.json (for code)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
