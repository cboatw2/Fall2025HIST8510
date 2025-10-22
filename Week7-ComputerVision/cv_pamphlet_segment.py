"""
cv_pamphlet_segment.py

Scan images inside the CivilWarPamphletTrial folder, classify pages as `drawing` or `text`,
and for text pages try to segment them into sections based on whitespace, divider lines,
and font-size proxies.

Outputs:
 - outputs/annotated_<original>.jpg  (visualization with boxes and labels)
 - outputs/<original>.json            (metadata: page_type, sections with bbox)

Usage:
  python3 cv_pamphlet_segment.py --input-folder CivilWarPamphletTrial

Adjustable parameters are near the top of the file.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ----- Parameters you can tune -----
ADAPTIVE_BLOCK_SIZE = 31  # odd int used for local thresholding (bigger -> fewer small blocks)
TEXT_EDGE_DENSITY_THRESHOLD = 0.018  # fraction of edge pixels that suggests text vs drawing
MIN_SECTION_HEIGHT = 30  # minimum height in pixels for a text section
TEXT_CLOSE_V = 25  # vertical gap threshold to merge nearby text lines/blocks
DIVIDER_LINE_WIDTH = 3  # thickness to consider a horizontal divider line
# -----------------------------------


def list_images(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def is_text_page(img_gray: np.ndarray) -> tuple[bool, dict]:
    """Heuristic classifier to decide whether page contains mostly text.

    Returns (is_text, stats)
    stats includes edge_density and other metrics useful for debugging.
    """
    # compute edges
    edges = cv2.Canny(img_gray, 50, 150)
    # edges is 0 or 255; normalize to fraction of edge pixels in [0,1]
    edge_density = float((edges > 0).mean())

    # compute stroke width variability proxy: distance transform on inverted binary
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    dist = cv2.distanceTransform((inv > 0).astype(np.uint8), cv2.DIST_L2, 5)
    sw_mean = dist[inv > 0].mean() if np.any(inv > 0) else 0
    sw_std = dist[inv > 0].std() if np.any(inv > 0) else 0

    # heuristic: text pages have low edge density but many short edges; drawings often have high edge density
    is_text = edge_density < TEXT_EDGE_DENSITY_THRESHOLD and sw_mean < 10

    stats = {'edge_density': float(edge_density), 'sw_mean': float(sw_mean), 'sw_std': float(sw_std)}
    return is_text, stats


def detect_divider_lines(img_gray: np.ndarray):
    """Return list of horizontal divider lines as y-coordinates (start_y, end_y).
    Uses morphological closing and Hough line or contour detection to find strong horizontal lines.
    """
    h, w = img_gray.shape
    # emphasize horizontal structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    morph = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    # threshold to binary
    _, th = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # require a long horizontal shape
        if cw > w * 0.5 and ch <= DIVIDER_LINE_WIDTH * 6:
            lines.append((y, y + ch))
    # sort by y
    lines.sort()
    return lines


def segment_text_sections(img_gray: np.ndarray):
    """Segment a text page into vertical sections based on horizontal whitespace and divider lines.

    Returns list of section bboxes (x, y, w, h)
    """
    h, w = img_gray.shape

    # Binarize with adaptive threshold to capture different font/ink
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, 10)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # project horizontally to find large vertical whitespace bands
    col_sum = th.sum(axis=1)  # sum of activated pixels per row
    # normalize
    col_sum_norm = col_sum / float(w * 255)

    # find candidate separators where col_sum is small (i.e., whitespace)
    separator_rows = np.where(col_sum_norm < 0.002)[0]

    # group consecutive rows into bands
    bands = []
    if separator_rows.size:
        start = separator_rows[0]
        prev = start
        for r in separator_rows[1:]:
            if r - prev > TEXT_CLOSE_V:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))

    # compute section y-ranges between bands and divider lines
    dividers = detect_divider_lines(img_gray)

    # assemble separator y positions
    sep_positions = [0]
    for b in bands:
        mid = (b[0] + b[1]) // 2
        sep_positions.append(mid)
    for d in dividers:
        sep_positions.append(d[0])
        sep_positions.append(d[1])
    sep_positions.append(h)

    sep_positions = sorted(set(sep_positions))

    sections = []
    for i in range(len(sep_positions) - 1):
        y0 = sep_positions[i]
        y1 = sep_positions[i + 1]
        if y1 - y0 < MIN_SECTION_HEIGHT:
            continue
        # crop and check if there's enough ink
        crop = th[y0:y1, :]
        ink_frac = (crop > 0).mean()
        if ink_frac < 0.001:
            continue
        sections.append((0, int(y0), w, int(y1 - y0)))

    return sections


def detect_boxes(img_gray: np.ndarray, th_binary: np.ndarray):
    """Detect rectangular boxes (text blocks or image illustrations) using connected components
    and morphological merging. Returns list of (x, y, w, h).
    """
    # Dilate to merge nearby text lines into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dil = cv2.dilate(th_binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = img_gray.shape
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # filter very small
        if cw < 20 or ch < 20:
            continue
        # clamp
        x = max(0, x)
        y = max(0, y)
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        boxes.append((x, y, cw, ch))

    # sort top-to-bottom
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def classify_box(img_gray: np.ndarray, box: tuple):
    """Classify a box as 'text' or 'image' using aspect ratio, edge density, and ink coverage.
    """
    x, y, w, h = box
    crop = img_gray[y:y + h, x:x + w]
    # binarize
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_frac = (bw == 0).mean()
    edges = cv2.Canny(crop, 50, 150)
    edge_density = float((edges > 0).mean())

    aspect = w / float(h)
    # heuristics: images often have higher edge density and larger ink fraction and varied aspect
    if edge_density > 0.06 or ink_frac > 0.12 or aspect > 3.0 or aspect < 0.33:
        return 'image', {'edge_density': edge_density, 'ink_frac': ink_frac, 'aspect': aspect}
    else:
        return 'text', {'edge_density': edge_density, 'ink_frac': ink_frac, 'aspect': aspect}


def annotate_and_save(img_orig, page_type, sections, out_img_path, out_meta_path):
    vis = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR) if len(img_orig.shape) == 2 else img_orig.copy()
    h, w = img_orig.shape[:2]
    cv2.putText(vis, page_type, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for i, (x, y, ww, hh) in enumerate(sections):
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.putText(vis, f's{i+1}', (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(out_img_path), vis)
    meta = {'page_type': page_type, 'sections': [{'x': int(x), 'y': int(y), 'w': int(ww), 'h': int(hh)} for (x, y, ww, hh) in sections]}
    with open(out_meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def process_folder(input_folder: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    images = list_images(input_folder)
    results = []
    for p in images:
        print('Processing', p.name)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print('  failed to read', p)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        is_text, stats = is_text_page(gray)

        # binary used for block detection
        th_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, 10)

        boxes = detect_boxes(gray, th_bin)
        boxes_info = []
        crops_dir = output_folder / 'crops' / p.stem
        crops_dir.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(boxes):
            label, bstats = classify_box(gray, b)
            x, y, cw, ch = b
            crop = cv2.cvtColor(img[y:y + ch, x:x + cw], cv2.COLOR_BGR2RGB) if img.ndim == 3 else img[y:y + ch, x:x + cw]
            crop_path = crops_dir / f'{p.stem}_b{i+1}_{label}.jpg'
            cv2.imwrite(str(crop_path), crop)
            boxes_info.append({'id': i + 1, 'label': label, 'bbox': {'x': int(x), 'y': int(y), 'w': int(cw), 'h': int(ch)}, 'stats': bstats, 'crop': str(crop_path)})

        # For backward compatibility keep sections (coarse segmentation)
        if is_text:
            sections = segment_text_sections(gray)
            page_type = 'text'
        else:
            sections = []
            page_type = 'drawing'

        out_img = output_folder / f'annotated_{p.name}'
        out_meta = output_folder / f'{p.stem}.json'
        # annotate with boxes and sections
        annotate_and_save(gray, page_type, sections, out_img, out_meta)
        # enrich meta with boxes
        with open(out_meta, 'r') as f:
            meta = json.load(f)
        meta['boxes'] = boxes_info
        with open(out_meta, 'w') as f:
            json.dump(meta, f, indent=2)

        results.append({'file': p.name, 'page_type': page_type, 'stats': stats, 'n_sections': len(sections), 'n_boxes': len(boxes)})

    summary_path = output_folder / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Done. Outputs in', output_folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-folder', default='Week7-ComputerVision/CivilWarPamphletTrial')
    ap.add_argument('--output-folder', default='Week7-ComputerVision/outputs')
    args = ap.parse_args()

    in_folder = Path(args.input_folder)
    out_folder = Path(args.output_folder)
    if not in_folder.exists():
        print('Input folder not found:', in_folder)
        return
    process_folder(in_folder, out_folder)


if __name__ == '__main__':
    main()
