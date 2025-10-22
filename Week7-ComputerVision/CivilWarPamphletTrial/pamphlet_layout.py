#!/usr/bin/env python3
"""
Magazine Layout Analysis with OpenCV
====================================

Analyzes historical magazine pages to detect and classify layout regions:
- Text columns
- Headlines
- Advertisements
- Images

Demonstrates computer vision techniques for document layout analysis without
machine learning - using morphological operations, contour detection, and
rule-based classification.

Usage:
    python magazine_layout_demo.py [--debug] [--ocr] [--min-width 30]

Inputs:
    img/*.png|jpg|jpeg|tif

Outputs:
    mag_output/
      - {image}_annotated.png      (visualization)
      - {image}_regions.csv         (structured data)
      - {image}_regions.json        (for code integration)
      - debug/{image}_*.png         (processing steps, if --debug)
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# ====== Parameters - Adjust based on your documents ======
# Binarization
ADAPTIVE_BLOCK_SIZE = 35     # Window size for adaptive threshold (must be odd)
                             # Larger = more global, smaller = more local
ADAPTIVE_C = 11              # Constant subtracted from mean
                             # Higher = more aggressive thresholding

# Line detection kernels (width, height)
VERT_KERNEL = (1, 60)        # Detect vertical lines at least 60px tall
HORIZ_KERNEL = (60, 1)       # Detect horizontal lines at least 60px wide
                             # Increase for thicker/longer lines only

# Text region morphology
TEXT_CLOSE_H = (35, 7)       # Increase horizontal gap closing (was (17, 3))
TEXT_CLOSE_V = (12, 36)      # Increase vertical gap closing (was (6, 18))
                             # Increase to merge more aggressively

# Filtering
MIN_TEXT_W = 80              # Minimum region width (pixels)
MIN_TEXT_H = 40              # Minimum region height (pixels)
MAX_ASPECT_RATIO = 14.0      # Max width:height ratio (filters thin lines)

# Classification thresholds
IMAGE_FILL_THRESHOLD = 0.15   # Regions with <15% filled are likely images
HEADLINE_Y_THRESHOLD = 0.15   # Top 15% of page
HEADLINE_ASPECT = 3.0         # Wide and short
AD_Y_THRESHOLD = 0.5          # Bottom half of page
AD_AREA_THRESHOLD = 0.2       # Large regions (20% of page)
COLUMN_ASPECT = 0.5           # Tall and narrow
COLUMN_HEIGHT_THRESHOLD = 0.3 # At least 30% of page height

# ====================================================


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_images(images_dir: Path):
    """Find all image files in directory."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(images)


def binarize(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert image to binary (black text on white background).
    Uses Otsu's method for automatic thresholding.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV inverts so text is white (255) on black (0)
    _, binary = cv2.threshold(
        gray, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def detect_lines(binary: np.ndarray):
    """
    Detect vertical and horizontal lines using morphological operations.
    Vertical lines often indicate column separators.
    Horizontal lines often indicate section breaks.
    """
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, VERT_KERNEL)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HORIZ_KERNEL)
    
    # Opening operation: erosion followed by dilation
    # Removes everything except structures matching kernel shape
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    
    return vertical_lines, horizontal_lines


def close_for_text(binary: np.ndarray) -> np.ndarray:
    """
    Close gaps in text to form coherent blocks.
    Horizontal closing connects words into lines.
    Vertical closing connects lines into paragraphs.
    """
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, TEXT_CLOSE_H)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, TEXT_CLOSE_V)
    
    # Closing = dilation then erosion (fills gaps)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_v, iterations=1)
    
    return closed


def find_regions(binary_closed: np.ndarray):
    """Find contours (boundaries) of all regions in image."""
    contours, _ = cv2.findContours(
        binary_closed, 
        cv2.RETR_EXTERNAL,  # Only outermost contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def filter_regions(contours, page_w, page_h, min_w, min_h):
    """
    Filter out noise and keep only meaningful regions.
    Remove regions that are too small or have extreme aspect ratios.
    """
    regions = []
    page_area = page_w * page_h
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter by minimum size
        if w < min_w or h < min_h:
            continue
        
        # Filter by aspect ratio (remove very thin lines)
        ratio = max(w / float(h), h / float(w))
        if ratio > MAX_ASPECT_RATIO:
            continue
        
        area = w * h
        regions.append({
            "x": int(x), 
            "y": int(y), 
            "w": int(w), 
            "h": int(h),
            "area": int(area), 
            "area_ratio": area / float(page_area)
        })
    
    return regions


def classify_region(region, binary_closed, page_w, page_h):
    """
    Classify region type based on position, size, and content.
    
    Classification rules:
    - Images: Low fill ratio (mostly white space)
    - Headlines: Top of page, wide and short
    - Advertisements: Lower page, large area
    - Columns: Tall and narrow
    - Text blocks: Default
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    roi = binary_closed[y:y+h, x:x+w]
    
    # Calculate features
    fill_ratio = float(cv2.countNonZero(roi)) / max(1, (w * h))
    aspect_ratio = w / float(h)
    position_y_ratio = y / float(page_h)
    
    # Classification logic
    label = "text_block"  # default
    
    # Images have lower fill ratios (more white space)
    if fill_ratio < IMAGE_FILL_THRESHOLD:
        label = "image"
    # Wide, short regions at top are likely headlines
    elif position_y_ratio < HEADLINE_Y_THRESHOLD and aspect_ratio > HEADLINE_ASPECT:
        label = "headline"
    # Large regions in lower half with mixed content
    elif position_y_ratio > AD_Y_THRESHOLD and region["area_ratio"] > AD_AREA_THRESHOLD:
        label = "advertisement"
    # Tall narrow regions are likely columns
    elif aspect_ratio < COLUMN_ASPECT and h > page_h * COLUMN_HEIGHT_THRESHOLD:
        label = "column"
    
    region.update({
        "fill_ratio": round(fill_ratio, 4), 
        "aspect_ratio": round(aspect_ratio, 2),
        "label": label
    })
    
    return region


def order_regions_reading_flow(regions, page_w):
    """
    Order regions in natural reading flow: top-to-bottom, left-to-right.
    Groups regions at similar y-positions into "rows", then sorts within rows.
    """
    if not regions:
        return regions
    
    # Sort by y-position first (top to bottom)
    regions_sorted = sorted(regions, key=lambda r: r["y"])
    
    # Group regions that are at similar y-positions (same "row")
    rows = []
    current_row = [regions_sorted[0]]
    y_threshold = 50  # pixels tolerance for "same row"
    
    for r in regions_sorted[1:]:
        if abs(r["y"] - current_row[0]["y"]) < y_threshold:
            current_row.append(r)
        else:
            # Sort current row left to right, add to rows
            rows.append(sorted(current_row, key=lambda x: x["x"]))
            current_row = [r]
    
    # Don't forget last row
    rows.append(sorted(current_row, key=lambda x: x["x"]))
    
    # Flatten and add reading_order field
    ordered = []
    for row in rows:
        ordered.extend(row)
    
    for i, region in enumerate(ordered):
        region["reading_order"] = i + 1
    
    return ordered


def extract_text_from_regions(image_bgr, regions):
    """
    Run OCR on each detected region.
    Chooses appropriate PSM mode based on region type.
    """
    try:
        import pytesseract
    except ImportError:
        print("Warning: pytesseract not installed. Skipping text extraction.")
        print("Install with: pip install pytesseract")
        return regions
    
    for region in regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        roi = image_bgr[y:y+h, x:x+w]
        
        # Convert to grayscale for OCR
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Choose PSM based on region type
        if region["label"] == "column":
            config = '--psm 4'  # Single column of text
        elif region["label"] == "headline":
            config = '--psm 7'  # Single text line
        elif region["label"] == "image":
            config = '--psm 11'  # Sparse text
        else:
            config = '--psm 6'  # Uniform block of text
        
        try:
            text = pytesseract.image_to_string(roi_gray, config=config)
            region["text"] = text.strip()
            region["word_count"] = len(text.split())
        except Exception as e:
            region["text"] = f"[OCR Error: {e}]"
            region["word_count"] = 0
    
    return regions


def regions_overlap(r1, r2, threshold=0.5):
    """Check if two regions overlap significantly."""
    x_overlap = max(0, min(r1["x"] + r1["w"], r2["x"] + r2["w"]) - 
                       max(r1["x"], r2["x"]))
    y_overlap = max(0, min(r1["y"] + r1["h"], r2["y"] + r2["h"]) - 
                       max(r1["y"], r2["y"]))
    overlap_area = x_overlap * y_overlap
    min_area = min(r1["area"], r2["area"])
    return (overlap_area / min_area) > threshold if min_area > 0 else False


def validate_regions(regions, page_w, page_h):
    """
    Validate detection results and warn about potential issues.
    Returns list of warning messages.
    """
    issues = []
    
    if not regions:
        issues.append("No regions detected - image may be too faded or parameters need adjustment")
        return issues
    
    # Check for overlapping regions
    overlaps = 0
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions[i+1:], i+1):
            if regions_overlap(r1, r2):
                overlaps += 1
    
    if overlaps > 0:
        issues.append(f"{overlaps} overlapping region pair(s) detected - may need parameter tuning")
    
    # Check coverage
    total_area = sum(r["area"] for r in regions)
    coverage = total_area / (page_w * page_h)
    
    if coverage > 0.95:
        issues.append(f"Very high coverage ({coverage:.1%}) - may be over-detecting regions")
    elif coverage < 0.10:
        issues.append(f"Very low coverage ({coverage:.1%}) - may be under-detecting regions")
    
    # Check for reasonable distribution of labels
    label_counts = {}
    for r in regions:
        label = r.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    if len(label_counts) == 1:
        issues.append(f"All regions classified as '{list(label_counts.keys())[0]}' - classification may need tuning")
    
    return issues


def annotate(image_bgr, regions, vertical_lines, horizontal_lines):
    """
    Create annotated visualization showing detected regions and lines.
    """
    vis = image_bgr.copy()
    
    # Overlay vertical lines in blue
    blue = vis.copy()
    blue[vertical_lines > 0] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 0.85, blue, 0.15, 0)
    
    # Overlay horizontal lines in red
    red = vis.copy()
    red[horizontal_lines > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.85, red, 0.15, 0)
    
    # Color scheme for region types
    colors = {
        "headline": (255, 0, 255),      # Magenta
        "column": (0, 255, 0),          # Green
        "text_block": (0, 255, 0),      # Green
        "advertisement": (0, 165, 255), # Orange
        "image": (255, 255, 0)          # Cyan
    }
    
    # Draw boxes around regions
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        label = r.get("label", "unknown")
        color = colors.get(label, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Add label and reading order
        reading_order = r.get("reading_order", "?")
        label_text = f"{reading_order}: {label}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(vis, (x, y - text_h - 8), 
                     (x + text_w + 4, y), color, -1)
        
        # Text label
        cv2.putText(vis, label_text, (x + 2, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return vis


def save_debug_images(image_path, binary, vertical_lines, horizontal_lines, 
                     closed, out_dir):
    """Save intermediate processing steps for educational purposes."""
    base = image_path.stem
    debug_dir = out_dir / "debug"
    ensure_dir(debug_dir)
    
    cv2.imwrite(str(debug_dir / f"{base}_1_binary.png"), binary)
    cv2.imwrite(str(debug_dir / f"{base}_2_vertical_lines.png"), vertical_lines)
    cv2.imwrite(str(debug_dir / f"{base}_3_horizontal_lines.png"), horizontal_lines)
    cv2.imwrite(str(debug_dir / f"{base}_4_closed.png"), closed)
    
    print(f"  Debug images saved to {debug_dir}/")


def export_csv_json(regions, csv_path: Path, json_path: Path):
    """Export regions to CSV and JSON formats."""
    if not regions:
        return
    
    # Determine fields (some regions may have text, others may not)
    base_fields = ["reading_order", "x", "y", "w", "h", "area", 
                   "area_ratio", "fill_ratio", "aspect_ratio", "label"]
    has_text = any("text" in r for r in regions)
    if has_text:
        base_fields.extend(["word_count", "text"])
    
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction='ignore')
        writer.writeheader()
        for r in regions:
            writer.writerow(r)
    
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2)


def process_image(img_path: Path, out_dir: Path, args):
    """Process a single image through the layout analysis pipeline."""
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not read {img_path.name}")
        return
    
    h, w = image.shape[:2]
    print(f"\nProcessing: {img_path.name} ({w}x{h})")
    
    # Step 1: Binarization
    binary = binarize(image)
    
    # Step 2: Line detection
    vertical_lines, horizontal_lines = detect_lines(binary)
    
    # Step 3: Morphological closing to form text blocks
    closed = close_for_text(binary)
    
    # Step 4: Find contours
    contours = find_regions(closed)
    
    # Step 5: Filter and classify regions
    regions = filter_regions(contours, w, h, args.min_width, args.min_height)
    regions = [classify_region(r, closed, w, h) for r in regions]
    
    # Step 6: Order regions by reading flow
    regions = order_regions_reading_flow(regions, w)
    
    # Step 7: Extract text if requested
    if args.ocr:
        print("  Running OCR on detected regions...")
        regions = extract_text_from_regions(image, regions)
    
    # Step 8: Validate results
    issues = validate_regions(regions, w, h)
    if issues:
        print("  Warnings:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Step 9: Create visualization
    annotated = annotate(image, regions, vertical_lines, horizontal_lines)
    
    # Step 10: Save outputs
    base = img_path.stem
    ensure_dir(out_dir)
    
    cv2.imwrite(str(out_dir / f"{base}_annotated.png"), annotated)
    export_csv_json(regions, out_dir / f"{base}_regions.csv", 
                   out_dir / f"{base}_regions.json")
    
    print(f"  Detected {len(regions)} regions")
    
    # Count by type
    label_counts = {}
    for r in regions:
        label = r.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"  By type: {dict(label_counts)}")
    
    # Save debug images if requested
    if args.debug:
        save_debug_images(img_path, binary, vertical_lines, horizontal_lines, 
                         closed, out_dir)
    
    print(f"  Outputs saved to {out_dir}/{base}_*")


def main():
    """Main function to run layout analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze historical magazine layouts with OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python magazine_layout_demo.py
  python magazine_layout_demo.py --debug --ocr
  python magazine_layout_demo.py --min-width 50 --min-height 30
        """
    )
    
    parser.add_argument('--debug', action='store_true',
                       help='Save intermediate processing steps for inspection')
    parser.add_argument('--ocr', action='store_true',
                       help='Extract text from detected regions using Tesseract')
    parser.add_argument('--min-width', type=int, default=MIN_TEXT_W,
                       help=f'Minimum region width in pixels (default: {MIN_TEXT_W})')
    parser.add_argument('--min-height', type=int, default=MIN_TEXT_H,
                       help=f'Minimum region height in pixels (default: {MIN_TEXT_H})')
    parser.add_argument('--images-dir', type=Path, default=None,
                       help='Directory containing images (default: img/)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: mag_output/)')
    
    args = parser.parse_args()
    
    # Set default directories
    root = Path(__file__).parent
    if args.images_dir is None:
        args.images_dir = root
    if args.output_dir is None:
        args.output_dir = root
    
    print("Magazine Layout Analysis with OpenCV")
    print("=" * 60)
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if images directory exists
    if not args.images_dir.exists():
        print(f"\nError: Images directory not found: {args.images_dir}")
        print("Please create it and add some images.")
        return 1
    
    # Load images
    images = load_images(args.images_dir)
    if not images:
        print(f"\nError: No images found in {args.images_dir}")
        print("Supported formats: .png, .jpg, .jpeg, .tif, .tiff")
        return 1
    
    print(f"Found {len(images)} image(s)")
    
    # Process each image
    for img in images:
        try:
            process_image(img, args.output_dir, args)
        except Exception as e:
            print(f"\nError processing {img.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  *_annotated.png  - Visual annotations")
    print("  *_regions.csv    - Structured data (spreadsheet-friendly)")
    print("  *_regions.json   - Structured data (code-friendly)")
    if args.debug:
        print(f"  debug/*          - Intermediate processing steps")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())