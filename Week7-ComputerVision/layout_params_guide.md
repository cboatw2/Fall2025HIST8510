# Magazine Layout Detection: Parameter Reference

## Parameters You Can Adjust

All parameters are defined at the top of `magazine_layout_demo.py`. Change them to tune detection for your specific documents.

---

## Binarization

### `ADAPTIVE_BLOCK_SIZE = 25`
Window size for thresholding (must be odd number)
- **Increase** (31, 41): For even lighting, simple documents
- **Decrease** (15, 11): For uneven lighting, shadows, complex contrast

### `ADAPTIVE_C = 11`
Threshold adjustment constant
- **Increase** (15, 20): If text is faded and not being detected
- **Decrease** (7, 5): If too much background noise is detected

---

## Line Detection

### `VERT_KERNEL = (1, 60)` *(width, height)*
Detects vertical lines (column separators)
- **Increase height** (80, 100): To detect only major column dividers
- **Decrease height** (40, 30): To detect shorter lines and table borders

### `HORIZ_KERNEL = (60, 1)` *(width, height)*
Detects horizontal lines (section breaks)
- **Increase width** (80, 100): To detect only major dividers
- **Decrease width** (40, 30): To detect short rules and underlines

---

## Text Connection

### `TEXT_CLOSE_H = (17, 3)` *(width, height)*
Connects letters horizontally into words
- **Increase width** (25, 30): If letters aren't connecting (wide spacing)
- **Decrease width** (12, 10): If separate words are merging

### `TEXT_CLOSE_V = (3, 9)` *(width, height)*
Connects lines vertically into blocks
- **Increase height** (12, 15): If lines aren't connecting (wide line spacing)
- **Decrease height** (6, 5): If separate paragraphs are merging

---

## Filtering

### `MIN_TEXT_W = 30` *(pixels)*
Minimum region width
- **Increase** (50, 70): To filter more noise
- **Decrease** (20, 15): If missing small text regions

### `MIN_TEXT_H = 20` *(pixels)*
Minimum region height
- **Increase** (30, 40): To filter more noise
- **Decrease** (10, 15): If missing single-line text

### `MAX_ASPECT_RATIO = 14.0`
Maximum width:height ratio
- **Increase** (20, 25): If missing very wide or very narrow regions
- **Decrease** (10, 8): If detecting lines as regions

---

## Classification Thresholds

### `IMAGE_FILL_THRESHOLD = 0.15`
Regions with < 15% filled pixels classified as images
- **Increase** (0.20): If text being misclassified as images
- **Decrease** (0.10): If images being misclassified as text

### `HEADLINE_Y_THRESHOLD = 0.15`
Top 15% of page considered headline zone
- Adjust based on where headlines appear in your documents

### `AD_Y_THRESHOLD = 0.5`
Bottom 50% of page considered ad zone
- Adjust based on where ads appear in your documents

### `COLUMN_ASPECT = 0.5` & `COLUMN_HEIGHT_THRESHOLD = 0.3`
Regions tall/narrow + >30% of page height = columns
- Adjust if columns being misclassified

---

## Command-Line Options

```bash
# Basic usage
python magazine_layout_demo.py

# See intermediate processing steps
python magazine_layout_demo.py --debug

# Run OCR on detected regions
python magazine_layout_demo.py --ocr

# Override minimum size filters
python magazine_layout_demo.py --min-width 50 --min-height 30
```

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Columns merging | Decrease `TEXT_CLOSE_H` |
| Each line separate | Increase `TEXT_CLOSE_V` |
| Missing column dividers | Decrease `VERT_KERNEL` height |
| Too many tiny regions | Increase `MIN_TEXT_W` and `MIN_TEXT_H` |
| Everything labeled "text_block" | Adjust classification thresholds |

---

## Workflow

1. Run with defaults and `--debug` flag
2. Examine debug images to identify issues
3. Adjust ONE parameter at a time
4. Re-run and observe changes
5. Iterate until results are satisfactory

Different document types need different parameters. Experimentation is part of the process.
