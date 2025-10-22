Civil War Pamphlet CV utilities
=================================

This folder contains a lightweight script `cv_pamphlet_segment.py` that scans the
images inside `CivilWarPamphletTrial/`, classifies pages as `drawing` or `text`, and
attempts to segment text pages into sections using whitespace and divider-line heuristics.

Quick run
---------

Activate your virtualenv (if you have one) and run:

```sh
python Week7-ComputerVision/cv_pamphlet_segment.py \
  --input-folder Week7-ComputerVision/CivilWarPamphletTrial \
  --output-folder Week7-ComputerVision/outputs
```

Outputs
-------
- `outputs/annotated_<image>.jpg` — visualization with labeled sections and page type
- `outputs/<image>.json` — metadata: page_type, sections
- `outputs/summary.json` — per-image summary (edge density, stroke width stats, number of sections)

Tuning tips
-----------
- ADAPTIVE_BLOCK_SIZE: larger odd numbers (e.g. 31, 51) reduce sensitivity to small text blobs.
- TEXT_CLOSE_V: larger values (30-80) will merge text lines that are farther apart vertically.
- TEXT_EDGE_DENSITY_THRESHOLD: lower to be stricter about calling something 'drawing' (default tuned for these scans).
- MIN_SECTION_HEIGHT: prevents tiny separators from producing spurious sections.

Next improvements
-----------------
- Use a small CNN or classical features + an SVM to better separate drawings vs text.
- Run OCR (Tesseract) on each section to extract text and use font/size clues.
- Provide an interactive correction UI (streamlit or simple Flask) to accept/reject sections.

If you'd like, I can add Tesseract OCR integration or tune parameters for a few example pages. 
