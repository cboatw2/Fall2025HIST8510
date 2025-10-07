#!/usr/bin/env python3
"""
Reset Week7-ComputerVision outputs
==================================

Removes generated output directories/files to return the repo to a clean state.

What it removes (if present):
- mag_output/                (magazine_layout_demo.py outputs)
- output/                    (find_text_regions.py outputs)
- lp_output/                 (layoutparser_demo.py outputs)

By default it performs deletion. Use --dry-run to preview.

Usage:
  python3 reset_outputs.py           # delete outputs
  python3 reset_outputs.py --dry-run # show what would be deleted
"""

from pathlib import Path
import shutil
import argparse


def rm_dir(path: Path, dry: bool) -> None:
    if not path.exists():
        return
    if dry:
        count = sum(1 for _ in path.rglob('*'))
        print(f"  Would remove: {path.name}/ ({count} items)")
        return
    shutil.rmtree(path)
    print(f"  Removed: {path.name}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset Week7 ComputerVision outputs")
    parser.add_argument('--dry-run', action='store_true', help='Preview without deleting')
    args = parser.parse_args()

    root = Path(__file__).parent
    targets = [
        root / 'mag_output',
        root / 'output',
        root / 'lp_output',
    ]

    print("Resetting outputs in Week7-ComputerVision")
    print("-" * 50)
    if args.dry_run:
        print("DRY RUN - No files will be deleted")

    for t in targets:
        rm_dir(t, args.dry_run)

    print("\nDone.")
    if args.dry_run:
        print("Run without --dry-run to apply deletions.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


