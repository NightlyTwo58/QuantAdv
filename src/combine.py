#!/usr/bin/env python3
"""Recover partial QuantAdv outputs and regenerate report artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import data as report_data


def parse_args() -> argparse.Namespace:
    """Parse recovery-report command-line options."""
    parser = argparse.ArgumentParser(
        description="Merge incomplete QuantAdv outputs and regenerate reports."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(report_data.DATA_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Do not print table sizes.")
    return parser.parse_args()


def main() -> None:
    """Run partial-output recovery using the shared report implementation."""
    args = parse_args()
    report_data.generate_reports(
        args.data_dir,
        plots=not args.no_plots,
        summary=not args.quiet,
    )


if __name__ == "__main__":
    main()
