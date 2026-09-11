#!/usr/bin/env python3
"""
s1_cellbin2.py - CellBin2 ssDNA-based cell segmentation (standardized)

Generate CellBin2 pipeline JSON configuration and optionally execute the
pipeline for ssDNA-based cell segmentation.

Supports two modes:
  - registered   (default): ssDNA image is pre-registered (_regist.tif),
                  qc=false (trackline not needed), alignment=true
  - unregistered (--unregistered): ssDNA image is raw,
                  qc=true (enables trackline detection), alignment=true

Usage:
  # Registered ssDNA image (default):
  python s1_cellbin2.py -c Y40266LD \\
    -s /path/to/Y40266LD_ssDNA_regist.tif \\
    -g /path/to/Y40266LD.gem.gz \\
    -o /data/work/4th/Y40266LD

  # Unregistered ssDNA image:
  python s1_cellbin2.py -c Y40266LD \\
    -s /path/to/Y40266LD_ssDNA.tif \\
    -g /path/to/Y40266LD.gem.gz \\
    -o /data/work/4th/Y40266LD \\
    --unregistered
"""

import json
import argparse
import os
import subprocess
import sys


def generate_config(chip_id, ssdna_path, gem_path,
                    is_registered=True, magnification=10, correct_r=0):
    """
    Generate CellBin2 pipeline JSON configuration.

    Replaces hand-written JSON files.

    Parameters
    ----------
    chip_id : str
        Chip identifier (e.g. Y40266LD).
    ssdna_path : str
        Absolute path to ssDNA staining image (.tif).
    gem_path : str
        Absolute path to GEM expression matrix (.gem.gz).
    is_registered : bool
        True  → ssDNA image is already registered (_regist.tif);
                qc=false (skip trackline), alignment=true.
                Registration parameters kept identical to original JSON.
        False → ssDNA image is raw; qc=true (enable trackline detection),
                alignment=true. Pipeline performs trackline-based registration
                against the GEM coordinate space.
    magnification : int
        Imaging magnification (default: 10).
    correct_r : int
        Cell/nuclei mask expansion radius in pixels. 0 means no expansion.

    Returns
    -------
    dict : JSON-serialisable configuration for cellbin_pipeline.py
    """
    config = {
        "image_process": {
            "0": {
                "file_path": ssdna_path,
                "tech_type": "ssDNA",
                "chip_detect": False,
                "quality_control": False,
                "tissue_segmentation": False,
                "cell_segmentation": True,
                "channel_align": 0,
                "magnification": magnification,
                "registration": {
                    # Registration parameters kept consistent with original
                    # JSON for BOTH modes. The pipeline uses these settings
                    # to perform/verify alignment regardless of whether the
                    # input image is pre-registered or raw.
                    "fixed_image": 1,
                    "trackline": True,
                    "reuse": -1
                }
            },
            "1": {
                "file_path": gem_path,
                "tech_type": "Transcriptomics",
                "chip_detect": False,
                "quality_control": False,
                "tissue_segmentation": False,
                "cell_segmentation": False,
                "channel_align": -1,
                "magnification": magnification,
                "registration": {
                    "fixed_image": -1,
                    "trackline": False,
                    "reuse": -1
                }
            }
        },
        "molecular_classify": {
            "0": {
                "exp_matrix": 1,
                "cell_mask": {"nuclei": [0], "interior": [], "boundary": []},
                "correct_r": correct_r,
                "extra_method": ""
            }
        },
        "run": {
            # [CHANGED] QC stage is required for unregistered images because
            # trackline detection (registration.trackline=true) executes in QC.
            # For registered images QC is skipped (consistent with original JSON).
            "qc": (not is_registered),
            "alignment": True,
            "matrix_extract": True,
            "report": False,
            "annotation": False
        }
    }
    return config


def validate_inputs(ssdna_path, gem_path):
    """Validate that input files exist before generating config."""
    missing = []
    if not os.path.isfile(ssdna_path):
        missing.append(f"ssDNA image not found: {ssdna_path}")
    if not os.path.isfile(gem_path):
        missing.append(f"GEM file not found: {gem_path}")
    if missing:
        for m in missing:
            print(f"[ERROR] {m}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CellBin2 ssDNA cell segmentation — config generator & runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python s1_cellbin2.py -c Y40266LD -s ssDNA_regist.tif -g chip.gem.gz -o out/\n"
               "  python s1_cellbin2.py -c Y40266LD -s ssDNA.tif -g chip.gem.gz -o out/ --unregistered\n"
    )
    parser.add_argument("-c", "--chip-id", required=True,
                        help="Chip ID (e.g. Y40266LD)")
    parser.add_argument("-s", "--ssdna", required=True,
                        help="Path to ssDNA staining image (.tif)")
    parser.add_argument("-g", "--gem", required=True,
                        help="Path to GEM expression matrix (.gem.gz)")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for pipeline results")
    # [NEW] Registration mode flag
    parser.add_argument("--unregistered", action="store_true",
                        help="ssDNA image is NOT pre-registered; "
                             "enable pipeline registration (trackline + alignment)")
    parser.add_argument("--magnification", type=int, default=10,
                        help="Imaging magnification (default: 10)")
    parser.add_argument("--correct-r", type=int, default=0,
                        help="Cell/nuclei mask expansion radius in pixels (default: 0, no expansion)")
    # [NEW] Pipeline execution options
    parser.add_argument("--pipeline",
                        default="/opt/software/cellbin2/cellbin2/cellbin_pipeline.py",
                        help="Path to cellbin_pipeline.py")
    parser.add_argument("--python", default=None,
                        help="Python interpreter (default: current interpreter)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate config JSON only, don't run pipeline")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip input file existence check")

    args = parser.parse_args()

    # --- Validate ---
    if not args.skip_validation:
        validate_inputs(args.ssdna, args.gem)

    # --- Generate config ---
    is_registered = not args.unregistered
    config = generate_config(
        args.chip_id, args.ssdna, args.gem,
        is_registered=is_registered,
        magnification=args.magnification,
        correct_r=args.correct_r
    )

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, f"{args.chip_id}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    mode_str = ("registered (pre-aligned ssDNA)" if is_registered
                else "unregistered (raw ssDNA, pipeline will register)")
    print(f"[s1] Config saved: {config_path}")
    print(f"     Mode: {mode_str}")
    print(f"     correct_r: {args.correct_r} pixel(s)")

    if args.dry_run:
        print("[s1] Dry run — pipeline not executed.")
        return

    # --- Run pipeline ---
    python_bin = args.python or sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        python_bin, args.pipeline,
        "-c", args.chip_id,
        "-p", config_path,
        "-o", args.output_dir,
    ]
    print(f"[s1] Running: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
