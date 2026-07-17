#!/usr/bin/env python3
"""
run_pipeline.py - Master Pipeline Runner
Drone Drainage Analysis Pipeline

Runs the complete analysis pipeline from Pix4D exports to delivery package.

Usage:
    python run_pipeline.py --input "C:/Pix4D/exports" --config config/sample_site.json
    python run_pipeline.py --input "C:/Pix4D/exports" --community "Sample Lakeside" --output ./output

Stages:
    1. Validate Pix4D exports
    2. Run hydrological analysis
    3. Generate report
    4. Package for delivery
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess


# =============================================================================
# Configuration
# =============================================================================

# QGIS Python path on Windows
QGIS_PYTHON = r"C:\Program Files\QGIS 3.40.15\apps\Python312\python.exe"

# Script paths (relative to this script)
SCRIPTS_DIR = Path(__file__).parent
VALIDATE_SCRIPT = SCRIPTS_DIR / "02_validate_pix4d.py"
HYDRO_SCRIPT = SCRIPTS_DIR / "03_hydro_analysis.py"
REPORT_SCRIPT = SCRIPTS_DIR / "04_generate_report.py"
PACKAGE_SCRIPT = SCRIPTS_DIR / "05_package_delivery.py"


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_script(script_path: Path, args: list, description: str) -> bool:
    """Run a Python script with QGIS Python."""

    print(f"\n{'='*60}")
    print(f"STAGE: {description}")
    print(f"{'='*60}\n")

    # Build command
    python_exe = QGIS_PYTHON if Path(QGIS_PYTHON).exists() else sys.executable
    cmd = [python_exe, str(script_path)] + args

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def find_dem_file(pix4d_dir: Path) -> tuple:
    """Find DSM or DTM in Pix4D output directory."""

    # Prefer DTM over DSM
    dtm_patterns = ['*dtm*.tif', '*DTM*.tif', '*_dtm.tif']
    dsm_patterns = ['*dsm*.tif', '*DSM*.tif', '*_dsm.tif']

    for pattern in dtm_patterns:
        matches = list(pix4d_dir.glob(pattern)) + list(pix4d_dir.glob(f'**/{pattern}'))
        if matches:
            return matches[0], 'dtm'

    for pattern in dsm_patterns:
        matches = list(pix4d_dir.glob(pattern)) + list(pix4d_dir.glob(f'**/{pattern}'))
        if matches:
            return matches[0], 'dsm'

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Run complete drainage analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python run_pipeline.py --input "C:\Pix4D\SampleSite" --config config/sample_site.json
  python run_pipeline.py --input ./exports --community "Sample Lakeside" --output ./output
  python run_pipeline.py --input ./exports --community "Test" --skip-report
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to Pix4D export directory')
    parser.add_argument('--config', '-c',
                        help='Path to community config JSON')
    parser.add_argument('--community',
                        help='Community name (if not using config)')
    parser.add_argument('--output', '-o',
                        help='Output directory (default: <input>/analysis_output)')
    parser.add_argument('--api-key',
                        help='Anthropic API key for report generation')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip Pix4D validation stage')
    parser.add_argument('--skip-report', action='store_true',
                        help='Skip report generation stage')
    parser.add_argument('--skip-package', action='store_true',
                        help='Skip delivery packaging stage')
    parser.add_argument('--archive', action='store_true',
                        help='Archive to NAS after packaging')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "analysis_output"

    # Load config
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            config_arg = ['--config', str(config_path)]
        else:
            print(f"Warning: Config file not found: {config_path}")
            config_arg = []
    else:
        config_arg = []

    # Get community name
    community_name = args.community or config.get('community_name', 'Project')

    print(f"\n{'#'*60}")
    print(f"# DRONE DRAINAGE ANALYSIS PIPELINE")
    print(f"# Community: {community_name}")
    print(f"# Input: {input_dir}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")

    pipeline_start = datetime.now()
    stages_completed = []
    stages_failed = []

    # Stage 1: Validate Pix4D exports
    if not args.skip_validation:
        success = run_script(
            VALIDATE_SCRIPT,
            ['--input', str(input_dir)],
            "Validate Pix4D Exports"
        )
        if success:
            stages_completed.append('validation')
        else:
            stages_failed.append('validation')
            print("\nValidation failed. Fix issues before continuing.")
            sys.exit(1)
    else:
        print("\nSkipping validation...")

    # Find DEM file
    dem_path, dem_type = find_dem_file(input_dir)
    if not dem_path:
        print("\nERROR: No DSM or DTM found in input directory")
        sys.exit(1)

    print(f"\nUsing {dem_type.upper()}: {dem_path}")

    # Stage 2: Hydrological analysis
    hydro_args = [
        f'--{dem_type}', str(dem_path),
        '--output', str(output_dir)
    ] + config_arg

    success = run_script(
        HYDRO_SCRIPT,
        hydro_args,
        "Hydrological Analysis"
    )
    if success:
        stages_completed.append('hydro_analysis')
    else:
        stages_failed.append('hydro_analysis')
        print("\nHydrological analysis failed.")
        sys.exit(1)

    # Stage 3: Generate report
    if not args.skip_report:
        report_args = [
            '--data', str(output_dir),
            '--community', community_name,
            '--format', 'both'
        ]
        if args.api_key:
            report_args.extend(['--api-key', args.api_key])
        else:
            report_args.append('--skip-narrative')

        success = run_script(
            REPORT_SCRIPT,
            report_args,
            "Generate Report"
        )
        if success:
            stages_completed.append('report')
        else:
            stages_failed.append('report')
            print("\nReport generation failed (continuing...)")
    else:
        print("\nSkipping report generation...")

    # Stage 4: Package for delivery
    if not args.skip_package:
        package_args = [
            '--project', str(output_dir),
            '--community', community_name
        ]
        if args.archive:
            package_args.append('--archive')

        success = run_script(
            PACKAGE_SCRIPT,
            package_args,
            "Package Delivery"
        )
        if success:
            stages_completed.append('packaging')
        else:
            stages_failed.append('packaging')
            print("\nPackaging failed (continuing...)")
    else:
        print("\nSkipping delivery packaging...")

    # Summary
    pipeline_end = datetime.now()
    duration = pipeline_end - pipeline_start

    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE")
    print(f"{'#'*60}")
    print(f"\nCommunity: {community_name}")
    print(f"Duration: {duration}")
    print(f"\nStages completed: {', '.join(stages_completed)}")
    if stages_failed:
        print(f"Stages failed: {', '.join(stages_failed)}")
    print(f"\nOutputs: {output_dir}")

    sys.exit(0 if not stages_failed else 1)


if __name__ == '__main__':
    main()
