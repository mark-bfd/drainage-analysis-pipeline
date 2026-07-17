#!/usr/bin/env python3
"""
05_package_delivery.py - Delivery Packaging & Archival
Drone Drainage Analysis Pipeline

Packages analysis outputs for client delivery and archives to NAS.

Usage:
    python 05_package_delivery.py --project ./output --community "Sample Lakeside"
    python 05_package_delivery.py --project ./output --config config/sample_site.json --archive

"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess


# =============================================================================
# Configuration
# =============================================================================

# Files to include in client delivery
DELIVERY_FILES = {
    'report': {
        'patterns': ['report*.pdf', 'report*.html'],
        'dest_folder': 'report',
        'required': True
    },
    'gis_rasters': {
        'patterns': ['dem_filled.tif', 'flow_accumulation.tif', 'depression_depth.tif',
                     'slope_percent.tif', 'streams_*.tif'],
        'dest_folder': 'gis',
        'required': False
    },
    'gis_vectors': {
        'patterns': ['*.gpkg', '*.geojson', 'watersheds.*', 'problem_areas.*'],
        'dest_folder': 'gis',
        'required': False
    },
    'data': {
        'patterns': ['analysis_results.json', '*.csv'],
        'dest_folder': 'data',
        'required': False
    },
    'orthomosaic': {
        'patterns': ['*ortho*.tif', '*orthomosaic*.tif'],
        'dest_folder': 'gis',
        'required': False
    }
}

# Default NAS archive path
DEFAULT_ARCHIVE_BASE = r"\\NAS\projects\drainage_assessments"  # network archive share (configurable)


# =============================================================================
# File Operations
# =============================================================================

def find_files(directory: Path, patterns: List[str]) -> List[Path]:
    """Find files matching any of the given glob patterns."""
    matches = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
        matches.extend(directory.glob(f'**/{pattern}'))
    return list(set(matches))


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def create_delivery_structure(project_dir: Path, output_dir: Path,
                              community_name: str) -> Dict[str, Any]:
    """Create organized delivery folder structure."""

    results = {
        'timestamp': datetime.now().isoformat(),
        'source': str(project_dir),
        'destination': str(output_dir),
        'community': community_name,
        'files': [],
        'total_size_mb': 0,
        'warnings': [],
        'errors': []
    }

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, config in DELIVERY_FILES.items():
        dest_folder = output_dir / config['dest_folder']
        dest_folder.mkdir(parents=True, exist_ok=True)

        files_found = find_files(project_dir, config['patterns'])

        if not files_found and config['required']:
            results['errors'].append(f"Required {category} files not found")
            continue

        for src_file in files_found:
            dest_file = dest_folder / src_file.name

            try:
                shutil.copy2(src_file, dest_file)
                size_mb = get_file_size_mb(src_file)
                results['files'].append({
                    'name': src_file.name,
                    'category': category,
                    'size_mb': round(size_mb, 2),
                    'destination': str(dest_file.relative_to(output_dir))
                })
                results['total_size_mb'] += size_mb
            except Exception as e:
                results['warnings'].append(f"Failed to copy {src_file.name}: {e}")

    results['total_size_mb'] = round(results['total_size_mb'], 2)
    return results


def create_readme(output_dir: Path, community_name: str, results: Dict):
    """Create README file for delivery package."""

    readme_content = f"""# Drainage Assessment Deliverables
## {community_name}
## Prepared by [Your Company]

Date: {datetime.now().strftime("%B %d, %Y")}

## Contents

### /report
Professional drainage assessment report in PDF and/or HTML format.
Start here for findings, problem areas, and recommendations.

### /gis
GIS data products for use in QGIS, ArcGIS, or other mapping software.

Files included:
- dem_filled.tif - Depression-filled terrain model
- flow_accumulation.tif - Flow accumulation raster
- depression_depth.tif - Depth of terrain depressions
- slope_percent.tif - Terrain slope in percent
- streams_*.tif - Extracted drainage networks at various thresholds

Coordinate System: See individual files (typically EPSG:32615 UTM Zone 15N or similar)

### /data
Raw analysis data in JSON and CSV formats for further analysis or integration.

- analysis_results.json - Complete analysis output including all statistics
- problem_areas.csv - Problem area inventory (if generated)
- watershed_statistics.csv - Watershed data (if generated)

## How to Use This Data

1. **Non-technical users**: Start with the PDF report in /report
2. **GIS users**: Load the .tif files in QGIS or ArcGIS
3. **Developers/analysts**: Use analysis_results.json for programmatic access

## Support

For questions about this assessment, contact:
- [Your Company]

## Disclaimer

This assessment is based on drone photogrammetry and automated analysis.
Field verification of findings is recommended before taking action.
This is not an engineering design document.

---
Package created: {datetime.now().isoformat()}
Total size: {results['total_size_mb']:.1f} MB
Files included: {len(results['files'])}
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    return readme_path


def create_zip_archive(source_dir: Path, zip_path: Path) -> float:
    """Create ZIP archive of delivery folder."""

    print(f"Creating ZIP archive: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    size_mb = get_file_size_mb(zip_path)
    print(f"Archive created: {size_mb:.1f} MB")
    return size_mb


def archive_to_nas(source_dir: Path, archive_base: str, community_name: str) -> bool:
    """Copy delivery package to NAS archive."""

    # Sanitize community name for folder
    safe_name = community_name.replace(' ', '_').replace('/', '-')
    date_str = datetime.now().strftime("%Y-%m-%d")

    archive_path = Path(archive_base) / safe_name / date_str

    print(f"\nArchiving to NAS: {archive_path}")

    try:
        # Check if NAS is accessible
        nas_base = Path(archive_base)
        if not nas_base.exists():
            print(f"Warning: NAS path not accessible: {archive_base}")
            print("Skipping NAS archive. Copy manually when NAS is available.")
            return False

        # Create archive directory
        archive_path.mkdir(parents=True, exist_ok=True)

        # Copy files
        for item in source_dir.iterdir():
            dest = archive_path / item.name
            if item.is_file():
                shutil.copy2(item, dest)
                print(f"  Copied: {item.name}")
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
                print(f"  Copied folder: {item.name}/")

        print(f"Archive complete: {archive_path}")
        return True

    except Exception as e:
        print(f"Archive failed: {e}")
        return False


def generate_manifest(results: Dict, output_dir: Path):
    """Generate manifest JSON file."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return manifest_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Package analysis outputs for delivery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python 05_package_delivery.py --project ./output --community "Sample Lakeside"
  python 05_package_delivery.py --project ./output --config config/sample_site.json --archive
  python 05_package_delivery.py --project ./output --community "Sample Lakeside" --zip
        """
    )

    parser.add_argument('--project', '-p', required=True,
                        help='Path to project output directory')
    parser.add_argument('--config', '-c',
                        help='Path to community config JSON')
    parser.add_argument('--community',
                        help='Community name (if not using config)')
    parser.add_argument('--output', '-o',
                        help='Output directory for delivery package')
    parser.add_argument('--zip', action='store_true',
                        help='Create ZIP archive of delivery package')
    parser.add_argument('--archive', action='store_true',
                        help='Archive to NAS (requires network access)')
    parser.add_argument('--archive-path',
                        help=f'NAS archive base path (default: {DEFAULT_ARCHIVE_BASE})')

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

    # Get community name
    community_name = args.community or config.get('community_name', 'Project')

    # Set paths
    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        sys.exit(1)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        safe_name = community_name.replace(' ', '_')
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = project_dir.parent / f"delivery_{safe_name}_{date_str}"

    print(f"\n{'='*60}")
    print(f"Packaging Delivery: {community_name}")
    print(f"{'='*60}\n")
    print(f"Source: {project_dir}")
    print(f"Output: {output_dir}\n")

    # Create delivery structure
    print("Organizing files...")
    results = create_delivery_structure(project_dir, output_dir, community_name)

    # Create README
    readme_path = create_readme(output_dir, community_name, results)
    print(f"Created: {readme_path.name}")

    # Generate manifest
    manifest_path = generate_manifest(results, output_dir)
    print(f"Created: {manifest_path.name}")

    # Summary
    print(f"\n{'='*60}")
    print("PACKAGING COMPLETE")
    print(f"{'='*60}")
    print(f"Files packaged: {len(results['files'])}")
    print(f"Total size: {results['total_size_mb']:.1f} MB")

    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for w in results['warnings']:
            print(f"  [!] {w}")

    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results['errors']:
            print(f"  [X] {e}")

    # Create ZIP if requested
    if args.zip:
        safe_name = community_name.replace(' ', '_')
        date_str = datetime.now().strftime("%Y%m%d")
        zip_path = output_dir.parent / f"Drainage_Assessment_{safe_name}_{date_str}.zip"
        create_zip_archive(output_dir, zip_path)
        print(f"ZIP archive: {zip_path}")

    # Archive to NAS if requested
    if args.archive:
        archive_base = args.archive_path or DEFAULT_ARCHIVE_BASE
        archive_to_nas(output_dir, archive_base, community_name)

    print(f"\nDelivery package ready: {output_dir}")


if __name__ == '__main__':
    main()
