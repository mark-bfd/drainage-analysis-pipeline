#!/usr/bin/env python3
"""
02_validate_pix4d.py - Pix4D Export Validation Script
Drone Drainage Analysis Pipeline

Validates Pix4D exports before running hydrological analysis.
Checks for required files, consistent CRS, resolution specs, and data quality.

Usage:
    python 02_validate_pix4d.py --input "C:/path/to/pix4d/exports"
    python 02_validate_pix4d.py --input "C:/path/to/pix4d/exports" --output validation_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Use GDAL from QGIS installation
try:
    from osgeo import gdal, osr
    import numpy as np
except ImportError:
    print("ERROR: GDAL/numpy not found. Run with QGIS Python:")
    print('  "C:\\Program Files\\QGIS 3.40.15\\apps\\Python312\\python.exe" 02_validate_pix4d.py')
    sys.exit(1)

# Enable GDAL exceptions
gdal.UseExceptions()


# =============================================================================
# Configuration
# =============================================================================

REQUIRED_FILES = {
    'dsm': {'patterns': ['*dsm*.tif', '*DSM*.tif', '*_dsm.tif', '*dsm*.tiff', '*DSM*.tiff', '*-dsm.tiff'], 'required': True,
            'description': 'Digital Surface Model'},
    'dtm': {'patterns': ['*dtm*.tif', '*DTM*.tif', '*_dtm.tif', '*dtm*.tiff', '*DTM*.tiff', '*-dtm.tiff'], 'required': False,
            'description': 'Digital Terrain Model (can be generated from DSM if missing)'},
    'orthomosaic': {'patterns': ['*ortho*.tif', '*mosaic*.tif', '*_orthomosaic.tif', '*ortho*.tiff', '*mosaic*.tiff', '*-orthomosaic.tiff'], 'required': True,
                   'description': 'Georeferenced Orthomosaic'},
    'point_cloud': {'patterns': ['*.laz', '*.las', '*point_cloud*'], 'required': False,
                   'description': 'Dense Point Cloud'},
    'quality_report': {'patterns': ['*.pdf', '*report*.pdf', '*quality*.pdf'], 'required': False,
                      'description': 'Pix4D Quality Report'}
}

# Target specifications
TARGET_GSD_CM = 3.0  # Maximum acceptable GSD in centimeters
MIN_COVERAGE_PERCENT = 95.0  # Minimum valid data coverage
VALID_CRS_TYPES = ['projected', 'geographic']  # Acceptable CRS types


# =============================================================================
# File Discovery
# =============================================================================

def find_files_by_pattern(directory: Path, patterns: List[str]) -> List[Path]:
    """Find files matching any of the given glob patterns."""
    matches = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
        # Also search subdirectories one level deep
        matches.extend(directory.glob(f'*/{pattern}'))
    return list(set(matches))  # Remove duplicates


def discover_pix4d_exports(input_dir: Path) -> Dict[str, List[Path]]:
    """Discover all Pix4D export files in the input directory."""
    discovered = {}
    for file_type, config in REQUIRED_FILES.items():
        matches = find_files_by_pattern(input_dir, config['patterns'])
        discovered[file_type] = sorted(matches, key=lambda p: p.stat().st_size, reverse=True)
    return discovered


# =============================================================================
# Raster Validation
# =============================================================================

def get_raster_info(filepath: Path) -> Dict[str, Any]:
    """Extract comprehensive information from a raster file."""
    ds = gdal.Open(str(filepath))
    if ds is None:
        raise ValueError(f"Cannot open raster: {filepath}")

    # Basic properties
    width = ds.RasterXSize
    height = ds.RasterYSize
    band_count = ds.RasterCount

    # Geotransform: (x_origin, pixel_width, rotation_x, y_origin, rotation_y, pixel_height)
    gt = ds.GetGeoTransform()
    pixel_width = abs(gt[1])
    pixel_height = abs(gt[5])

    # Bounds
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + (width * gt[1])
    y_min = y_max + (height * gt[5])

    # CRS info
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    crs_info = {
        'wkt': ds.GetProjection(),
        'proj4': srs.ExportToProj4() if srs.Validate() == 0 else None,
        'epsg': None,
        'is_projected': srs.IsProjected(),
        'is_geographic': srs.IsGeographic(),
        'linear_units': srs.GetLinearUnitsName() if srs.IsProjected() else 'degrees',
        'authority': None
    }

    # Try to get EPSG code
    if srs.GetAuthorityName(None) == 'EPSG':
        crs_info['epsg'] = int(srs.GetAuthorityCode(None))
        crs_info['authority'] = f"EPSG:{crs_info['epsg']}"

    # Calculate GSD in the raster's native units
    gsd_native = (pixel_width + pixel_height) / 2

    # Convert to centimeters if projected (assumes meters)
    if srs.IsProjected():
        gsd_cm = gsd_native * 100  # meters to cm
    else:
        # Geographic CRS - approximate conversion at equator
        # 1 degree ≈ 111,320 meters
        gsd_m = gsd_native * 111320
        gsd_cm = gsd_m * 100

    # NoData analysis (sample first band)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    # Calculate valid data coverage (sample-based for large rasters)
    stats = band.GetStatistics(True, True)  # approx, force
    data_min, data_max, data_mean, data_std = stats

    # For coverage calculation, read a sample
    sample_size = min(1000, width, height)
    step_x = max(1, width // sample_size)
    step_y = max(1, height // sample_size)

    sample_data = band.ReadAsArray(0, 0, width, height,
                                    buf_xsize=sample_size,
                                    buf_ysize=sample_size)

    if nodata is not None:
        valid_mask = sample_data != nodata
    else:
        valid_mask = ~np.isnan(sample_data) & np.isfinite(sample_data)

    valid_coverage = (np.sum(valid_mask) / valid_mask.size) * 100

    # File info
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    ds = None  # Close dataset

    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'file_size_mb': round(file_size_mb, 2),
        'width_px': width,
        'height_px': height,
        'band_count': band_count,
        'pixel_width': pixel_width,
        'pixel_height': pixel_height,
        'gsd_cm': round(gsd_cm, 4),
        'bounds': {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        },
        'crs': crs_info,
        'nodata_value': nodata,
        'valid_coverage_percent': round(valid_coverage, 2),
        'statistics': {
            'min': data_min,
            'max': data_max,
            'mean': data_mean,
            'std': data_std
        }
    }


def check_crs_consistency(rasters: Dict[str, Dict]) -> Tuple[bool, List[str]]:
    """Check that all rasters have consistent CRS."""
    issues = []
    epsg_codes = set()

    for name, info in rasters.items():
        if info and info.get('crs', {}).get('epsg'):
            epsg_codes.add(info['crs']['epsg'])

    if len(epsg_codes) > 1:
        issues.append(f"Inconsistent CRS: found EPSG codes {epsg_codes}")
        return False, issues
    elif len(epsg_codes) == 0:
        issues.append("Warning: Could not determine EPSG codes for CRS comparison")

    return len(issues) == 0, issues


def check_resolution_spec(raster_info: Dict, target_gsd_cm: float) -> Tuple[bool, List[str]]:
    """Check if raster meets GSD specification."""
    issues = []
    gsd = raster_info.get('gsd_cm', float('inf'))

    if gsd > target_gsd_cm:
        issues.append(f"GSD {gsd:.2f} cm exceeds target {target_gsd_cm:.2f} cm")
        return False, issues

    return True, issues


def check_data_coverage(raster_info: Dict, min_coverage: float) -> Tuple[bool, List[str]]:
    """Check if raster has sufficient valid data coverage."""
    issues = []
    coverage = raster_info.get('valid_coverage_percent', 0)

    if coverage < min_coverage:
        issues.append(f"Valid data coverage {coverage:.1f}% below minimum {min_coverage:.1f}%")
        return False, issues

    return True, issues


def check_elevation_data(dsm_info: Dict, dtm_info: Optional[Dict]) -> Tuple[bool, List[str]]:
    """Validate elevation data makes sense."""
    issues = []
    warnings = []

    # Check DSM elevation range
    dsm_stats = dsm_info.get('statistics', {})
    dsm_min = dsm_stats.get('min')
    dsm_max = dsm_stats.get('max')

    if dsm_min is not None and dsm_max is not None:
        elev_range = dsm_max - dsm_min

        # Sanity checks
        if dsm_min < -500:
            issues.append(f"DSM minimum elevation {dsm_min:.1f}m seems unrealistic (< -500m)")
        if dsm_max > 9000:
            issues.append(f"DSM maximum elevation {dsm_max:.1f}m seems unrealistic (> 9000m)")
        if elev_range > 1000:
            warnings.append(f"DSM elevation range {elev_range:.1f}m is very large - verify data")
        if elev_range < 0.1:
            warnings.append(f"DSM elevation range {elev_range:.2f}m is very small - may be flat terrain or data issue")

    # Compare DSM and DTM if both exist
    if dtm_info:
        dtm_stats = dtm_info.get('statistics', {})
        dtm_min = dtm_stats.get('min')
        dtm_max = dtm_stats.get('max')

        if dtm_max is not None and dsm_max is not None:
            if dtm_max > dsm_max:
                issues.append(f"DTM max ({dtm_max:.1f}m) > DSM max ({dsm_max:.1f}m) - DTM should be <= DSM")

    for w in warnings:
        issues.append(f"WARNING: {w}")

    return len([i for i in issues if not i.startswith('WARNING')]) == 0, issues


# =============================================================================
# Main Validation
# =============================================================================

def validate_pix4d_exports(input_dir: Path, verbose: bool = True,
                           target_gsd_cm: float = TARGET_GSD_CM,
                           min_coverage_pct: float = MIN_COVERAGE_PERCENT) -> Dict[str, Any]:
    """Run complete validation on Pix4D exports."""

    results = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': str(input_dir),
        'status': 'UNKNOWN',
        'summary': {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0
        },
        'discovered_files': {},
        'raster_info': {},
        'checks': [],
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    if not input_dir.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Input directory does not exist: {input_dir}")
        return results

    # Step 1: Discover files
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pix4D Export Validation")
        print(f"Input: {input_dir}")
        print(f"{'='*60}\n")
        print("Discovering files...")

    discovered = discover_pix4d_exports(input_dir)
    results['discovered_files'] = {k: [str(p) for p in v] for k, v in discovered.items()}

    # Step 2: Check required files exist
    for file_type, config in REQUIRED_FILES.items():
        check_result = {
            'name': f'{file_type}_exists',
            'description': f"Check {config['description']} exists",
            'passed': False,
            'details': None
        }

        if discovered[file_type]:
            check_result['passed'] = True
            check_result['details'] = f"Found: {discovered[file_type][0].name}"
            if len(discovered[file_type]) > 1:
                check_result['details'] += f" (+{len(discovered[file_type])-1} more)"
        else:
            if config['required']:
                results['issues'].append(f"Missing required file: {config['description']}")
            else:
                results['warnings'].append(f"Optional file not found: {config['description']}")
                check_result['passed'] = True  # Optional, so not a failure
            check_result['details'] = "Not found"

        results['checks'].append(check_result)
        results['summary']['total_checks'] += 1
        if check_result['passed']:
            results['summary']['passed'] += 1
        else:
            results['summary']['failed'] += 1

    if verbose:
        print("\nFile discovery:")
        for file_type, files in discovered.items():
            status = "[OK]" if files else "[MISSING]" if REQUIRED_FILES[file_type]['required'] else "[OPTIONAL]"
            print(f"  {status} {file_type}: {files[0].name if files else 'NOT FOUND'}")

    # Step 3: Analyze rasters
    raster_types = ['dsm', 'dtm', 'orthomosaic']
    for rtype in raster_types:
        if discovered[rtype]:
            try:
                info = get_raster_info(discovered[rtype][0])
                results['raster_info'][rtype] = info

                if verbose:
                    print(f"\n{rtype.upper()} Info:")
                    print(f"  Size: {info['width_px']} x {info['height_px']} pixels")
                    print(f"  GSD: {info['gsd_cm']:.2f} cm")
                    print(f"  CRS: {info['crs'].get('authority', 'Unknown')}")
                    print(f"  Coverage: {info['valid_coverage_percent']:.1f}%")
                    print(f"  File size: {info['file_size_mb']:.1f} MB")

            except Exception as e:
                results['issues'].append(f"Error reading {rtype}: {str(e)}")

    # Step 4: Run quality checks
    if verbose:
        print("\nRunning quality checks...")

    # Check CRS consistency
    if len(results['raster_info']) > 1:
        passed, issues = check_crs_consistency(results['raster_info'])
        check = {'name': 'crs_consistency', 'description': 'CRS consistency across rasters',
                 'passed': passed, 'details': issues[0] if issues else 'All CRS match'}
        results['checks'].append(check)
        results['summary']['total_checks'] += 1
        results['summary']['passed' if passed else 'failed'] += 1
        if issues:
            results['issues'].extend(issues)

    # Check GSD for DSM
    if 'dsm' in results['raster_info']:
        passed, issues = check_resolution_spec(results['raster_info']['dsm'], target_gsd_cm)
        check = {'name': 'dsm_gsd', 'description': f'DSM GSD <= {target_gsd_cm} cm',
                 'passed': passed, 'details': f"GSD: {results['raster_info']['dsm']['gsd_cm']:.2f} cm"}
        results['checks'].append(check)
        results['summary']['total_checks'] += 1
        results['summary']['passed' if passed else 'failed'] += 1
        if issues:
            results['issues'].extend(issues)

    # Check coverage
    for rtype in ['dsm', 'orthomosaic']:
        if rtype in results['raster_info']:
            passed, issues = check_data_coverage(results['raster_info'][rtype], min_coverage_pct)
            check = {'name': f'{rtype}_coverage', 'description': f'{rtype.upper()} data coverage >= {min_coverage_pct}%',
                     'passed': passed, 'details': f"Coverage: {results['raster_info'][rtype]['valid_coverage_percent']:.1f}%"}
            results['checks'].append(check)
            results['summary']['total_checks'] += 1
            results['summary']['passed' if passed else 'failed'] += 1
            if issues:
                results['issues'].extend(issues)

    # Check elevation data
    if 'dsm' in results['raster_info']:
        dtm_info = results['raster_info'].get('dtm')
        passed, issues = check_elevation_data(results['raster_info']['dsm'], dtm_info)
        check = {'name': 'elevation_validity', 'description': 'Elevation data sanity check',
                 'passed': passed, 'details': 'Elevation values within expected range'}
        results['checks'].append(check)
        results['summary']['total_checks'] += 1
        results['summary']['passed' if passed else 'failed'] += 1
        for issue in issues:
            if issue.startswith('WARNING'):
                results['warnings'].append(issue.replace('WARNING: ', ''))
            else:
                results['issues'].extend(issues)

    # Generate recommendations
    if 'dtm' not in results['raster_info'] and 'dsm' in results['raster_info']:
        results['recommendations'].append(
            "DTM not found. Will need to generate DTM from DSM using ground classification. "
            "Consider using Bulldozer (pip install bulldozer-dtm) or GRASS r.fillnulls."
        )

    if discovered['point_cloud']:
        results['recommendations'].append(
            f"Point cloud available ({discovered['point_cloud'][0].name}). "
            "Can use for higher-quality DTM extraction via ground classification."
        )

    # Final status
    results['summary']['warnings'] = len(results['warnings'])

    if results['summary']['failed'] == 0:
        results['status'] = 'PASSED'
    else:
        results['status'] = 'FAILED'

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION {results['status']}")
        print(f"{'='*60}")
        print(f"Checks: {results['summary']['passed']}/{results['summary']['total_checks']} passed")

        if results['issues']:
            print(f"\nIssues ({len(results['issues'])}):")
            for issue in results['issues']:
                print(f"  [X] {issue}")

        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  [!] {warning}")

        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  -> {rec}")

        print()

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate Pix4D exports for drainage analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python 02_validate_pix4d.py --input "C:\Pix4D\SampleSite\exports"
  python 02_validate_pix4d.py --input ./exports --output validation.json --quiet
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to Pix4D export directory')
    parser.add_argument('--output', '-o',
                        help='Output JSON file for validation results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress console output')
    parser.add_argument('--target-gsd', type=float, default=TARGET_GSD_CM,
                        help=f'Target GSD in cm (default: {TARGET_GSD_CM})')
    parser.add_argument('--min-coverage', type=float, default=MIN_COVERAGE_PERCENT,
                        help=f'Minimum coverage %% (default: {MIN_COVERAGE_PERCENT})')

    args = parser.parse_args()

    input_dir = Path(args.input)

    # Run validation with provided parameters
    results = validate_pix4d_exports(
        input_dir,
        verbose=not args.quiet,
        target_gsd_cm=args.target_gsd,
        min_coverage_pct=args.min_coverage
    )

    # Write output JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if not args.quiet:
            print(f"Results written to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'PASSED' else 1)


if __name__ == '__main__':
    main()
