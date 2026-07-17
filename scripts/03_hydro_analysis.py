#!/usr/bin/env python3
"""
03_hydro_analysis.py - Hydrological Analysis Pipeline
Drone Drainage Analysis Pipeline

Complete hydrological analysis from DTM/DSM to problem area detection.
Uses GRASS GIS 8.4 for terrain analysis (via QGIS installation).

Usage:
    python 03_hydro_analysis.py --dsm dsm.tif --config config/sample_site.json --output ./output
    python 03_hydro_analysis.py --dtm dtm.tif --config config/sample_site.json --output ./output

Requires QGIS 3.40.15 environment with GRASS GIS.
Run via: run_with_qgis_python.bat 03_hydro_analysis.py [args]
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

# GDAL/numpy imports
try:
    from osgeo import gdal, ogr, osr
    import numpy as np
except ImportError:
    print("ERROR: GDAL/numpy not found. Run with QGIS Python:")
    print('  run_with_qgis_python.bat 03_hydro_analysis.py [args]')
    sys.exit(1)

gdal.UseExceptions()
ogr.UseExceptions()


# =============================================================================
# Configuration
# =============================================================================

# GRASS GIS paths (QGIS 3.40.15 on Windows)
QGIS_ROOT = Path(r"C:\Program Files\QGIS 3.40.15")
GISBASE = QGIS_ROOT / "apps" / "grass" / "grass84"
GRASS_BIN = QGIS_ROOT / "bin" / "grass84.bat"

# Analysis resolution — 1.7cm is overkill for hydrology.
# Professional hydro analysis uses 1-3m LiDAR. We use 25cm as a balance
# between detail and performance.
DEFAULT_ANALYSIS_RESOLUTION_M = 0.25

DEFAULT_CONFIG = {
    "hydrology": {
        "analysis_resolution_m": DEFAULT_ANALYSIS_RESOLUTION_M,
        "flow_accumulation_thresholds": [1000, 5000, 10000],
        "depression_min_depth_m": 0.15,
        "depression_max_depth_m": 3.0,  # Filter out lake/edge artifacts
        "depression_min_area_sqm": 46.5,
        "flat_area_slope_threshold": 0.005,
        "chokepoint_accumulation_ratio": 5.0,
        "lake_edge_buffer_m": 25.0,
        "storm_return_periods": [2, 10, 25, 100]
    }
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Depression:
    """Represents a terrain depression / potential ponding area."""
    id: int
    centroid_x: float
    centroid_y: float
    min_elevation: float
    max_elevation: float
    depth_m: float
    area_sqm: float
    volume_m3: float
    perimeter_m: float
    has_outflow: bool
    overflow_elevation: float


@dataclass
class ProblemArea:
    """Represents an identified drainage problem."""
    id: int
    type: str  # 'ponding', 'chokepoint', 'flat_area'
    severity: str  # 'high', 'medium', 'low'
    centroid_x: float
    centroid_y: float
    area_sqm: float
    description: str
    upstream_area_sqm: float
    flood_depth_2yr_m: Optional[float] = None
    flood_depth_10yr_m: Optional[float] = None
    flood_depth_25yr_m: Optional[float] = None
    flood_depth_100yr_m: Optional[float] = None


# =============================================================================
# GRASS GIS Session Management
# =============================================================================

class GrassSession:
    """Manages a temporary GRASS GIS session for hydrological analysis."""

    def __init__(self, epsg: int, verbose: bool = True):
        self.epsg = epsg
        self.verbose = verbose
        self.grassdb = None
        self.location = "hydro_analysis"
        self.mapset = "PERMANENT"
        self.env = None

    def setup(self, grassdb_path: Path):
        """Create GRASS database, location, and mapset."""
        self.grassdb = grassdb_path
        self.grassdb.mkdir(parents=True, exist_ok=True)

        # Build environment for GRASS
        self.env = os.environ.copy()
        self.env["GISBASE"] = str(GISBASE)
        self.env["GISRC"] = str(self.grassdb / "gisrc")
        self.env["GRASS_PYTHON"] = str(QGIS_ROOT / "apps" / "Python312" / "python.exe")
        self.env["GRASS_PROJSHARE"] = str(QGIS_ROOT / "share" / "proj")
        self.env["PROJ_LIB"] = str(QGIS_ROOT / "share" / "proj")
        self.env["GDAL_DATA"] = str(QGIS_ROOT / "apps" / "gdal" / "share" / "gdal")

        # Add GRASS and QGIS bins to PATH
        grass_bin = str(GISBASE / "bin")
        grass_lib = str(GISBASE / "lib")
        grass_scripts = str(GISBASE / "scripts")
        qgis_bin = str(QGIS_ROOT / "bin")
        gdal_bin = str(QGIS_ROOT / "apps" / "gdal" / "bin")
        self.env["PATH"] = f"{grass_bin};{grass_lib};{grass_scripts};{qgis_bin};{gdal_bin};{self.env['PATH']}"

        # Create GRASS location with EPSG code
        location_path = self.grassdb / self.location
        if not location_path.exists():
            self._run_grass_command(
                ["--text", "--exec", "g.version"],
                create_location=True
            )

        # Write GISRC
        gisrc_path = self.grassdb / "gisrc"
        gisrc_path.write_text(
            f"GISDBASE: {self.grassdb}\n"
            f"LOCATION_NAME: {self.location}\n"
            f"MAPSET: {self.mapset}\n"
        )

        if self.verbose:
            print(f"  GRASS session: {self.grassdb}")
            print(f"  Location CRS: EPSG:{self.epsg}")

    def _run_grass_command(self, args: list, create_location: bool = False,
                           capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a GRASS command via grass84.bat."""
        grass_bat = str(GRASS_BIN)
        location_path = str(self.grassdb / self.location / self.mapset)

        if create_location:
            cmd = [grass_bat, "-c", f"EPSG:{self.epsg}", str(self.grassdb / self.location),
                   "--text", "--exec", "g.version"]
        else:
            cmd = [grass_bat, location_path, "--text", "--exec"] + args

        if self.verbose and not capture_output:
            print(f"  > {' '.join(args[:3])}...")

        result = subprocess.run(
            cmd,
            env=self.env,
            capture_output=capture_output,
            text=True,
            timeout=3600  # 1 hour max per command
        )

        if result.returncode != 0 and capture_output:
            stderr = result.stderr.strip() if result.stderr else ""
            # Filter out common GRASS warnings that aren't errors
            if stderr and "WARNING" not in stderr and "HINT" not in stderr:
                raise RuntimeError(f"GRASS command failed: {' '.join(args[:3])}\n{stderr}")

        return result

    def run(self, module: str, **kwargs) -> str:
        """Run a GRASS module with keyword arguments. Returns stdout."""
        args = [module]
        for k, v in kwargs.items():
            if k == 'flags':
                # GRASS flags are passed as -g, -a, etc.
                args.append(f"-{v}")
            elif v is True:
                args.append(f"--{k}" if k in ('overwrite', 'quiet', 'verbose') else f"-{k}")
            elif v is not None and v is not False:
                # Replace double underscores with dots for parameter names
                key = k.replace("__", ".")
                args.append(f"{key}={v}")

        result = self._run_grass_command(args)
        return result.stdout.strip() if result.stdout else ""

    def cleanup(self):
        """Remove temporary GRASS database."""
        if self.grassdb and self.grassdb.exists():
            try:
                shutil.rmtree(self.grassdb)
            except Exception:
                pass  # Best effort cleanup


# =============================================================================
# Raster Utilities
# =============================================================================

def read_raster_meta(filepath: Path) -> dict:
    """Read raster metadata without loading data."""
    ds = gdal.Open(str(filepath))
    if ds is None:
        raise ValueError(f"Cannot open raster: {filepath}")

    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    band = ds.GetRasterBand(1)
    stats = band.GetStatistics(True, True)

    meta = {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'geotransform': gt,
        'projection': ds.GetProjection(),
        'pixel_width': abs(gt[1]),
        'pixel_height': abs(gt[5]),
        'nodata': band.GetNoDataValue(),
        'crs_epsg': None,
        'stats': {'min': stats[0], 'max': stats[1], 'mean': stats[2], 'std': stats[3]}
    }

    if srs.GetAuthorityName(None) == 'EPSG':
        meta['crs_epsg'] = int(srs.GetAuthorityCode(None))

    ds = None
    return meta


def read_raster(filepath: Path) -> Tuple[np.ndarray, dict]:
    """Read a raster file and return array + metadata."""
    ds = gdal.Open(str(filepath))
    if ds is None:
        raise ValueError(f"Cannot open raster: {filepath}")

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    meta = {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'geotransform': gt,
        'projection': ds.GetProjection(),
        'pixel_width': abs(gt[1]),
        'pixel_height': abs(gt[5]),
        'nodata': band.GetNoDataValue(),
        'crs_epsg': None
    }

    if srs.GetAuthorityName(None) == 'EPSG':
        meta['crs_epsg'] = int(srs.GetAuthorityCode(None))

    ds = None
    return data, meta


def write_raster(data: np.ndarray, meta: dict, filepath: Path,
                 nodata: float = -9999, dtype=gdal.GDT_Float32):
    """Write a numpy array to a GeoTIFF."""
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        str(filepath), meta['width'], meta['height'], 1, dtype,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    ds.SetGeoTransform(meta['geotransform'])
    ds.SetProjection(meta['projection'])
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    band.FlushCache()
    ds = None


def pixel_to_coord(row: int, col: int, gt: tuple) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates."""
    x = gt[0] + col * gt[1] + row * gt[2]
    y = gt[3] + col * gt[4] + row * gt[5]
    return x, y


# =============================================================================
# Downsampling
# =============================================================================

def downsample_dem(input_path: Path, output_path: Path,
                   target_res_m: float, verbose: bool = True) -> Path:
    """Downsample DEM to target resolution using GDAL (average resampling)."""
    if verbose:
        print(f"  Downsampling to {target_res_m}m resolution...")

    meta = read_raster_meta(input_path)
    current_res = meta['pixel_width']

    if current_res >= target_res_m * 0.9:
        if verbose:
            print(f"  Current resolution ({current_res:.4f}m) already at or above target. Skipping.")
        # Just copy
        shutil.copy2(str(input_path), str(output_path))
        return output_path

    # Use gdalwarp for high-quality resampling
    cmd = [
        str(QGIS_ROOT / "bin" / "gdalwarp.exe"),
        "-tr", str(target_res_m), str(target_res_m),
        "-r", "average",  # average resampling preserves elevation characteristics
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",
        str(input_path),
        str(output_path)
    ]

    env = os.environ.copy()
    env["PATH"] = f"{QGIS_ROOT / 'bin'};{QGIS_ROOT / 'apps' / 'gdal' / 'bin'};{env['PATH']}"
    env["GDAL_DATA"] = str(QGIS_ROOT / "apps" / "gdal" / "share" / "gdal")
    env["PROJ_LIB"] = str(QGIS_ROOT / "share" / "proj")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed: {result.stderr}")

    # Report new dimensions
    new_meta = read_raster_meta(output_path)
    if verbose:
        factor = current_res / target_res_m
        print(f"  Downsampled: {meta['width']}x{meta['height']} -> "
              f"{new_meta['width']}x{new_meta['height']} "
              f"({1/factor:.0f}x reduction)")

    return output_path


# =============================================================================
# GRASS-based Hydrological Analysis
# =============================================================================

def run_grass_hydro(grass: GrassSession, dem_path: Path, output_dir: Path,
                    config: dict, verbose: bool = True) -> Dict[str, Any]:
    """Run the complete GRASS-based hydro analysis pipeline."""

    hydro_config = config.get('hydrology', DEFAULT_CONFIG['hydrology'])
    outputs = {}
    statistics = {}
    t_start = time.time()

    # ---- Step 1: Import DEM into GRASS ----
    if verbose:
        print("\nStep 1: Importing DEM into GRASS GIS...")

    grass.run("r.in.gdal", input=str(dem_path), output="dem", overwrite=True)
    grass.run("g.region", raster="dem")

    # Get region info
    region_info = grass.run("g.region", flags="p")
    if verbose:
        for line in region_info.split('\n'):
            if any(k in line for k in ['rows:', 'cols:', 'nsres:', 'ewres:']):
                print(f"    {line.strip()}")

    # Get DEM stats
    dem_stats = grass.run("r.univar", map="dem", flags="g")
    stats_dict = {}
    for line in dem_stats.split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            try:
                stats_dict[k.strip()] = float(v.strip())
            except ValueError:
                stats_dict[k.strip()] = v.strip()

    statistics['dem'] = {
        'min_elevation': stats_dict.get('min', 0),
        'max_elevation': stats_dict.get('max', 0),
        'mean_elevation': stats_dict.get('mean', 0),
        'n_valid_cells': int(stats_dict.get('n', 0))
    }

    if verbose:
        print(f"  Elevation range: {statistics['dem']['min_elevation']:.2f} - "
              f"{statistics['dem']['max_elevation']:.2f} m")
        print(f"  Valid cells: {statistics['dem']['n_valid_cells']:,}")

    # ---- Step 2: Fill depressions ----
    if verbose:
        print("\nStep 2: Filling depressions (GRASS r.fill.dir)...")

    t2 = time.time()
    grass.run("r.fill.dir", input="dem", output="dem_filled",
              direction="flow_dir_grass", areas="depression_areas", overwrite=True)

    # Compute depression depth = filled - original
    grass.run("r.mapcalc",
              expression="depression_depth = if(dem_filled - dem > 0.001, dem_filled - dem, 0)",
              overwrite=True)

    # Export depression depth
    dep_depth_path = output_dir / "depression_depth.tif"
    grass.run("r.out.gdal", input="depression_depth", output=str(dep_depth_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              overwrite=True)
    outputs['depression_depth'] = str(dep_depth_path)

    # Export filled DEM
    filled_path = output_dir / "dem_filled.tif"
    grass.run("r.out.gdal", input="dem_filled", output=str(filled_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              overwrite=True)
    outputs['filled_dem'] = str(filled_path)

    if verbose:
        print(f"  Depression filling complete ({time.time()-t2:.0f}s)")

    # ---- Step 3: Compute slope ----
    if verbose:
        print("\nStep 3: Computing slope (GRASS r.slope.aspect)...")

    grass.run("r.slope.aspect", elevation="dem_filled", slope="slope_pct",
              format="percent", overwrite=True)

    slope_path = output_dir / "slope_percent.tif"
    grass.run("r.out.gdal", input="slope_pct", output=str(slope_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              overwrite=True)
    outputs['slope'] = str(slope_path)

    slope_stats = grass.run("r.univar", map="slope_pct", flags="g")
    slope_dict = {}
    for line in slope_stats.split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            try:
                slope_dict[k.strip()] = float(v.strip())
            except ValueError:
                pass

    statistics['slope'] = {
        'mean_pct': slope_dict.get('mean', 0),
        'max_pct': slope_dict.get('max', 0),
        'min_pct': slope_dict.get('min', 0)
    }

    # ---- Step 4: Watershed analysis (flow acc, basins, streams) ----
    if verbose:
        print("\nStep 4: Running watershed analysis (GRASS r.watershed)...")
        print("  This computes flow accumulation, basins, and stream networks in one pass...")

    t4 = time.time()

    # r.watershed computes everything at once — much faster than separate steps
    # threshold = minimum size of exterior watershed basin (in cells)
    # Use the medium threshold for basin delineation
    thresholds = hydro_config.get('flow_accumulation_thresholds', [1000, 5000, 10000])
    basin_threshold = thresholds[1] if len(thresholds) > 1 else 5000

    grass.run("r.watershed", elevation="dem_filled",
              accumulation="flow_acc", drainage="flow_dir",
              basin="basins", stream="streams_main",
              threshold=str(basin_threshold),
              flags="a",  # positive accumulation values
              overwrite=True)

    if verbose:
        print(f"  Watershed analysis complete ({time.time()-t4:.0f}s)")

    # Export flow accumulation
    flow_acc_path = output_dir / "flow_accumulation.tif"
    grass.run("r.out.gdal", input="flow_acc", output=str(flow_acc_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              type="Float64", overwrite=True)
    outputs['flow_accumulation'] = str(flow_acc_path)

    # Export flow direction
    flow_dir_path = output_dir / "flow_direction.tif"
    grass.run("r.out.gdal", input="flow_dir", output=str(flow_dir_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              overwrite=True)
    outputs['flow_direction'] = str(flow_dir_path)

    # Export basins
    basins_path = output_dir / "watersheds.tif"
    grass.run("r.out.gdal", input="basins", output=str(basins_path),
              format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
              overwrite=True)
    outputs['watersheds'] = str(basins_path)

    # Flow accumulation stats
    fa_stats = grass.run("r.univar", map="flow_acc", flags="g")
    fa_dict = {}
    for line in fa_stats.split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            try:
                fa_dict[k.strip()] = float(v.strip())
            except ValueError:
                pass

    statistics['flow_accumulation'] = {
        'max_cells': fa_dict.get('max', 0),
        'mean_cells': fa_dict.get('mean', 0)
    }

    # ---- Step 5: Extract stream networks at multiple thresholds ----
    if verbose:
        print("\nStep 5: Extracting stream networks at multiple thresholds...")

    for thresh in thresholds:
        stream_name = f"streams_{thresh}"
        grass.run("r.mapcalc",
                  expression=f"{stream_name} = if(flow_acc >= {thresh}, 1, null())",
                  overwrite=True)

        stream_path = output_dir / f"streams_thresh_{thresh}.tif"
        grass.run("r.out.gdal", input=stream_name, output=str(stream_path),
                  format="GTiff", createopt="COMPRESS=LZW,TILED=YES,BIGTIFF=YES",
                  nodata="-9999", overwrite=True)
        outputs[f'streams_{thresh}'] = str(stream_path)

        # Get stream length estimate
        try:
            stream_stats = grass.run("r.univar", map=stream_name, flags="g")
            for line in stream_stats.split('\n'):
                if line.startswith('n='):
                    n_cells = int(line.split('=')[1])
                    # Approximate length assuming cells are roughly linear
                    res_m = float(hydro_config.get('analysis_resolution_m', DEFAULT_ANALYSIS_RESOLUTION_M))
                    length_m = n_cells * res_m
                    if verbose:
                        print(f"  Threshold {thresh}: ~{length_m:.0f}m of stream channels ({n_cells:,} cells)")
                    break
        except Exception:
            pass

    # ---- Step 6: Vectorize stream network for GeoPackage output ----
    if verbose:
        print("\nStep 6: Vectorizing stream network...")

    try:
        # Thin the main streams to single-pixel width
        grass.run("r.thin", input="streams_main", output="streams_thin", overwrite=True)
        grass.run("r.to.vect", input="streams_thin", output="streams_vect",
                  type="line", overwrite=True)

        streams_gpkg = output_dir / "stream_network.gpkg"
        grass.run("v.out.ogr", input="streams_vect", output=str(streams_gpkg),
                  format="GPKG", overwrite=True)
        outputs['stream_network_gpkg'] = str(streams_gpkg)

        if verbose:
            print("  Stream network exported to GeoPackage")
    except Exception as e:
        if verbose:
            print(f"  Warning: Stream vectorization failed: {e}")

    # ---- Step 7: Vectorize watersheds ----
    if verbose:
        print("\nStep 7: Vectorizing watersheds...")

    try:
        grass.run("r.to.vect", input="basins", output="basins_vect",
                  type="area", overwrite=True)

        basins_gpkg = output_dir / "watersheds.gpkg"
        grass.run("v.out.ogr", input="basins_vect", output=str(basins_gpkg),
                  format="GPKG", overwrite=True)
        outputs['watersheds_gpkg'] = str(basins_gpkg)

        if verbose:
            print("  Watersheds exported to GeoPackage")
    except Exception as e:
        if verbose:
            print(f"  Warning: Watershed vectorization failed: {e}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Total GRASS processing time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return outputs, statistics


# =============================================================================
# Post-processing: Depression & Problem Analysis (NumPy on exported rasters)
# =============================================================================

def analyze_depressions(depression_depth_path: Path, dem_path: Path,
                        config: dict, verbose: bool = True) -> Tuple[List[Depression], dict]:
    """Analyze depressions from GRASS output rasters using NumPy."""
    if verbose:
        print("\nStep 8: Analyzing depressions...")

    hydro_config = config.get('hydrology', DEFAULT_CONFIG['hydrology'])
    min_depth = hydro_config.get('depression_min_depth_m', 0.15)
    max_depth = hydro_config.get('depression_max_depth_m', 3.0)
    min_area = hydro_config.get('depression_min_area_sqm', 46.5)

    from scipy import ndimage

    dep_data, dep_meta = read_raster(depression_depth_path)
    dem_data, _ = read_raster(dem_path)

    pixel_area = dep_meta['pixel_width'] * dep_meta['pixel_height']
    min_pixels = max(1, int(min_area / pixel_area))

    # Find significant depressions (filter out extreme depths = lake/edge artifacts)
    nodata = dep_meta['nodata']
    if nodata is not None:
        valid = (dep_data != nodata) & np.isfinite(dep_data)
    else:
        valid = np.isfinite(dep_data)

    depression_mask = (dep_data > min_depth) & (dep_data <= max_depth) & valid
    labeled, num_features = ndimage.label(depression_mask)

    depressions = []
    for i in range(1, num_features + 1):
        region = labeled == i
        pixel_count = np.sum(region)
        if pixel_count < min_pixels:
            continue

        depths = dep_data[region]
        elevations = dem_data[region]
        elevations = elevations[np.isfinite(elevations)]

        if len(elevations) == 0:
            continue

        area_sqm = pixel_count * pixel_area
        max_depth = float(np.max(depths))
        volume = float(np.sum(depths) * pixel_area)

        # Perimeter
        eroded = ndimage.binary_erosion(region)
        perimeter_pixels = np.sum(region & ~eroded)
        perimeter_m = perimeter_pixels * dep_meta['pixel_width']

        # Centroid
        rows, cols = np.where(region)
        cx, cy = pixel_to_coord(int(np.mean(rows)), int(np.mean(cols)),
                                dep_meta['geotransform'])

        depressions.append(Depression(
            id=len(depressions) + 1,
            centroid_x=cx, centroid_y=cy,
            min_elevation=float(np.min(elevations)),
            max_elevation=float(np.max(elevations)),
            depth_m=max_depth,
            area_sqm=area_sqm,
            volume_m3=volume,
            perimeter_m=perimeter_m,
            has_outflow=max_depth < 0.3,
            overflow_elevation=float(np.max(elevations))
        ))

    # Sort by volume (largest first)
    depressions.sort(key=lambda d: d.volume_m3, reverse=True)
    # Re-ID after sorting
    for i, d in enumerate(depressions):
        d.id = i + 1

    dep_stats = {
        'total_depressions': len(depressions),
        'total_volume_m3': sum(d.volume_m3 for d in depressions),
        'total_area_sqm': sum(d.area_sqm for d in depressions),
        'max_depth_m': max((d.depth_m for d in depressions), default=0),
        'largest_volume_m3': depressions[0].volume_m3 if depressions else 0
    }

    if verbose:
        print(f"  Found {len(depressions)} significant depressions")
        print(f"  Total ponding volume: {dep_stats['total_volume_m3']:.1f} m³")
        print(f"  Max depth: {dep_stats['max_depth_m']:.2f}m")

    return depressions, dep_stats


def analyze_flat_areas(slope_path: Path, config: dict,
                       verbose: bool = True) -> Tuple[List[Dict], dict]:
    """Identify flat areas from slope raster."""
    if verbose:
        print("\nStep 9: Analyzing flat areas...")

    hydro_config = config.get('hydrology', DEFAULT_CONFIG['hydrology'])
    flat_threshold = hydro_config.get('flat_area_slope_threshold', 0.005) * 100  # to percent
    min_area = hydro_config.get('depression_min_area_sqm', 46.5)

    from scipy import ndimage

    slope_data, slope_meta = read_raster(slope_path)
    pixel_area = slope_meta['pixel_width'] * slope_meta['pixel_height']
    min_pixels = max(1, int(min_area / pixel_area))

    nodata = slope_meta['nodata']
    if nodata is not None:
        valid = (slope_data != nodata) & np.isfinite(slope_data)
    else:
        valid = np.isfinite(slope_data)

    flat_mask = (slope_data < flat_threshold) & (slope_data >= 0) & valid
    labeled, num_features = ndimage.label(flat_mask)

    flat_areas = []
    for i in range(1, num_features + 1):
        region = labeled == i
        pixel_count = np.sum(region)
        if pixel_count < min_pixels:
            continue

        area_sqm = pixel_count * pixel_area
        rows, cols = np.where(region)
        cx, cy = pixel_to_coord(int(np.mean(rows)), int(np.mean(cols)),
                                slope_meta['geotransform'])

        flat_areas.append({
            'id': len(flat_areas) + 1,
            'centroid_x': cx, 'centroid_y': cy,
            'area_sqm': area_sqm,
            'avg_slope_pct': float(np.mean(slope_data[region]))
        })

    flat_areas.sort(key=lambda f: f['area_sqm'], reverse=True)
    for i, f in enumerate(flat_areas):
        f['id'] = i + 1

    flat_stats = {
        'total_flat_areas': len(flat_areas),
        'total_flat_area_sqm': sum(f['area_sqm'] for f in flat_areas)
    }

    if verbose:
        print(f"  Found {len(flat_areas)} significant flat areas")

    return flat_areas, flat_stats


def analyze_chokepoints(flow_acc_path: Path, dem_path: Path, config: dict,
                        verbose: bool = True) -> Tuple[List[Dict], dict]:
    """Identify chokepoints from flow accumulation raster.
    Filters out shoreline/reservoir-edge points using distance-from-nodata."""
    if verbose:
        print("\nStep 10: Analyzing chokepoints...")

    hydro_config = config.get('hydrology', DEFAULT_CONFIG['hydrology'])
    ratio_threshold = hydro_config.get('chokepoint_accumulation_ratio', 5.0)
    lake_buffer_m = hydro_config.get('lake_edge_buffer_m', 25.0)

    from scipy import ndimage

    fa_data, fa_meta = read_raster(flow_acc_path)

    pixel_size = fa_meta['pixel_width']

    # Compute distance from lake/nodata edge (in meters)
    # Use flow_acc's own valid mask — nodata = lake/off-survey
    land_mask = np.isfinite(fa_data) & (fa_data != 0)
    dist_from_lake = ndimage.distance_transform_edt(land_mask) * pixel_size

    if verbose:
        print(f"  Lake edge buffer: {lake_buffer_m}m "
              f"(max inland distance: {np.max(dist_from_lake):.0f}m)")

    # Find flow accumulation spikes
    with np.errstate(invalid='ignore'):
        local_min_upstream = ndimage.minimum_filter(fa_data, size=3)

        # Valid: significant flow, on land, away from lake edge
        valid = (fa_data > 500) & np.isfinite(fa_data) & (local_min_upstream > 10) \
                & (dist_from_lake > lake_buffer_m)
        ratio = np.where(valid, fa_data / local_min_upstream, 0)
        chokepoint_mask = ratio >= ratio_threshold

    # Get coordinates of chokepoints
    rows, cols = np.where(chokepoint_mask)

    chokepoints = []
    for r, c in zip(rows, cols):
        cx, cy = pixel_to_coord(int(r), int(c), fa_meta['geotransform'])
        chokepoints.append({
            'centroid_x': cx, 'centroid_y': cy,
            'flow_accumulation': float(fa_data[r, c]),
            'ratio': float(ratio[r, c]),
            'dist_from_lake_m': float(dist_from_lake[r, c])
        })

    # Sort by ratio and limit to top 20 most significant
    chokepoints.sort(key=lambda x: x['ratio'], reverse=True)
    chokepoints = chokepoints[:20]
    for i, cp in enumerate(chokepoints):
        cp['id'] = i + 1

    if verbose:
        # Count candidates before vs after lake filter for reporting
        with np.errstate(invalid='ignore'):
            ratio_no_filter = np.where(
                (fa_data > 500) & np.isfinite(fa_data) & (local_min_upstream > 10),
                fa_data / local_min_upstream, 0)
            total_before = int(np.sum(ratio_no_filter >= ratio_threshold))
            total_after = len(rows)
        print(f"  Found {len(chokepoints)} inland chokepoints "
              f"({total_before - total_after} filtered as too close to reservoir)")

    return chokepoints, {'total_chokepoints': len(chokepoints)}


# =============================================================================
# Problem Classification
# =============================================================================

def classify_problems(depressions: List[Depression], flat_areas: List[Dict],
                      chokepoints: List[Dict]) -> List[ProblemArea]:
    """Classify and rank all identified problem areas."""
    problems = []

    for dep in depressions:
        if dep.depth_m > 0.5 or dep.volume_m3 > 100:
            severity = 'high'
        elif dep.depth_m > 0.25 or dep.volume_m3 > 50:
            severity = 'medium'
        else:
            severity = 'low'

        problems.append(ProblemArea(
            id=0, type='ponding', severity=severity,
            centroid_x=dep.centroid_x, centroid_y=dep.centroid_y,
            area_sqm=dep.area_sqm,
            description=f"Depression {dep.depth_m:.2f}m deep, {dep.area_sqm:.0f} m², "
                        f"holds {dep.volume_m3:.1f} m³",
            upstream_area_sqm=0
        ))

    for flat in flat_areas:
        severity = 'medium' if flat['area_sqm'] > 1000 else 'low'
        problems.append(ProblemArea(
            id=0, type='flat_area', severity=severity,
            centroid_x=flat['centroid_x'], centroid_y=flat['centroid_y'],
            area_sqm=flat['area_sqm'],
            description=f"Flat area ({flat['avg_slope_pct']:.2f}% slope), "
                        f"{flat['area_sqm']:.0f} m²",
            upstream_area_sqm=0
        ))

    for cp in chokepoints:
        if cp['ratio'] > 25:
            severity = 'high'
        elif cp['ratio'] > 15:
            severity = 'medium'
        else:
            severity = 'low'

        problems.append(ProblemArea(
            id=0, type='chokepoint', severity=severity,
            centroid_x=cp['centroid_x'], centroid_y=cp['centroid_y'],
            area_sqm=100,
            description=f"Flow chokepoint - {cp['ratio']:.1f}x accumulation spike",
            upstream_area_sqm=cp.get('flow_accumulation', 0)
        ))

    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    problems.sort(key=lambda p: (severity_order[p.severity], -p.area_sqm))
    for i, p in enumerate(problems):
        p.id = i + 1

    return problems


# =============================================================================
# Main Pipeline
# =============================================================================

def fetch_parcel_mask(output_dir: Path, dem_meta: dict, config: dict,
                      verbose: bool = True) -> Optional[Path]:
    """Download parcel boundaries from TNRIS and rasterize as a land mask.
    Returns path to parcel_mask.tif or None if unavailable."""
    import subprocess as _sp

    mask_path = output_dir / "parcel_mask.tif"
    parcels_gpkg = output_dir / "parcels_utm.gpkg"

    # Check if we already have a mask
    if mask_path.exists():
        if verbose:
            print("  Using existing parcel mask")
        return mask_path

    epsg = dem_meta['crs_epsg']
    if epsg is None:
        return None

    # Compute bounding box in WGS84 for the TNRIS query
    # We need to convert UTM bounds to lat/lon
    from osgeo import osr as _osr
    srs_utm = _osr.SpatialReference()
    srs_utm.ImportFromEPSG(epsg)
    srs_wgs = _osr.SpatialReference()
    srs_wgs.ImportFromEPSG(4326)
    srs_wgs.SetAxisMappingStrategy(_osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs_utm.SetAxisMappingStrategy(_osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = _osr.CoordinateTransformation(srs_utm, srs_wgs)

    gt = dem_meta['geotransform']
    w = dem_meta['width']
    h = dem_meta['height']
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + w * gt[1]
    y_min = y_max + h * gt[5]

    # Add 500m buffer for parcels that extend beyond the DSM
    ll = transform.TransformPoint(x_min - 500, y_min - 500)
    ur = transform.TransformPoint(x_max + 500, y_max + 500)
    bbox = f"{ll[0]:.4f},{ll[1]:.4f},{ur[0]:.4f},{ur[1]:.4f}"

    if verbose:
        print(f"  Downloading parcels from TNRIS (bbox: {bbox})...")

    # Download from TNRIS StratMap 2022
    import urllib.request
    url = (
        "https://feature.tnris.org/arcgis/rest/services/Parcels/"
        "stratmap22_land_parcels_48/MapServer/0/query?"
        f"where=1%3D1&geometry={bbox}&geometryType=esriGeometryEnvelope"
        "&inSR=4326&spatialRel=esriSpatialRelIntersects"
        "&outFields=objectid&returnGeometry=true&f=geojson&resultRecordCount=2000"
    )

    geojson_path = output_dir / "_work" / "parcels_raw.geojson"
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(geojson_path))
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not download parcels: {e}")
        return None

    if not geojson_path.exists() or geojson_path.stat().st_size < 100:
        if verbose:
            print("  Warning: Empty parcel response")
        return None

    # Reproject to UTM with ogr2ogr
    ogr2ogr = str(QGIS_ROOT / "bin" / "ogr2ogr.exe")
    env = os.environ.copy()
    env["PATH"] = f"{QGIS_ROOT / 'bin'};{QGIS_ROOT / 'apps' / 'gdal' / 'bin'};{env['PATH']}"
    env["GDAL_DATA"] = str(QGIS_ROOT / "apps" / "gdal" / "share" / "gdal")
    env["PROJ_LIB"] = str(QGIS_ROOT / "share" / "proj")

    if parcels_gpkg.exists():
        parcels_gpkg.unlink()

    _sp.run([ogr2ogr, "-f", "GPKG", "-t_srs", f"EPSG:{epsg}", "-s_srs", "EPSG:4326",
             str(parcels_gpkg), str(geojson_path)],
            env=env, capture_output=True, timeout=60)

    if not parcels_gpkg.exists():
        if verbose:
            print("  Warning: Parcel reprojection failed")
        return None

    # Rasterize parcels onto the analysis grid
    # Use the filled DEM as the grid template
    filled_path = output_dir / "dem_filled.tif"
    if not filled_path.exists():
        return None

    ref_ds = gdal.Open(str(filled_path))
    ref_gt = ref_ds.GetGeoTransform()
    ref_w = ref_ds.RasterXSize
    ref_h = ref_ds.RasterYSize
    ref_proj = ref_ds.GetProjection()
    ref_ds = None

    parcel_ds = ogr.Open(str(parcels_gpkg))
    layer = parcel_ds.GetLayer(0)
    n_parcels = layer.GetFeatureCount()

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(mask_path), ref_w, ref_h, 1, gdal.GDT_Byte,
                           ['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(ref_gt)
    out_ds.SetProjection(ref_proj)
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)

    gdal.RasterizeLayer(out_ds, [1], layer, burn_values=[1])
    out_ds.FlushCache()

    mask = band.ReadAsArray()
    land_pct = np.sum(mask == 1) / mask.size * 100

    out_ds = None
    parcel_ds = None

    if verbose:
        print(f"  Downloaded {n_parcels} parcels, land coverage: {land_pct:.1f}%")

    return mask_path


def filter_problems_by_parcels(problems: list, parcel_mask_path: Path,
                               verbose: bool = True) -> list:
    """Remove problem areas that don't fall on parcel (property) land."""
    mask_ds = gdal.Open(str(parcel_mask_path))
    mask_band = mask_ds.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray()
    gt = mask_ds.GetGeoTransform()
    mask_ds = None

    filtered = []
    removed = 0
    for p in problems:
        col = int((p.centroid_x - gt[0]) / gt[1])
        row = int((p.centroid_y - gt[3]) / gt[5])
        if (0 <= row < mask_data.shape[0] and 0 <= col < mask_data.shape[1]
                and mask_data[row, col] == 1):
            filtered.append(p)
        else:
            removed += 1

    if verbose:
        print(f"  Parcel filter: kept {len(filtered)}, "
              f"removed {removed} (off-property/reservoir)")

    # Re-number
    for i, p in enumerate(filtered):
        p.id = i + 1

    return filtered


def run_hydro_analysis(dem_path: Path, config: dict, output_dir: Path,
                       is_dsm: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Run complete hydrological analysis pipeline."""

    results = {
        'timestamp': datetime.now().isoformat(),
        'input_dem': str(dem_path),
        'dem_type': 'DSM' if is_dsm else 'DTM',
        'config': config,
        'status': 'running',
        'outputs': {},
        'statistics': {},
        'depressions': [],
        'flat_areas': [],
        'chokepoints': [],
        'problem_areas': [],
        'errors': [],
        'warnings': []
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    hydro_config = config.get('hydrology', DEFAULT_CONFIG['hydrology'])

    if verbose:
        print(f"\n{'='*60}")
        print("Hydrological Analysis Pipeline")
        print(f"Input: {dem_path}")
        print(f"Type: {'DSM (surface model)' if is_dsm else 'DTM (terrain model)'}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

    if is_dsm:
        results['warnings'].append(
            "Using DSM (includes buildings/trees). Results will show surface drainage "
            "patterns. For bare-earth hydrology, generate a DTM from the point cloud."
        )
        if verbose:
            print("\n  WARNING: Using DSM (includes buildings/trees).")
            print("  Surface features will affect flow paths. Consider generating a DTM.")

    # Read input metadata
    meta = read_raster_meta(dem_path)
    epsg = meta['crs_epsg']
    if epsg is None:
        raise ValueError("Input DEM has no EPSG code. Cannot proceed.")

    results['statistics']['input'] = {
        'width_px': meta['width'],
        'height_px': meta['height'],
        'pixel_size_m': meta['pixel_width'],
        'crs_epsg': epsg,
        'elevation_range': [meta['stats']['min'], meta['stats']['max']],
        'mean_elevation': meta['stats']['mean']
    }

    # Downsample if needed
    target_res = hydro_config.get('analysis_resolution_m', DEFAULT_ANALYSIS_RESOLUTION_M)
    work_dir = output_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"\n  Input resolution: {meta['pixel_width']:.4f}m ({meta['pixel_width']*100:.2f}cm)")
        print(f"  Analysis resolution: {target_res}m ({target_res*100:.0f}cm)")

    downsampled_path = work_dir / "dem_analysis.tif"
    downsample_dem(dem_path, downsampled_path, target_res, verbose)

    # Set up GRASS session
    grass_db = work_dir / "grassdb"
    grass = GrassSession(epsg, verbose)

    try:
        grass.setup(grass_db)

        # Run GRASS analysis
        grass_outputs, grass_stats = run_grass_hydro(
            grass, downsampled_path, output_dir, config, verbose
        )
        results['outputs'].update(grass_outputs)
        results['statistics'].update(grass_stats)

        # Post-processing with NumPy
        dep_depth_path = Path(results['outputs'].get('depression_depth', ''))
        filled_dem_path = Path(results['outputs'].get('filled_dem', ''))
        slope_path = Path(results['outputs'].get('slope', ''))
        flow_acc_path = Path(results['outputs'].get('flow_accumulation', ''))

        # Analyze depressions
        if dep_depth_path.exists():
            try:
                depressions, dep_stats = analyze_depressions(
                    dep_depth_path, downsampled_path, config, verbose
                )
                results['depressions'] = [asdict(d) for d in depressions]
                results['statistics']['depressions'] = dep_stats
            except Exception as e:
                results['errors'].append(f"Depression analysis failed: {e}")
                depressions = []
                if verbose:
                    print(f"  Error in depression analysis: {e}")
        else:
            depressions = []

        # Analyze flat areas
        if slope_path.exists():
            try:
                flat_areas, flat_stats = analyze_flat_areas(slope_path, config, verbose)
                results['flat_areas'] = flat_areas
                results['statistics']['flat_areas'] = flat_stats
            except Exception as e:
                results['errors'].append(f"Flat area analysis failed: {e}")
                flat_areas = []
        else:
            flat_areas = []

        # Analyze chokepoints
        if flow_acc_path.exists():
            try:
                chokepoints, cp_stats = analyze_chokepoints(flow_acc_path, downsampled_path, config, verbose)
                results['chokepoints'] = chokepoints
                results['statistics']['chokepoints'] = cp_stats
            except Exception as e:
                results['errors'].append(f"Chokepoint analysis failed: {e}")
                chokepoints = []
        else:
            chokepoints = []

        # Classify problems
        if verbose:
            print("\nStep 11: Classifying problem areas...")

        problems = classify_problems(depressions, flat_areas, chokepoints)

        # Step 12: Filter by parcel boundaries (property land only)
        if verbose:
            print("\nStep 12: Applying parcel boundary filter...")

        parcel_mask_path = fetch_parcel_mask(output_dir, {
            'crs_epsg': epsg,
            'geotransform': meta['geotransform'],
            'width': meta['width'],
            'height': meta['height']
        }, config, verbose)

        if parcel_mask_path and parcel_mask_path.exists():
            problems = filter_problems_by_parcels(problems, parcel_mask_path, verbose)
            results['outputs']['parcel_mask'] = str(parcel_mask_path)
            results['outputs']['parcels_gpkg'] = str(output_dir / "parcels_utm.gpkg")
        else:
            if verbose:
                print("  No parcel data available — reporting all detected problems")

        results['problem_areas'] = [asdict(p) for p in problems]

        results['statistics']['problems'] = {
            'total': len(problems),
            'high_severity': len([p for p in problems if p.severity == 'high']),
            'medium_severity': len([p for p in problems if p.severity == 'medium']),
            'low_severity': len([p for p in problems if p.severity == 'low']),
            'by_type': {
                'ponding': len([p for p in problems if p.type == 'ponding']),
                'flat_area': len([p for p in problems if p.type == 'flat_area']),
                'chokepoint': len([p for p in problems if p.type == 'chokepoint'])
            }
        }

        results['status'] = 'completed'

    except Exception as e:
        results['status'] = 'failed'
        results['errors'].append(str(e))
        if verbose:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        raise

    finally:
        # Clean up GRASS session but keep work dir for debugging
        grass.cleanup()

    # Save results JSON
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    results['outputs']['results_json'] = str(results_path)

    if verbose:
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        stats = results['statistics'].get('problems', {})
        print(f"Problem areas found: {stats.get('total', 0)}")
        print(f"  High severity:   {stats.get('high_severity', 0)}")
        print(f"  Medium severity:  {stats.get('medium_severity', 0)}")
        print(f"  Low severity:     {stats.get('low_severity', 0)}")
        print(f"\nOutputs saved to: {output_dir}")
        for name, path in results['outputs'].items():
            print(f"  {name}: {Path(path).name}")
        print()

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run hydrological analysis on DTM/DSM using GRASS GIS',
        epilog='Run via: run_with_qgis_python.bat 03_hydro_analysis.py [args]'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dtm', help='Path to Digital Terrain Model (preferred)')
    group.add_argument('--dsm', help='Path to Digital Surface Model')

    parser.add_argument('--config', '-c', help='Community config JSON')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--resolution', '-r', type=float,
                        help=f'Analysis resolution in meters (default: {DEFAULT_ANALYSIS_RESOLUTION_M})')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config.update(json.load(f))

    if args.resolution:
        config.setdefault('hydrology', {})['analysis_resolution_m'] = args.resolution

    if args.dtm:
        dem_path = Path(args.dtm)
        is_dsm = False
    else:
        dem_path = Path(args.dsm)
        is_dsm = True

    results = run_hydro_analysis(
        dem_path, config, Path(args.output),
        is_dsm=is_dsm, verbose=not args.quiet
    )

    sys.exit(0 if results['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
