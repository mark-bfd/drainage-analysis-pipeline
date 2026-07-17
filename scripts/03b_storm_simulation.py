#!/usr/bin/env python3
"""
03b_storm_simulation.py - Rainfall-Runoff Storm Simulation
Drone Drainage Analysis Pipeline

Simulates rainfall at incremental depths (0.5" to 2.0") on the terrain
using GRASS r.sim.water to identify where water accumulates, backs up
behind structures, and floods properties.

This replaces the static flow-accumulation chokepoint analysis with
physics-based shallow water simulation.

Usage:
    python 03b_storm_simulation.py --dsm dsm.tif --config config.json --output ./output
    python 03b_storm_simulation.py --dsm dsm.tif --output ./output --parcels parcels.gpkg

Requires QGIS 3.40.15 environment with GRASS GIS 8.4.
"""

import os
import sys
import json
import argparse
import shutil
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from osgeo import gdal, ogr, osr
    import numpy as np
except ImportError:
    print("ERROR: GDAL/numpy not found. Run with QGIS Python.")
    sys.exit(1)

gdal.UseExceptions()

# =============================================================================
# Configuration
# =============================================================================

QGIS_ROOT = Path(r"C:\Program Files\QGIS 3.40.15")
GISBASE = QGIS_ROOT / "apps" / "grass" / "grass84"
GRASS_BIN = QGIS_ROOT / "bin" / "grass84.bat"

# Rainfall scenarios: inches -> mm/hr intensity for a 1-hour storm
# 0.5" in 1 hour = 12.7 mm/hr, etc.
RAINFALL_SCENARIOS = [
    {"label": "0.5 inch", "inches": 0.5, "mm_hr": 12.7},
    {"label": "1.0 inch", "inches": 1.0, "mm_hr": 25.4},
    {"label": "1.5 inch", "inches": 1.5, "mm_hr": 38.1},
    {"label": "2.0 inch", "inches": 2.0, "mm_hr": 50.8},
]

# Default soil infiltration for East Texas clay (Group D) in mm/hr
# This represents steady-state infiltration after initial saturation
DEFAULT_INFILTRATION_MM_HR = 2.5

# Manning's n for mixed residential (lawns, pavement, structures)
DEFAULT_MANNINGS_N = 0.035

# Simulation duration in minutes (how long to run each scenario)
SIM_DURATION_MINUTES = 10

# Minimum water depth to be considered "flooding" (meters)
FLOOD_DEPTH_THRESHOLDS = {
    'nuisance': 0.025,   # 1 inch - puddles, nuisance
    'minor':    0.075,   # 3 inches - yard flooding
    'moderate': 0.15,    # 6 inches - threatens structures
    'major':    0.30,    # 12 inches - significant property damage
}

# Analysis resolution for simulation
ANALYSIS_RESOLUTION_M = 0.5  # 50cm for faster sim (r.sim.water is intensive)


@dataclass
class FloodZone:
    """A contiguous area of flooding on property land."""
    id: int
    scenario_inches: float
    centroid_x: float
    centroid_y: float
    area_sqm: float
    max_depth_m: float
    mean_depth_m: float
    volume_m3: float
    severity: str  # nuisance, minor, moderate, major
    first_appears_at_inches: float  # rainfall amount when this zone first floods


@dataclass
class Chokepoint:
    """A location where water backs up during storms."""
    id: int
    centroid_x: float
    centroid_y: float
    description: str
    depth_at_05in: float  # water depth at 0.5" rainfall
    depth_at_10in: float
    depth_at_15in: float
    depth_at_20in: float
    escalation_rate: float  # how fast depth increases per inch of rain
    severity: str


# =============================================================================
# GRASS Session (reuse from 03_hydro_analysis.py)
# =============================================================================

class GrassSession:
    """Manages a GRASS GIS session."""

    def __init__(self, epsg: int, verbose: bool = True):
        self.epsg = epsg
        self.verbose = verbose
        self.grassdb = None
        self.location = "storm_sim"
        self.mapset = "PERMANENT"
        self.env = None

    def setup(self, grassdb_path: Path):
        self.grassdb = grassdb_path
        self.grassdb.mkdir(parents=True, exist_ok=True)

        self.env = os.environ.copy()
        self.env["GISBASE"] = str(GISBASE)
        self.env["GISRC"] = str(self.grassdb / "gisrc")
        self.env["GRASS_PYTHON"] = str(QGIS_ROOT / "apps" / "Python312" / "python.exe")
        self.env["GRASS_PROJSHARE"] = str(QGIS_ROOT / "share" / "proj")
        self.env["PROJ_LIB"] = str(QGIS_ROOT / "share" / "proj")
        self.env["GDAL_DATA"] = str(QGIS_ROOT / "apps" / "gdal" / "share" / "gdal")

        grass_bin = str(GISBASE / "bin")
        grass_lib = str(GISBASE / "lib")
        grass_scripts = str(GISBASE / "scripts")
        qgis_bin = str(QGIS_ROOT / "bin")
        gdal_bin = str(QGIS_ROOT / "apps" / "gdal" / "bin")
        self.env["PATH"] = f"{grass_bin};{grass_lib};{grass_scripts};{qgis_bin};{gdal_bin};{self.env['PATH']}"

        # Create location
        location_path = self.grassdb / self.location
        if not location_path.exists():
            self._run_grass(["-c", f"EPSG:{self.epsg}",
                            str(location_path), "--text", "--exec", "g.version"],
                           raw=True)

        # Write GISRC
        (self.grassdb / "gisrc").write_text(
            f"GISDBASE: {self.grassdb}\n"
            f"LOCATION_NAME: {self.location}\n"
            f"MAPSET: {self.mapset}\n"
        )

    def _run_grass(self, args, raw=False, timeout=3600):
        grass_bat = str(GRASS_BIN)
        if raw:
            cmd = [grass_bat] + args
        else:
            location_path = str(self.grassdb / self.location / self.mapset)
            cmd = [grass_bat, location_path, "--text", "--exec"] + args

        result = subprocess.run(cmd, env=self.env, capture_output=True,
                               text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            stderr = result.stderr.strip()
            if stderr and "WARNING" not in stderr and "HINT" not in stderr:
                # Only raise on real errors
                if "ERROR" in stderr:
                    raise RuntimeError(f"GRASS: {stderr[:500]}")
        return result

    def run(self, module: str, **kwargs) -> str:
        args = [module]
        for k, v in kwargs.items():
            if k == 'flags':
                args.append(f"-{v}")
            elif v is True:
                args.append(f"--{k}" if k in ('overwrite', 'quiet', 'verbose') else f"-{k}")
            elif v is not None and v is not False:
                key = k.replace("__", ".")
                args.append(f"{key}={v}")

        result = self._run_grass(args)
        return result.stdout.strip() if result.stdout else ""

    def cleanup(self):
        if self.grassdb and self.grassdb.exists():
            try:
                shutil.rmtree(self.grassdb)
            except Exception:
                pass


# =============================================================================
# Raster Utilities
# =============================================================================

def read_raster_meta(filepath: Path) -> dict:
    ds = gdal.Open(str(filepath))
    if ds is None:
        raise ValueError(f"Cannot open: {filepath}")
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    band = ds.GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    meta = {
        'width': ds.RasterXSize, 'height': ds.RasterYSize,
        'geotransform': gt, 'projection': ds.GetProjection(),
        'pixel_width': abs(gt[1]), 'pixel_height': abs(gt[5]),
        'nodata': band.GetNoDataValue(),
        'crs_epsg': None,
        'stats': {'min': stats[0], 'max': stats[1], 'mean': stats[2], 'std': stats[3]}
    }
    if srs.GetAuthorityName(None) == 'EPSG':
        meta['crs_epsg'] = int(srs.GetAuthorityCode(None))
    ds = None
    return meta


def read_raster(filepath: Path):
    ds = gdal.Open(str(filepath))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    gt = ds.GetGeoTransform()
    nodata = band.GetNoDataValue()
    ds = None
    if nodata is not None:
        data = data.astype(np.float32)
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data, gt


def downsample_dem(input_path: Path, output_path: Path, target_res: float):
    """Downsample DEM using GDAL."""
    env = os.environ.copy()
    env["PATH"] = f"{QGIS_ROOT / 'bin'};{QGIS_ROOT / 'apps' / 'gdal' / 'bin'};{env['PATH']}"
    env["GDAL_DATA"] = str(QGIS_ROOT / "apps" / "gdal" / "share" / "gdal")
    env["PROJ_LIB"] = str(QGIS_ROOT / "share" / "proj")

    cmd = [
        str(QGIS_ROOT / "bin" / "gdalwarp.exe"),
        "-tr", str(target_res), str(target_res),
        "-r", "average",
        "-co", "COMPRESS=LZW", "-co", "TILED=YES", "-co", "BIGTIFF=YES",
        str(input_path), str(output_path)
    ]
    subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)


# =============================================================================
# Storm Simulation
# =============================================================================

def run_storm_simulation(grass: GrassSession, dem_path: Path, output_dir: Path,
                         config: dict, verbose: bool = True) -> Dict:
    """Run r.sim.water for each rainfall scenario."""

    hydro = config.get('hydrology', {})
    soil = config.get('soil', {})

    infil = soil.get('infiltration_mm_hr', DEFAULT_INFILTRATION_MM_HR)
    mannings = hydro.get('mannings_n', DEFAULT_MANNINGS_N)
    sim_minutes = hydro.get('sim_duration_minutes', SIM_DURATION_MINUTES)
    scenarios = hydro.get('rainfall_scenarios', RAINFALL_SCENARIOS)

    results = {
        'scenarios': [],
        'depth_rasters': {},
        'discharge_rasters': {}
    }

    # Import DEM
    if verbose:
        print("\n  Importing DEM into GRASS...")

    grass.run("r.in.gdal", input=str(dem_path), output="dem", overwrite=True)
    grass.run("g.region", raster="dem")

    # Compute slope derivatives (required by r.sim.water)
    if verbose:
        print("  Computing terrain derivatives (dx, dy)...")
    grass.run("r.slope.aspect", elevation="dem",
              dx="dx", dy="dy", overwrite=True)

    # Get CPU count for parallel processing
    nprocs = min(os.cpu_count() or 4, 8)

    # Run each rainfall scenario
    for scenario in scenarios:
        label = scenario['label']
        rain_mm_hr = scenario['mm_hr']
        rain_inches = scenario['inches']

        # Net rainfall = gross rainfall - infiltration
        net_rain = max(rain_mm_hr - infil, 0.1)

        if verbose:
            print(f"\n  === Scenario: {label} ({rain_mm_hr:.1f} mm/hr, "
                  f"net: {net_rain:.1f} mm/hr after {infil} mm/hr infiltration) ===")

        depth_name = f"depth_{label.replace(' ', '_').replace('.', '')}"
        discharge_name = f"discharge_{label.replace(' ', '_').replace('.', '')}"

        t0 = time.time()

        try:
            grass.run("r.sim.water",
                      elevation="dem",
                      dx="dx", dy="dy",
                      rain_value=str(net_rain),
                      man_value=str(mannings),
                      depth=depth_name,
                      discharge=discharge_name,
                      niterations=str(sim_minutes),
                      nwalkers=str(200000),  # 200K walkers (good balance of speed/accuracy)
                      nprocs=str(nprocs),
                      overwrite=True)

            elapsed = time.time() - t0
            if verbose:
                print(f"    Simulation complete ({elapsed:.0f}s)")

            # Export depth raster
            depth_path = output_dir / f"storm_depth_{rain_inches:.1f}in.tif"
            grass.run("r.out.gdal", input=depth_name, output=str(depth_path),
                      format="GTiff", createopt="COMPRESS=LZW,TILED=YES",
                      overwrite=True)

            # Get statistics
            stats_raw = grass.run("r.univar", map=depth_name, flags="g")
            stats = {}
            for line in stats_raw.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try:
                        stats[k.strip()] = float(v.strip())
                    except ValueError:
                        pass

            scenario_result = {
                'label': label,
                'inches': rain_inches,
                'rain_mm_hr': rain_mm_hr,
                'net_rain_mm_hr': net_rain,
                'infiltration_mm_hr': infil,
                'sim_duration_min': sim_minutes,
                'depth_raster': str(depth_path),
                'max_depth_m': stats.get('max', 0),
                'mean_depth_m': stats.get('mean', 0),
                'cells_gt_1in': int(stats.get('n', 0)),  # will refine below
                'elapsed_s': elapsed
            }

            results['scenarios'].append(scenario_result)
            results['depth_rasters'][rain_inches] = str(depth_path)

            if verbose:
                print(f"    Max depth: {stats.get('max', 0):.3f}m "
                      f"({stats.get('max', 0) * 39.37:.1f} inches)")
                print(f"    Mean depth: {stats.get('mean', 0):.4f}m")

        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            results['scenarios'].append({
                'label': label, 'inches': rain_inches, 'error': str(e)
            })

    return results


# =============================================================================
# Post-processing: Find flood zones and chokepoints on parcels
# =============================================================================

def analyze_flood_zones(depth_rasters: Dict[float, Path], parcel_mask_path: Optional[Path],
                        config: dict, verbose: bool = True) -> Tuple[List[FloodZone], List[Chokepoint]]:
    """Analyze water depth across scenarios to find flood zones and chokepoints."""
    from scipy import ndimage

    if verbose:
        print("\n  Analyzing flood zones across scenarios...")

    # Load parcel mask
    if parcel_mask_path and parcel_mask_path.exists():
        parcel_data, parcel_gt = read_raster(parcel_mask_path)
        on_parcel = np.isfinite(parcel_data) & (parcel_data > 0)
    else:
        on_parcel = None

    # Load all depth rasters
    depths = {}
    gt = None
    for inches, path in sorted(depth_rasters.items()):
        path = Path(path)
        if path.exists():
            data, g = read_raster(path)
            data[~np.isfinite(data)] = 0
            data[data < 0] = 0
            depths[inches] = data
            if gt is None:
                gt = g

    if not depths:
        return [], []

    pixel_area = abs(gt[1]) * abs(gt[5])

    # Use the highest rainfall scenario for flood zone detection
    max_scenario = max(depths.keys())
    max_depth = depths[max_scenario]

    # Apply parcel mask if available
    if on_parcel is not None:
        # Ensure shapes match (they should if both came from same GRASS region)
        if on_parcel.shape != max_depth.shape:
            if verbose:
                print(f"    Warning: parcel mask shape {on_parcel.shape} != "
                      f"depth shape {max_depth.shape}. Skipping parcel filter.")
            on_parcel = None

    # Find flood zones at the highest scenario
    flood_zones = []
    for sev_name, threshold in FLOOD_DEPTH_THRESHOLDS.items():
        flood_mask = max_depth > threshold
        if on_parcel is not None:
            flood_mask = flood_mask & on_parcel

        labeled, num = ndimage.label(flood_mask)

        for i in range(1, num + 1):
            region = labeled == i
            n_px = int(np.sum(region))
            area = n_px * pixel_area

            if area < 5.0:  # Skip tiny zones < 5 m²
                continue

            region_depth = max_depth[region]
            max_d = float(np.max(region_depth))
            mean_d = float(np.mean(region_depth))
            vol = float(np.sum(region_depth) * pixel_area)

            rows, cols = np.where(region)
            cx = gt[0] + np.mean(cols) * gt[1]
            cy = gt[3] + np.mean(rows) * gt[5]

            # Find when this zone first appears (at what rainfall)
            first_appears = max_scenario
            for inc in sorted(depths.keys()):
                if inc in depths:
                    zone_depth_at_inc = depths[inc][region]
                    if np.any(zone_depth_at_inc > threshold):
                        first_appears = inc
                        break

            fz = FloodZone(
                id=len(flood_zones) + 1,
                scenario_inches=max_scenario,
                centroid_x=cx, centroid_y=cy,
                area_sqm=area,
                max_depth_m=max_d,
                mean_depth_m=mean_d,
                volume_m3=vol,
                severity=sev_name,
                first_appears_at_inches=first_appears
            )
            flood_zones.append(fz)

    # Deduplicate: smaller severity zones overlap with larger ones
    # Keep only the highest severity for each location
    # (a "major" zone contains the "moderate" zone at the same spot)
    unique_zones = []
    sev_rank = {'major': 0, 'moderate': 1, 'minor': 2, 'nuisance': 3}
    flood_zones.sort(key=lambda z: (sev_rank.get(z.severity, 9), -z.area_sqm))

    used_locations = set()
    for fz in flood_zones:
        # Round centroid to 5m grid for dedup
        loc_key = (round(fz.centroid_x / 5) * 5, round(fz.centroid_y / 5) * 5)
        if loc_key not in used_locations:
            used_locations.add(loc_key)
            unique_zones.append(fz)

    # Re-ID
    for i, fz in enumerate(unique_zones):
        fz.id = i + 1

    if verbose:
        by_sev = {}
        for fz in unique_zones:
            by_sev[fz.severity] = by_sev.get(fz.severity, 0) + 1
        print(f"    Flood zones found: {len(unique_zones)}")
        for s in ['major', 'moderate', 'minor', 'nuisance']:
            if s in by_sev:
                print(f"      {s}: {by_sev[s]}")

    # --- Identify chokepoints ---
    # A chokepoint is where water depth escalates rapidly with rainfall
    # Look for pixels where depth increases disproportionately between scenarios
    if verbose:
        print("\n  Identifying storm chokepoints...")

    sorted_scenarios = sorted(depths.keys())
    if len(sorted_scenarios) >= 2:
        # Build depth-progression array for each pixel
        first = depths[sorted_scenarios[0]]
        last = depths[sorted_scenarios[-1]]

        # Chokepoint: large depth at final scenario AND rapid escalation
        # Escalation = depth increase per inch of additional rainfall
        rainfall_range = sorted_scenarios[-1] - sorted_scenarios[0]
        if rainfall_range > 0:
            escalation = (last - first) / rainfall_range
        else:
            escalation = np.zeros_like(first)

        # Filter: on parcel, significant depth, high escalation
        cp_mask = (last > 0.05) & (escalation > 0.02) & np.isfinite(escalation)
        if on_parcel is not None and on_parcel.shape == cp_mask.shape:
            cp_mask = cp_mask & on_parcel

        # Cluster into chokepoint regions
        labeled_cp, num_cp = ndimage.label(cp_mask)

        chokepoints = []
        for i in range(1, num_cp + 1):
            region = labeled_cp == i
            n_px = int(np.sum(region))
            area = n_px * pixel_area

            if area < 3.0:
                continue

            rows, cols = np.where(region)
            cx = gt[0] + np.mean(cols) * gt[1]
            cy = gt[3] + np.mean(rows) * gt[5]

            # Get depth at each scenario
            depth_by_scenario = {}
            for inc in sorted_scenarios:
                region_depth = depths[inc][region]
                depth_by_scenario[inc] = float(np.max(region_depth))

            esc_rate = float(np.max(escalation[region]))

            # Severity based on depth at 2" and escalation
            max_d = depth_by_scenario.get(2.0, depth_by_scenario.get(sorted_scenarios[-1], 0))
            if max_d > 0.30:
                severity = 'high'
            elif max_d > 0.15:
                severity = 'medium'
            else:
                severity = 'low'

            desc = (f"Water backs up to {max_d:.2f}m ({max_d*39.37:.1f}\") at 2\" rain, "
                    f"escalating at {esc_rate:.3f}m per inch")

            cp = Chokepoint(
                id=len(chokepoints) + 1,
                centroid_x=cx, centroid_y=cy,
                description=desc,
                depth_at_05in=depth_by_scenario.get(0.5, 0),
                depth_at_10in=depth_by_scenario.get(1.0, 0),
                depth_at_15in=depth_by_scenario.get(1.5, 0),
                depth_at_20in=depth_by_scenario.get(2.0, 0),
                escalation_rate=esc_rate,
                severity=severity
            )
            chokepoints.append(cp)

        # Sort by severity then escalation
        sev_order = {'high': 0, 'medium': 1, 'low': 2}
        chokepoints.sort(key=lambda c: (sev_order.get(c.severity, 3), -c.escalation_rate))

        # Cap at top 30
        chokepoints = chokepoints[:30]
        for i, cp in enumerate(chokepoints):
            cp.id = i + 1

        if verbose:
            print(f"    Chokepoints found: {len(chokepoints)}")
            for cp in chokepoints[:5]:
                print(f"      #{cp.id} [{cp.severity}] {cp.description}")

    else:
        chokepoints = []

    return unique_zones, chokepoints


# =============================================================================
# Main
# =============================================================================

def run_storm_analysis(dem_path: Path, config: dict, output_dir: Path,
                       verbose: bool = True) -> Dict:
    """Run complete storm simulation pipeline."""

    results = {
        'timestamp': datetime.now().isoformat(),
        'input_dem': str(dem_path),
        'status': 'running',
        'simulation': {},
        'flood_zones': [],
        'chokepoints': [],
        'statistics': {},
        'outputs': {},
        'errors': []
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    meta = read_raster_meta(dem_path)
    epsg = meta['crs_epsg']

    if verbose:
        print(f"\n{'='*60}")
        print("Storm Simulation")
        print(f"Input: {dem_path}")
        print(f"{'='*60}")

    # Downsample for simulation
    target_res = config.get('hydrology', {}).get(
        'sim_resolution_m', ANALYSIS_RESOLUTION_M)

    if verbose:
        print(f"\n  Input resolution: {meta['pixel_width']:.4f}m")
        print(f"  Simulation resolution: {target_res}m")

    sim_dem = work_dir / "dem_sim.tif"
    if meta['pixel_width'] < target_res * 0.9:
        if verbose:
            print(f"  Downsampling to {target_res}m...")
        downsample_dem(dem_path, sim_dem, target_res)
    else:
        shutil.copy2(str(dem_path), str(sim_dem))

    sim_meta = read_raster_meta(sim_dem)
    if verbose:
        print(f"  Simulation grid: {sim_meta['width']}x{sim_meta['height']} cells")

    # Set up GRASS
    grass_db = work_dir / "grassdb_sim"
    grass = GrassSession(epsg, verbose)

    try:
        grass.setup(grass_db)

        # Run simulations
        sim_results = run_storm_simulation(grass, sim_dem, output_dir, config, verbose)
        results['simulation'] = sim_results

        # Analyze flood zones with parcel filter
        parcel_mask = output_dir / "parcel_mask.tif"
        if not parcel_mask.exists():
            parcel_mask = None

        flood_zones, chokepoints = analyze_flood_zones(
            sim_results.get('depth_rasters', {}),
            parcel_mask, config, verbose
        )

        results['flood_zones'] = [asdict(fz) for fz in flood_zones]
        results['chokepoints'] = [asdict(cp) for cp in chokepoints]

        results['statistics'] = {
            'total_flood_zones': len(flood_zones),
            'total_chokepoints': len(chokepoints),
            'by_severity': {
                s: len([fz for fz in flood_zones if fz.severity == s])
                for s in ['major', 'moderate', 'minor', 'nuisance']
            },
            'chokepoints_by_severity': {
                s: len([cp for cp in chokepoints if cp.severity == s])
                for s in ['high', 'medium', 'low']
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
        grass.cleanup()

    # Save results
    results_path = output_dir / "storm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    results['outputs']['results_json'] = str(results_path)

    if verbose:
        print(f"\n{'='*60}")
        print("STORM SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Flood zones on property: {len(flood_zones)}")
        for s in ['major', 'moderate', 'minor', 'nuisance']:
            count = len([fz for fz in flood_zones if fz.severity == s])
            if count:
                print(f"  {s}: {count}")
        print(f"Chokepoints: {len(chokepoints)}")
        print(f"\nOutputs: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run rainfall storm simulation on DSM/DTM'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dtm', help='Path to DTM')
    group.add_argument('--dsm', help='Path to DSM')

    parser.add_argument('--config', '-c', help='Community config JSON')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--resolution', '-r', type=float,
                        help=f'Sim resolution in meters (default: {ANALYSIS_RESOLUTION_M})')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    config = {}
    if args.config:
        p = Path(args.config)
        if p.exists():
            with open(p) as f:
                config = json.load(f)

    if args.resolution:
        config.setdefault('hydrology', {})['sim_resolution_m'] = args.resolution

    dem_path = Path(args.dtm or args.dsm)
    results = run_storm_analysis(dem_path, config, Path(args.output),
                                 verbose=not args.quiet)
    sys.exit(0 if results['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
