#!/usr/bin/env python3
"""
03d_anuga_storm.py - ANUGA 2D Shallow Water Storm Simulation
Drone Drainage Analysis Pipeline

Fully automated rainfall-runoff simulation using ANUGA.
No GUI required — runs entirely from command line.

Simulates rainfall at 0.5" increments on the terrain mesh,
producing water depth rasters that show where flooding occurs
on properties during storm events.

Usage:
    python 03d_anuga_storm.py --dsm terrain.tif --boundary boundary.shp --output ./output
    python 03d_anuga_storm.py --dsm terrain.tif --parcels parcels.gpkg --output ./output --config config.json

Requires: anuga, rasterio, numpy, scipy, matplotlib (use anuga-env venv)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import rasterio
from rasterio.transform import from_bounds, rowcol
from scipy import ndimage

# ANUGA imports
os.environ['ANUGA_PARALLEL'] = '0'  # Disable MPI warnings
import anuga
from anuga import Domain, Reflective_boundary, Dirichlet_boundary, Transmissive_boundary
from anuga import Rainfall

# =============================================================================
# Configuration
# =============================================================================

RAINFALL_SCENARIOS = [
    {"label": "0.5 inch", "inches": 0.5},
    {"label": "1.0 inch", "inches": 1.0},
    {"label": "1.5 inch", "inches": 1.5},
    {"label": "2.0 inch", "inches": 2.0},
]

# Soil infiltration (East TX clay, Group D)
DEFAULT_INFILTRATION_MM_HR = 2.5

# Manning's n
DEFAULT_MANNINGS_N = 0.035

# Simulation time: how long to run (seconds of simulated time)
# For a 1-hour storm, run long enough for water to reach steady state
SIM_DURATION_S = 1800  # 30 minutes simulated time

# Output interval
OUTPUT_INTERVAL_S = 300  # save state every 5 min

# Analysis resolution
DEFAULT_RESOLUTION_M = 1.0

# Flood depth thresholds (meters)
FLOOD_THRESHOLDS = {
    'nuisance': 0.025,   # 1"
    'minor':    0.075,   # 3"
    'moderate': 0.15,    # 6"
    'major':    0.30,    # 12"
}


@dataclass
class FloodZone:
    id: int
    scenario_inches: float
    centroid_x: float
    centroid_y: float
    area_sqm: float
    max_depth_m: float
    mean_depth_m: float
    volume_m3: float
    severity: str
    first_appears_at_inches: float


@dataclass
class Chokepoint:
    id: int
    centroid_x: float
    centroid_y: float
    description: str
    depth_at_05in: float
    depth_at_10in: float
    depth_at_15in: float
    depth_at_20in: float
    escalation_rate: float
    severity: str


# =============================================================================
# Terrain & Mesh Setup
# =============================================================================

def load_terrain(dsm_path: Path, target_res: float, verbose=True):
    """Load and downsample DSM. Returns (elevation_array, transform, bounds, crs)."""
    if verbose:
        print(f"  Loading terrain: {dsm_path}")

    with rasterio.open(str(dsm_path)) as src:
        native_res = abs(src.res[0])
        bounds = src.bounds
        crs = src.crs

        if native_res < target_res * 0.9:
            # Downsample
            scale = native_res / target_res
            out_h = max(1, int(src.height * scale))
            out_w = max(1, int(src.width * scale))
            data = src.read(1, out_shape=(out_h, out_w),
                           resampling=rasterio.enums.Resampling.average)
            transform = from_bounds(bounds.left, bounds.bottom,
                                   bounds.right, bounds.top, out_w, out_h)
        else:
            data = src.read(1)
            out_h, out_w = data.shape
            transform = src.transform

    # Handle nodata
    data = data.astype(np.float64)
    data[~np.isfinite(data)] = np.nan

    if verbose:
        valid = np.isfinite(data)
        print(f"  Grid: {out_w}x{out_h} at {target_res}m")
        print(f"  Elevation: {np.nanmin(data):.1f} - {np.nanmax(data):.1f}m")
        print(f"  Valid cells: {np.sum(valid):,} ({np.sum(valid)/data.size*100:.0f}%)")

    return data, transform, bounds, crs


def load_boundary(boundary_path: Path, verbose=True):
    """Load boundary polygon from GeoJSON, shapefile, or GeoPackage."""
    path = str(boundary_path)

    # Read GeoJSON with stdlib json (no GDAL needed)
    if path.endswith('.geojson') or path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        geom = data['features'][0]['geometry']
        if geom['type'] == 'MultiPolygon':
            # Take the ring with most coordinates
            rings = [ring[0] for ring in geom['coordinates']]
            ring = max(rings, key=len)
        else:
            ring = geom['coordinates'][0]
        coords = [(c[0], c[1]) for c in ring]
    else:
        # Try osgeo for shapefiles/gpkg
        try:
            from osgeo import ogr
            ds = ogr.Open(path)
            layer = ds.GetLayer(0)
            feat = layer.GetNextFeature()
            geom = feat.GetGeometryRef()
            if geom.GetGeometryType() == 6:  # MultiPolygon
                max_area, largest_idx = 0, 0
                for i in range(geom.GetGeometryCount()):
                    if geom.GetGeometryRef(i).GetArea() > max_area:
                        max_area = geom.GetGeometryRef(i).GetArea()
                        largest_idx = i
                geom = geom.GetGeometryRef(largest_idx)
            ring = geom.GetGeometryRef(0)
            coords = [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]
            ds = None
        except ImportError:
            raise RuntimeError(f"Cannot read {path}: install osgeo or use GeoJSON format")

    # Simplify if too many points (ANUGA mesh gen slows with >200 boundary points)
    if len(coords) > 150:
        step = max(1, len(coords) // 100)
        simplified = coords[::step]
        if simplified[-1] != coords[0]:
            simplified.append(coords[0])
        coords = simplified

    if verbose:
        print(f"  Boundary: {len(coords)} vertices")

    return coords


def create_anuga_domain(elevation: np.ndarray, transform, bounds,
                        boundary_coords: list, resolution: float,
                        mannings_n: float, verbose=True) -> Domain:
    """Create ANUGA domain with terrain and mesh."""
    if verbose:
        print("\n  Creating ANUGA mesh...")

    # ANUGA needs the boundary as a closed polygon
    bounding_polygon = [(x, y) for x, y in boundary_coords]

    # Create domain with mesh
    domain = anuga.create_domain_from_regions(
        bounding_polygon,
        boundary_tags={'exterior': range(len(bounding_polygon))},
        maximum_triangle_area=resolution * resolution,
        use_cache=False,
        verbose=verbose
    )

    domain.set_name('storm_sim')
    domain.set_datadir('.')

    # Set Manning friction
    domain.set_quantity('friction', mannings_n)

    # Set elevation from the raster
    # Create an interpolation function from the grid
    rows, cols = elevation.shape
    x_coords = np.array([transform[2] + j * transform[0] + transform[0]/2
                         for j in range(cols)])
    y_coords = np.array([transform[5] + i * transform[4] + transform[4]/2
                         for i in range(rows)])

    def elevation_function(x, y):
        """Interpolate elevation at arbitrary (x,y) points."""
        # Convert to grid coordinates
        col_f = (x - transform[2]) / transform[0]
        row_f = (y - transform[5]) / transform[4]

        col_i = np.clip(np.round(col_f).astype(int), 0, cols - 1)
        row_i = np.clip(np.round(row_f).astype(int), 0, rows - 1)

        z = elevation[row_i, col_i]
        # Replace NaN with a high value (acts as a wall)
        z = np.where(np.isfinite(z), z, 200.0)
        return z

    domain.set_quantity('elevation', elevation_function, location='centroids')
    domain.set_quantity('stage', elevation_function, location='centroids')  # dry initially
    domain.set_quantity('xmomentum', 0.0)
    domain.set_quantity('ymomentum', 0.0)

    # Boundaries: reflective (water bounces off edges)
    Br = Reflective_boundary(domain)
    domain.set_boundary({'exterior': Br})

    if verbose:
        n_triangles = len(domain)
        print(f"  Mesh: {n_triangles:,} triangles")

    return domain


# =============================================================================
# Simulation
# =============================================================================

def run_scenario(domain: Domain, rain_inches: float, infil_mm_hr: float,
                 duration_s: float, verbose=True) -> np.ndarray:
    """Run one rainfall scenario. Returns final water depth at centroids."""
    if verbose:
        print(f"\n  === Simulating {rain_inches:.1f}\" rainfall ===")

    # Convert inches/hr to m/s for ANUGA
    rain_mm_hr = rain_inches * 25.4  # inch to mm
    net_rain_mm_hr = max(rain_mm_hr - infil_mm_hr, 0.1)
    rain_m_s = net_rain_mm_hr / (1000 * 3600)  # mm/hr to m/s

    if verbose:
        print(f"    Gross: {rain_mm_hr:.1f} mm/hr, Net: {net_rain_mm_hr:.1f} mm/hr, "
              f"Rate: {rain_m_s:.2e} m/s")

    # Reset domain to dry (stage = elevation = no water)
    elev_vals = domain.get_quantity('elevation').get_values(location='centroids')
    domain.set_quantity('stage', elev_vals.flatten(), location='centroids')
    domain.set_quantity('xmomentum', 0.0)
    domain.set_quantity('ymomentum', 0.0)
    domain.set_time(0.0)

    # Run simulation with manual rainfall addition at each timestep
    # (ANUGA's Rainfall operator has a units bug; manual addition is reliable)
    t0 = time.time()
    yieldstep = 60  # add rain every 60 seconds
    dt_report = max(yieldstep, duration_s / 5)

    stage_q = domain.get_quantity('stage')
    elev_q = domain.get_quantity('elevation')
    last_report = 0

    for t in domain.evolve(yieldstep=yieldstep, finaltime=duration_s):
        # Add rainfall depth for this timestep
        s_vals = stage_q.get_values(location='centroids').flatten()
        s_vals += rain_m_s * yieldstep
        stage_q.set_values(s_vals, location='centroids')

        elapsed_sim = domain.get_time()
        if verbose and (elapsed_sim - last_report >= dt_report or elapsed_sim >= duration_s):
            e_vals = elev_q.get_values(location='centroids').flatten()
            depth_now = np.maximum(s_vals - e_vals, 0)
            max_d = np.max(depth_now)
            print(f"    t={elapsed_sim:.0f}s  max_depth={max_d:.3f}m ({max_d*39.37:.1f}\")"
                  f"  wet={np.sum(depth_now > 0.001):,}")
            last_report = elapsed_sim

    wall_time = time.time() - t0

    # Extract final water depth
    stage = stage_q.get_values(location='centroids').flatten()
    elev = elev_q.get_values(location='centroids').flatten()
    depth = np.maximum(stage - elev, 0)

    if verbose:
        print(f"    Complete ({wall_time:.0f}s wall time)")
        print(f"    Max depth: {np.max(depth):.3f}m ({np.max(depth)*39.37:.1f}\")")
        print(f"    Mean depth: {np.mean(depth):.4f}m")
        print(f"    Cells > 1\": {np.sum(depth > 0.025):,}")

    return depth


def depth_to_raster(domain: Domain, depth: np.ndarray, bounds, resolution: float,
                    output_path: Path, crs) -> Path:
    """Rasterize ANUGA triangular depth values to a regular grid GeoTIFF."""
    # Get centroid coordinates
    centroids = domain.get_centroid_coordinates(absolute=True)
    cx = centroids[:, 0]
    cy = centroids[:, 1]

    # Create output grid
    out_w = int((bounds.right - bounds.left) / resolution)
    out_h = int((bounds.top - bounds.bottom) / resolution)
    grid = np.full((out_h, out_w), np.nan, dtype=np.float32)

    # Map centroids to grid cells
    col_i = ((cx - bounds.left) / resolution).astype(int)
    row_i = ((bounds.top - cy) / resolution).astype(int)

    valid = (col_i >= 0) & (col_i < out_w) & (row_i >= 0) & (row_i < out_h)
    grid[row_i[valid], col_i[valid]] = depth[valid].astype(np.float32)

    # Write GeoTIFF
    transform = from_bounds(bounds.left, bounds.bottom,
                           bounds.right, bounds.top, out_w, out_h)

    with rasterio.open(str(output_path), 'w', driver='GTiff',
                       height=out_h, width=out_w, count=1,
                       dtype='float32', crs=crs, transform=transform,
                       nodata=np.nan,
                       compress='lzw', tiled=True) as dst:
        dst.write(grid, 1)

    return output_path


# =============================================================================
# Post-processing
# =============================================================================

def analyze_results(depth_rasters: Dict[float, Path], parcel_mask_path: Optional[Path],
                    verbose=True):
    """Analyze flood zones and chokepoints from depth rasters."""
    if verbose:
        print("\n  Analyzing flood zones across scenarios...")

    # Load parcel mask
    on_parcel = None
    if parcel_mask_path and Path(parcel_mask_path).exists():
        with rasterio.open(str(parcel_mask_path)) as src:
            parcel_data = src.read(1)
            on_parcel = np.isfinite(parcel_data) & (parcel_data > 0)

    # Load depth rasters
    depths = {}
    gt = None
    shape = None
    for inches, path in sorted(depth_rasters.items()):
        with rasterio.open(str(path)) as src:
            data = src.read(1)
            data[~np.isfinite(data)] = 0
            data[data < 0] = 0
            depths[inches] = data
            if gt is None:
                gt = src.transform
                shape = data.shape

    if not depths:
        return [], []

    pixel_area = abs(gt[0]) * abs(gt[4])
    max_scenario = max(depths.keys())
    max_depth = depths[max_scenario]

    # Resize parcel mask if shapes don't match
    if on_parcel is not None and on_parcel.shape != shape:
        if verbose:
            print(f"    Resizing parcel mask from {on_parcel.shape} to {shape}")
        from scipy.ndimage import zoom
        scale_y = shape[0] / on_parcel.shape[0]
        scale_x = shape[1] / on_parcel.shape[1]
        on_parcel = zoom(on_parcel.astype(float), (scale_y, scale_x), order=0) > 0.5

    # Find flood zones
    flood_zones = []
    sev_used = set()
    for sev_name, threshold in sorted(FLOOD_THRESHOLDS.items(),
                                       key=lambda x: -x[1]):  # major first
        flood_mask = max_depth > threshold
        if on_parcel is not None:
            flood_mask = flood_mask & on_parcel

        labeled, num = ndimage.label(flood_mask)
        for i in range(1, num + 1):
            region = labeled == i
            n_px = int(np.sum(region))
            area = n_px * pixel_area
            if area < 5.0:
                continue

            rows, cols = np.where(region)
            cx = gt[2] + np.mean(cols) * gt[0]
            cy = gt[5] + np.mean(rows) * gt[4]

            loc_key = (round(cx / 5) * 5, round(cy / 5) * 5)
            if loc_key in sev_used:
                continue
            sev_used.add(loc_key)

            region_d = max_depth[region]
            # Find first scenario where this floods
            first = max_scenario
            for inc in sorted(depths.keys()):
                if np.any(depths[inc][region] > threshold):
                    first = inc
                    break

            flood_zones.append(FloodZone(
                id=len(flood_zones) + 1,
                scenario_inches=max_scenario,
                centroid_x=cx, centroid_y=cy,
                area_sqm=area,
                max_depth_m=float(np.max(region_d)),
                mean_depth_m=float(np.mean(region_d)),
                volume_m3=float(np.sum(region_d) * pixel_area),
                severity=sev_name,
                first_appears_at_inches=first
            ))

    if verbose:
        by_sev = {}
        for fz in flood_zones:
            by_sev[fz.severity] = by_sev.get(fz.severity, 0) + 1
        print(f"    Flood zones: {len(flood_zones)}")
        for s in ['major', 'moderate', 'minor', 'nuisance']:
            if s in by_sev:
                print(f"      {s}: {by_sev[s]}")

    # Find chokepoints (rapid depth escalation)
    if verbose:
        print("\n  Identifying chokepoints...")

    sorted_scenarios = sorted(depths.keys())
    chokepoints = []

    if len(sorted_scenarios) >= 2:
        first_d = depths[sorted_scenarios[0]]
        last_d = depths[sorted_scenarios[-1]]
        rain_range = sorted_scenarios[-1] - sorted_scenarios[0]

        if rain_range > 0:
            escalation = (last_d - first_d) / rain_range
        else:
            escalation = np.zeros_like(first_d)

        cp_mask = (last_d > 0.05) & (escalation > 0.02) & np.isfinite(escalation)
        if on_parcel is not None and on_parcel.shape == cp_mask.shape:
            cp_mask = cp_mask & on_parcel

        labeled_cp, num_cp = ndimage.label(cp_mask)
        for i in range(1, num_cp + 1):
            region = labeled_cp == i
            n_px = int(np.sum(region))
            area = n_px * pixel_area
            if area < 3.0:
                continue

            rows, cols = np.where(region)
            cx = gt[2] + np.mean(cols) * gt[0]
            cy = gt[5] + np.mean(rows) * gt[4]

            depth_by_sc = {}
            for inc in sorted_scenarios:
                depth_by_sc[inc] = float(np.max(depths[inc][region]))

            esc = float(np.max(escalation[region]))
            max_d = depth_by_sc.get(2.0, depth_by_sc.get(sorted_scenarios[-1], 0))

            severity = 'high' if max_d > 0.30 else 'medium' if max_d > 0.15 else 'low'

            chokepoints.append(Chokepoint(
                id=len(chokepoints) + 1,
                centroid_x=cx, centroid_y=cy,
                description=f"Water backs up to {max_d:.2f}m ({max_d*39.37:.0f}\") "
                           f"at 2\" rain, escalating {esc:.3f}m/inch",
                depth_at_05in=depth_by_sc.get(0.5, 0),
                depth_at_10in=depth_by_sc.get(1.0, 0),
                depth_at_15in=depth_by_sc.get(1.5, 0),
                depth_at_20in=depth_by_sc.get(2.0, 0),
                escalation_rate=esc,
                severity=severity
            ))

        sev_order = {'high': 0, 'medium': 1, 'low': 2}
        chokepoints.sort(key=lambda c: (sev_order[c.severity], -c.escalation_rate))
        chokepoints = chokepoints[:30]
        for i, cp in enumerate(chokepoints):
            cp.id = i + 1

    if verbose:
        print(f"    Chokepoints: {len(chokepoints)}")
        for cp in chokepoints[:5]:
            print(f"      #{cp.id} [{cp.severity}] {cp.description}")

    return flood_zones, chokepoints


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ANUGA 2D storm simulation for drainage analysis'
    )
    parser.add_argument('--dsm', required=True, help='Path to DSM GeoTIFF')
    parser.add_argument('--boundary', help='Path to boundary shapefile/GeoJSON')
    parser.add_argument('--parcels', help='Path to parcels GeoPackage (auto-creates boundary)')
    parser.add_argument('--parcel-mask', help='Path to parcel mask raster for filtering')
    parser.add_argument('--config', '-c', help='Community config JSON')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--resolution', '-r', type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument('--duration', type=float, default=SIM_DURATION_S,
                        help=f'Simulation duration in seconds (default: {SIM_DURATION_S})')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    config = {}
    if args.config:
        p = Path(args.config)
        if p.exists():
            with open(p) as f:
                config = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("ANUGA Storm Simulation")
        print(f"{'='*60}")

    # Load terrain
    elevation, transform, bounds, crs = load_terrain(
        Path(args.dsm), args.resolution, verbose)

    # Load or create boundary
    if args.boundary:
        boundary_coords = load_boundary(Path(args.boundary), verbose)
    elif args.parcels:
        boundary_coords = load_boundary(Path(args.parcels), verbose)
    else:
        # Use valid data extent as boundary
        valid = np.isfinite(elevation)
        rows, cols = np.where(valid)
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        x0 = bounds.left + min_c * args.resolution
        x1 = bounds.left + max_c * args.resolution
        y0 = bounds.bottom + (elevation.shape[0] - max_r) * args.resolution
        y1 = bounds.bottom + (elevation.shape[0] - min_r) * args.resolution
        boundary_coords = [(x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0)]
        if verbose:
            print(f"  Using data extent as boundary")

    # Change to output dir for ANUGA temp files
    orig_dir = os.getcwd()
    os.chdir(str(output_dir))

    try:
        # Create domain
        domain = create_anuga_domain(
            elevation, transform, bounds, boundary_coords,
            args.resolution,
            config.get('hydrology', {}).get('mannings_n', DEFAULT_MANNINGS_N),
            verbose
        )

        # Run each scenario
        soil = config.get('soil', {})
        infil = soil.get('infiltration_mm_hr', DEFAULT_INFILTRATION_MM_HR)
        scenarios = config.get('hydrology', {}).get('rainfall_scenarios', RAINFALL_SCENARIOS)

        depth_rasters = {}
        scenario_results = []

        for scenario in scenarios:
            inches = scenario['inches']
            label = scenario['label']

            depth = run_scenario(domain, inches, infil, args.duration, verbose)

            # Rasterize to GeoTIFF
            raster_path = output_dir / f"anuga_depth_{inches:.1f}in.tif"
            depth_to_raster(domain, depth, bounds, args.resolution, raster_path, crs)
            depth_rasters[inches] = raster_path

            scenario_results.append({
                'label': label,
                'inches': inches,
                'depth_raster': str(raster_path),
                'max_depth_m': float(np.max(depth)),
                'mean_depth_m': float(np.mean(depth)),
                'cells_flooded': int(np.sum(depth > 0.025))
            })

        # Analyze
        parcel_mask = None
        if args.parcel_mask:
            parcel_mask = args.parcel_mask
        else:
            # Try to find parcel mask in output dir
            for name in ['parcel_mask_1m.tif', 'parcel_mask.tif']:
                p = output_dir / name
                if p.exists():
                    parcel_mask = str(p)
                    break

        flood_zones, chokepoints = analyze_results(depth_rasters, parcel_mask, verbose)

    finally:
        os.chdir(orig_dir)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'solver': 'ANUGA',
        'solver_version': anuga.__version__,
        'input_dsm': args.dsm,
        'resolution_m': args.resolution,
        'sim_duration_s': args.duration,
        'simulation': {'scenarios': scenario_results},
        'flood_zones': [asdict(fz) for fz in flood_zones],
        'chokepoints': [asdict(cp) for cp in chokepoints],
        'statistics': {
            'total_flood_zones': len(flood_zones),
            'total_chokepoints': len(chokepoints),
            'by_severity': {s: len([fz for fz in flood_zones if fz.severity == s])
                            for s in ['major', 'moderate', 'minor', 'nuisance']},
            'chokepoints_by_severity': {s: len([cp for cp in chokepoints if cp.severity == s])
                                        for s in ['high', 'medium', 'low']},
            'wall_time_s': time.time() - t_start
        },
        'depth_rasters': {str(k): str(v) for k, v in depth_rasters.items()}
    }

    # Save as storm_results.json (same format as r.sim.water output)
    results_path = output_dir / "storm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        total_time = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE ({total_time:.0f}s total)")
        print(f"{'='*60}")
        print(f"Flood zones: {len(flood_zones)}")
        for s in ['major', 'moderate', 'minor', 'nuisance']:
            c = len([fz for fz in flood_zones if fz.severity == s])
            if c:
                print(f"  {s}: {c}")
        print(f"Chokepoints: {len(chokepoints)}")
        print(f"Results: {results_path}")


if __name__ == '__main__':
    main()
