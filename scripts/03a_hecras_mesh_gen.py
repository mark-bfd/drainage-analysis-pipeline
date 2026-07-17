#!/usr/bin/env python3
"""
03a_hecras_mesh_gen.py - Automated HEC-RAS 2D Mesh Generation
Drone Drainage Analysis Pipeline

Generates a complete HEC-RAS 2D geometry (mesh) programmatically,
bypassing the RAS Mapper GUI entirely.

Creates:
  - .g01 text file (geometry definition)
  - .g01.hdf file (mesh topology + terrain elevations)

The mesh is a regular rectangular grid clipped to a boundary polygon,
matching the schema used by HEC-RAS 6.6 RAS Mapper.

Usage:
    python 03a_hecras_mesh_gen.py \
        --terrain terrain.tif \
        --boundary boundary.geojson \
        --project-dir ./hecras \
        --project-name SampleSite \
        --cell-size 1.0 \
        --mannings 0.035
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import h5py

# Optional: rasterio for terrain sampling
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# =============================================================================
# Boundary Loading
# =============================================================================

def load_boundary(path: str) -> List[Tuple[float, float]]:
    """Load boundary polygon from GeoJSON. Returns list of (x, y) tuples."""
    with open(path) as f:
        data = json.load(f)

    geom = data['features'][0]['geometry']
    if geom['type'] == 'MultiPolygon':
        rings = [ring[0] for ring in geom['coordinates']]
        ring = max(rings, key=len)
    elif geom['type'] == 'Polygon':
        ring = geom['coordinates'][0]
    else:
        raise ValueError(f"Unsupported geometry type: {geom['type']}")

    coords = [(c[0], c[1]) for c in ring]
    # Ensure closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def points_in_polygon_vectorized(xs: np.ndarray, ys: np.ndarray,
                                  poly: List[Tuple[float, float]]) -> np.ndarray:
    """Vectorized point-in-polygon for arrays of points."""
    poly_x = np.array([p[0] for p in poly])
    poly_y = np.array([p[1] for p in poly])
    n = len(poly)

    inside = np.zeros(len(xs), dtype=bool)
    j = n - 1
    for i in range(n):
        cond = ((poly_y[i] > ys) != (poly_y[j] > ys)) & \
               (xs < (poly_x[j] - poly_x[i]) * (ys - poly_y[i]) /
                (poly_y[j] - poly_y[i]) + poly_x[i])
        inside = inside ^ cond
        j = i
    return inside


# =============================================================================
# Terrain Sampling
# =============================================================================

def sample_terrain(terrain_path: str, x_coords: np.ndarray,
                   y_coords: np.ndarray) -> np.ndarray:
    """Sample terrain elevation at given coordinates."""
    if not HAS_RASTERIO:
        print("  Warning: rasterio not available, using flat terrain")
        return np.full(len(x_coords), 100.0)

    with rasterio.open(terrain_path) as src:
        # Convert coordinates to pixel indices
        rows, cols = rasterio.transform.rowcol(src.transform, x_coords, y_coords)
        rows = np.clip(np.array(rows), 0, src.height - 1)
        cols = np.clip(np.array(cols), 0, src.width - 1)

        # Read elevation band
        elev = src.read(1)
        z = elev[rows, cols].astype(np.float64)

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        z[~np.isfinite(z)] = np.nan

        # Fill NaN with nearest valid value
        nan_mask = np.isnan(z)
        if np.any(nan_mask) and np.any(~nan_mask):
            from scipy.ndimage import distance_transform_edt
            # Simple: fill with mean of valid values
            z[nan_mask] = np.nanmean(z)

    return z


# =============================================================================
# Mesh Generation
# =============================================================================

def generate_rectangular_mesh(boundary: List[Tuple[float, float]],
                              cell_size: float,
                              verbose: bool = True) -> Dict:
    """
    Generate a regular rectangular mesh clipped to the boundary polygon.

    Returns dict with all mesh topology data:
      - cell_centers: (N, 2) active cell center coordinates
      - facepoint_coords: (M, 2) mesh vertex coordinates
      - cell_facepoint_idx: (N, 8) int32, vertex indices per cell (-1 padded)
      - faces_cell_idx: (F, 2) int32, cell pair per face
      - faces_facepoint_idx: (F, 2) int32, vertex pair per face
      - faces_normal_length: (F, 3) float32, [nx, ny, length]
      - facepoint_is_perimeter: (M,) int32, -1 for perimeter points
      - perimeter_coords: (P, 2) boundary polygon vertices
      - cell_face_info/values: indexed lookup tables
      - facepoint_face_info/values: indexed lookup tables
      - facepoint_cell_info/values: indexed lookup tables
    """
    t0 = time.time()

    # Compute grid extent from boundary
    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    x_min, x_max = min(bx), max(bx)
    y_min, y_max = min(by), max(by)

    # Align to grid
    x_min = np.floor(x_min / cell_size) * cell_size
    y_min = np.floor(y_min / cell_size) * cell_size
    x_max = np.ceil(x_max / cell_size) * cell_size
    y_max = np.ceil(y_max / cell_size) * cell_size

    nx = int((x_max - x_min) / cell_size)
    ny = int((y_max - y_min) / cell_size)

    if verbose:
        print(f"  Grid: {nx} x {ny} = {nx*ny:,} potential cells at {cell_size}m")

    # Generate all cell centers
    cx = x_min + (np.arange(nx) + 0.5) * cell_size
    cy = y_max - (np.arange(ny) + 0.5) * cell_size  # top to bottom
    grid_cx, grid_cy = np.meshgrid(cx, cy)
    all_cx = grid_cx.ravel()
    all_cy = grid_cy.ravel()

    # Test which cells are inside the boundary
    inside = points_in_polygon_vectorized(all_cx, all_cy, boundary)
    n_inside = np.sum(inside)

    if verbose:
        print(f"  Cells inside boundary: {n_inside:,} ({n_inside/(nx*ny)*100:.0f}%)")

    if n_inside == 0:
        raise ValueError("No cells inside boundary polygon")

    # Build index mapping: grid position -> active cell index
    # grid_idx[row, col] = active cell index or -1
    grid_idx = np.full((ny, nx), -1, dtype=np.int32)
    active_mask = inside.reshape(ny, nx)

    cell_count = 0
    for r in range(ny):
        for c in range(nx):
            if active_mask[r, c]:
                grid_idx[r, c] = cell_count
                cell_count += 1

    # Active cell centers
    cell_centers = np.column_stack([all_cx[inside], all_cy[inside]])

    # --- Build face points (grid vertices) ---
    # Each cell has 4 corners. Face points are the unique grid vertices.
    # Vertex (i, j) is the top-left corner of cell (i, j)
    # We need vertices for all active cells plus their neighbors

    # Create vertex grid: (ny+1) x (nx+1)
    vx = x_min + np.arange(nx + 1) * cell_size
    vy = y_max - np.arange(ny + 1) * cell_size  # top to bottom

    # Determine which vertices are needed (adjacent to at least one active cell)
    vertex_needed = np.zeros((ny + 1, nx + 1), dtype=bool)
    for r in range(ny):
        for c in range(nx):
            if active_mask[r, c]:
                vertex_needed[r, c] = True
                vertex_needed[r, c + 1] = True
                vertex_needed[r + 1, c] = True
                vertex_needed[r + 1, c + 1] = True

    # Assign vertex indices
    vertex_idx = np.full((ny + 1, nx + 1), -1, dtype=np.int32)
    fp_count = 0
    fp_coords_list = []
    for r in range(ny + 1):
        for c in range(nx + 1):
            if vertex_needed[r, c]:
                vertex_idx[r, c] = fp_count
                fp_coords_list.append((vx[c], vy[r]))
                fp_count += 1

    facepoint_coords = np.array(fp_coords_list, dtype=np.float64)

    if verbose:
        print(f"  Face points (vertices): {fp_count:,}")

    # --- Cell -> FacePoint mapping ---
    # Each cell (r, c) has corners: TL(r,c), TR(r,c+1), BR(r+1,c+1), BL(r+1,c)
    cell_fp_idx = np.full((cell_count, 8), -1, dtype=np.int32)
    cell_rc = []  # track row, col for each active cell

    idx = 0
    for r in range(ny):
        for c in range(nx):
            if active_mask[r, c]:
                tl = vertex_idx[r, c]
                tr = vertex_idx[r, c + 1]
                br = vertex_idx[r + 1, c + 1]
                bl = vertex_idx[r + 1, c]
                cell_fp_idx[idx, 0] = tl
                cell_fp_idx[idx, 1] = tr
                cell_fp_idx[idx, 2] = br
                cell_fp_idx[idx, 3] = bl
                cell_rc.append((r, c))
                idx += 1

    # --- Build faces (edges between cells) ---
    # Two types: horizontal faces (between vertically adjacent cells)
    #            vertical faces (between horizontally adjacent cells)
    faces_cells = []      # (cell_a, cell_b)
    faces_fps = []        # (facepoint_a, facepoint_b)
    faces_normal_len = [] # (nx, ny, length)

    # Horizontal faces (cell above / cell below, shared top/bottom edge)
    for r in range(ny - 1):
        for c in range(nx):
            top_cell = grid_idx[r, c]
            bot_cell = grid_idx[r + 1, c]
            if top_cell >= 0 and bot_cell >= 0:
                # Face between top and bottom cell
                # Shared edge: vertices (r+1, c) and (r+1, c+1)
                fp_left = vertex_idx[r + 1, c]
                fp_right = vertex_idx[r + 1, c + 1]
                faces_cells.append((top_cell, bot_cell))
                faces_fps.append((fp_left, fp_right))
                # Normal: points from top_cell to bot_cell = (0, -1)
                faces_normal_len.append((0.0, -1.0, cell_size))

    # Vertical faces (cell left / cell right, shared left/right edge)
    for r in range(ny):
        for c in range(nx - 1):
            left_cell = grid_idx[r, c]
            right_cell = grid_idx[r, c + 1]
            if left_cell >= 0 and right_cell >= 0:
                # Shared edge: vertices (r, c+1) and (r+1, c+1)
                fp_top = vertex_idx[r, c + 1]
                fp_bot = vertex_idx[r + 1, c + 1]
                faces_cells.append((left_cell, right_cell))
                faces_fps.append((fp_top, fp_bot))
                # Normal: points from left_cell to right_cell = (1, 0)
                faces_normal_len.append((1.0, 0.0, cell_size))

    n_faces = len(faces_cells)
    faces_cell_idx = np.array(faces_cells, dtype=np.int32) if faces_cells else np.zeros((0, 2), dtype=np.int32)
    faces_facepoint_idx = np.array(faces_fps, dtype=np.int32) if faces_fps else np.zeros((0, 2), dtype=np.int32)
    faces_nvl = np.array(faces_normal_len, dtype=np.float32) if faces_normal_len else np.zeros((0, 3), dtype=np.float32)

    if verbose:
        print(f"  Faces (edges): {n_faces:,}")
        print(f"  Ratio faces/cells: {n_faces/cell_count:.2f}")

    # --- Build indexed lookup tables ---

    # Cell -> Face lookup (Info/Values pattern)
    cell_face_lists = [[] for _ in range(cell_count)]
    for fi in range(n_faces):
        c0, c1 = faces_cell_idx[fi]
        cell_face_lists[c0].append((fi, 1))   # normal points away from c0
        cell_face_lists[c1].append((fi, -1))  # normal points toward c1

    cf_info = np.zeros((cell_count, 2), dtype=np.int32)
    cf_values = []
    offset = 0
    for ci in range(cell_count):
        faces = cell_face_lists[ci]
        cf_info[ci] = [offset, len(faces)]
        for fi, orient in faces:
            cf_values.append((fi, orient))
        offset += len(faces)
    cf_values = np.array(cf_values, dtype=np.int32) if cf_values else np.zeros((0, 2), dtype=np.int32)

    # FacePoint -> Face lookup
    fp_face_lists = [[] for _ in range(fp_count)]
    for fi in range(n_faces):
        fp0, fp1 = faces_facepoint_idx[fi]
        fp_face_lists[fp0].append((fi, 1))
        fp_face_lists[fp1].append((fi, -1))

    fpf_info = np.zeros((fp_count, 2), dtype=np.int32)
    fpf_values = []
    offset = 0
    for fpi in range(fp_count):
        faces = fp_face_lists[fpi]
        fpf_info[fpi] = [offset, len(faces)]
        for fi, orient in faces:
            fpf_values.append((fi, orient))
        offset += len(faces)
    fpf_values = np.array(fpf_values, dtype=np.int32) if fpf_values else np.zeros((0, 2), dtype=np.int32)

    # FacePoint -> Cell lookup
    fp_cell_lists = [set() for _ in range(fp_count)]
    for ci in range(cell_count):
        for k in range(8):
            fpi = cell_fp_idx[ci, k]
            if fpi >= 0:
                fp_cell_lists[fpi].add(ci)

    fpc_info = np.zeros((fp_count, 2), dtype=np.int32)
    fpc_values = []
    offset = 0
    for fpi in range(fp_count):
        cells = sorted(fp_cell_lists[fpi])
        fpc_info[fpi] = [offset, len(cells)]
        fpc_values.extend(cells)
        offset += len(cells)
    fpc_values = np.array(fpc_values, dtype=np.int32) if fpc_values else np.zeros((0,), dtype=np.int32)

    # --- Mark perimeter face points ---
    fp_is_perimeter = np.zeros(fp_count, dtype=np.int32)
    # A face point is on the perimeter if it's a corner of fewer than 4 active cells
    for fpi in range(fp_count):
        if len(fp_cell_lists[fpi]) < 4:
            fp_is_perimeter[fpi] = -1

    # --- Faces perimeter info ---
    # For boundary faces, store perimeter intersection coordinates
    # For simplicity with a regular grid, we mark faces with -1 cell index as boundary
    faces_perim_info = np.zeros((n_faces, 2), dtype=np.int32)
    # All faces connect two active cells, so no boundary faces in interior
    # Boundary faces would have cell index = -1 on one side
    # For a clipped grid, all our faces connect two active cells

    elapsed = time.time() - t0
    if verbose:
        print(f"  Mesh generation: {elapsed:.1f}s")

    return {
        'cell_centers': cell_centers,
        'cell_count': cell_count,
        'facepoint_coords': facepoint_coords,
        'cell_facepoint_idx': cell_fp_idx,
        'faces_cell_idx': faces_cell_idx,
        'faces_facepoint_idx': faces_facepoint_idx,
        'faces_normal_length': faces_nvl,
        'fp_is_perimeter': fp_is_perimeter,
        'cell_face_info': cf_info,
        'cell_face_values': cf_values,
        'facepoint_face_info': fpf_info,
        'facepoint_face_values': fpf_values,
        'facepoint_cell_info': fpc_info,
        'facepoint_cell_values': fpc_values,
        'faces_perimeter_info': faces_perim_info,
        'n_faces': n_faces,
        'n_facepoints': fp_count,
        'cell_size': cell_size,
        'grid_nx': nx,
        'grid_ny': ny,
        'x_min': x_min,
        'y_max': y_max,
    }


# =============================================================================
# HDF5 Writer
# =============================================================================

def write_geometry_hdf(mesh: Dict, boundary: List[Tuple[float, float]],
                       area_name: str, mannings_n: float,
                       output_path: Path, terrain_path: Optional[str] = None,
                       verbose: bool = True):
    """Write the complete HEC-RAS geometry HDF5 file."""

    if verbose:
        print(f"  Writing geometry HDF5: {output_path}")

    cell_centers = mesh['cell_centers']
    cell_count = mesh['cell_count']
    cell_size = mesh['cell_size']

    # Sample terrain if available
    if terrain_path and HAS_RASTERIO:
        if verbose:
            print("  Sampling terrain elevations...")
        cell_elevations = sample_terrain(
            terrain_path, cell_centers[:, 0], cell_centers[:, 1])
        fp_elevations = sample_terrain(
            terrain_path, mesh['facepoint_coords'][:, 0], mesh['facepoint_coords'][:, 1])
    else:
        cell_elevations = np.full(cell_count, 100.0)
        fp_elevations = np.full(mesh['n_facepoints'], 100.0)

    # Compute extents
    x_min = np.min(cell_centers[:, 0]) - cell_size
    x_max = np.max(cell_centers[:, 0]) + cell_size
    y_min = np.min(cell_centers[:, 1]) - cell_size
    y_max = np.max(cell_centers[:, 1]) + cell_size

    boundary_arr = np.array(boundary, dtype=np.float64)

    with h5py.File(str(output_path), 'w') as f:
        # Root attributes
        f.attrs['File Type'] = np.bytes_('HEC-RAS Geometry')
        f.attrs['File Version'] = np.bytes_('HEC-RAS 6.6 September 2024')
        f.attrs['Units System'] = np.bytes_('SI')

        # Geometry group
        geom = f.create_group('Geometry')
        geom.attrs['Extents'] = np.array([x_min, x_max, y_min, y_max], dtype=np.float64)
        geom.attrs['Geometry Time'] = np.bytes_(datetime.now().strftime('%d%b%Y %H:%M:%S'))

        # 2D Flow Areas group
        fa = geom.create_group('2D Flow Areas')

        # Attributes structured array
        attr_dtype = np.dtype([
            ('Name', 'S16'), ('Mann', '<f4'),
            ('Multiple Face Mann n', 'u1'),
            ('Cell Vol Tol', '<f4'), ('Cell Min Area Fraction', '<f4'),
            ('Face Profile Tol', '<f4'), ('Face Area Tol', '<f4'),
            ('Face Conv Ratio', '<f4'), ('Laminar Depth', '<f4'),
            ('Spacing dx', '<f4'), ('Spacing dy', '<f4'),
            ('Shift dx', '<f4'), ('Shift dy', '<f4'),
            ('Cell Count', '<i4')
        ])
        attr_data = np.array([(
            area_name.encode('utf-8')[:16],
            mannings_n, 0,
            0.01, 0.01, 0.01, 0.01, 0.02, 0.2,
            cell_size, cell_size, np.nan, np.nan,
            cell_count
        )], dtype=attr_dtype)
        fa.create_dataset('Attributes', data=attr_data, compression='gzip')

        # Polygon Points (boundary)
        fa.create_dataset('Polygon Points', data=boundary_arr, compression='gzip')

        # Polygon Info: [start_idx, point_count, part_start, part_count]
        fa.create_dataset('Polygon Info',
                          data=np.array([[0, len(boundary), 0, 1]], dtype=np.int32),
                          compression='gzip')

        # Polygon Parts: [start_idx, point_count]
        fa.create_dataset('Polygon Parts',
                          data=np.array([[0, len(boundary)]], dtype=np.int32),
                          compression='gzip')

        # Cell Points (active cell centers)
        fa.create_dataset('Cell Points', data=cell_centers, compression='gzip')

        # Cell Info: [start_idx, cell_count]
        fa.create_dataset('Cell Info',
                          data=np.array([[0, cell_count]], dtype=np.int32),
                          compression='gzip')

        # --- Per-area group ---
        area = fa.create_group(area_name)
        area.attrs['Cell Average Size'] = np.float32(cell_size * cell_size)
        area.attrs['Cell Maximum Index'] = np.int32(cell_count - 1)
        area.attrs['Cell Maximum Size'] = np.float32(cell_size * cell_size)
        area.attrs['Cell Minimum Size'] = np.float32(cell_size * cell_size)
        area.attrs['Data Date'] = np.bytes_(datetime.now().strftime('%d%b%Y %H:%M:%S'))
        area.attrs['Extents'] = np.array([x_min, x_max, y_min, y_max], dtype=np.float64)
        area.attrs['Version'] = np.bytes_('1.0')

        # Cell centers (may include boundary ghost cells, but for regular grid = same as active)
        area.create_dataset('Cells Center Coordinate', data=cell_centers,
                           chunks=(min(4096, cell_count), 2), compression='gzip')

        # Cell -> FacePoint indexes
        area.create_dataset('Cells FacePoint Indexes', data=mesh['cell_facepoint_idx'],
                           chunks=(min(2048, cell_count), 8), compression='gzip')

        # Cell -> Face connectivity (indexed)
        area.create_dataset('Cells Face and Orientation Info', data=mesh['cell_face_info'],
                           chunks=(min(4096, cell_count), 2), compression='gzip')
        area.create_dataset('Cells Face and Orientation Values', data=mesh['cell_face_values'],
                           chunks=(min(8192, len(mesh['cell_face_values'])), 2),
                           compression='gzip')

        # FacePoints (vertices)
        area.create_dataset('FacePoints Coordinate', data=mesh['facepoint_coords'],
                           chunks=(min(4096, mesh['n_facepoints']), 2), compression='gzip')

        # FacePoint -> is perimeter
        area.create_dataset('FacePoints Is Perimeter', data=mesh['fp_is_perimeter'],
                           chunks=(min(16384, mesh['n_facepoints']),), compression='gzip')

        # FacePoint -> Face connectivity (indexed)
        area.create_dataset('FacePoints Face and Orientation Info',
                           data=mesh['facepoint_face_info'],
                           chunks=(min(4096, mesh['n_facepoints']), 2), compression='gzip')
        area.create_dataset('FacePoints Face and Orientation Values',
                           data=mesh['facepoint_face_values'],
                           chunks=(min(8192, len(mesh['facepoint_face_values'])), 2),
                           compression='gzip')

        # FacePoint -> Cell connectivity (indexed)
        area.create_dataset('FacePoints Cell Info', data=mesh['facepoint_cell_info'],
                           chunks=(min(4096, mesh['n_facepoints']), 2), compression='gzip')
        area.create_dataset('FacePoints Cell Index Values',
                           data=mesh['facepoint_cell_values'],
                           chunks=(min(16384, len(mesh['facepoint_cell_values'])),),
                           compression='gzip')

        # Faces (edges between cells)
        n_faces = mesh['n_faces']
        area.create_dataset('Faces Cell Indexes', data=mesh['faces_cell_idx'],
                           chunks=(min(8192, n_faces), 2), compression='gzip')
        area.create_dataset('Faces FacePoint Indexes', data=mesh['faces_facepoint_idx'],
                           chunks=(min(8192, n_faces), 2), compression='gzip')
        area.create_dataset('Faces NormalUnitVector and Length', data=mesh['faces_normal_length'],
                           chunks=(min(5461, n_faces), 3), compression='gzip')

        # Faces perimeter info
        area.create_dataset('Faces Perimeter Info', data=mesh['faces_perimeter_info'],
                           chunks=(min(8192, n_faces), 2), compression='gzip')
        # Empty perimeter values (no boundary faces in a fully clipped grid)
        area.create_dataset('Faces Perimeter Values',
                           data=np.zeros((0, 2), dtype=np.float64))

        # Perimeter (copy of boundary)
        area.create_dataset('Perimeter', data=boundary_arr, compression='gzip')

        # --- Additional required groups ---
        geom.create_group('Land Cover (Manning\'s n)')
        geom['Land Cover (Manning\'s n)'].create_dataset(
            'Calibration Table',
            data=np.zeros((0,), dtype=np.float32))

        geom.create_group('Structures')

    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"  Written: {file_size:.0f} KB")


# =============================================================================
# Text File Writer
# =============================================================================

def write_geometry_text(boundary: List[Tuple[float, float]],
                        area_name: str, cell_count: int,
                        cell_size: float, mannings_n: float,
                        output_path: Path, verbose: bool = True):
    """Write the .g01 text file that references the HDF5 geometry."""

    # Compute centroid
    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    centroid_x = sum(bx) / len(bx)
    centroid_y = sum(by) / len(by)

    lines = [
        f"Geom Title={area_name} 2D Geometry",
        "Program Version=6.60",
        f"Viewing Rectangle= {min(bx):.0f} , {min(by):.0f} , {max(bx):.0f} , {max(by):.0f}",
        "",
        f"Storage Area={area_name:<16s},{centroid_x:.6f},{centroid_y:.6f}",
        "Storage Area Type= 0",
        "Storage Area Is2D=-1",
        f"Storage Area Point Generation Data=,,{cell_size},{cell_size}",
        f"Storage Area 2D Points= {cell_count}",
        f"Storage Area Mannings={mannings_n}",
        "",
        "2D Cell Volume Filter Tolerance=0.01",
        "2D Cell Minimum Area Fraction=0.01",
        "2D Face Profile Filter Tolerance=0.01",
        "2D Face Area Elevation Profile Filter Tolerance=0.01",
        "2D Face Area Elevation Conveyance Ratio=0.02",
        "2D Face Area Laminar Depth=0.2",
        "",
        f"Storage Area Surface Line= {len(boundary)}",
    ]

    # Add boundary coordinates
    for x, y in boundary:
        lines.append(f"   {x:.6f}  ,  {y:.6f}  ,  True")

    lines.extend([
        "",
        "LCMann Time=Dec/30/1899 00:00:00",
        "LCMann Region Time=Dec/30/1899 00:00:00",
        "LCMann Table=0",
        "Chan Stop Cuts=-1",
        "",
        "Use User Specified Reach Order=0",
        "GIS Ratio Cuts To Invert=-1",
        "GIS Limit At Bridges=0",
        "Composite Channel Slope=5",
        "",
        "BEGIN DESCRIPTION:",
        f"2D flow area for {area_name} drainage assessment.",
        "Mesh generated programmatically by drainage-analysis pipeline.",
        f"Cell size: {cell_size}m, Cells: {cell_count}",
        "END DESCRIPTION:",
    ])

    output_path.write_text('\n'.join(lines))

    if verbose:
        print(f"  Written: {output_path}")


# =============================================================================
# Main
# =============================================================================

def generate_hecras_geometry(boundary_path: str, project_dir: str,
                             project_name: str, area_name: str,
                             cell_size: float = 1.0,
                             mannings_n: float = 0.035,
                             terrain_path: Optional[str] = None,
                             verbose: bool = True) -> Dict:
    """Generate complete HEC-RAS 2D geometry files."""

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print("HEC-RAS 2D Mesh Generator")
        print(f"{'='*60}")
        print(f"  Boundary: {boundary_path}")
        print(f"  Cell size: {cell_size}m")
        print(f"  Manning's n: {mannings_n}")
        if terrain_path:
            print(f"  Terrain: {terrain_path}")

    # Load boundary
    boundary = load_boundary(boundary_path)
    if verbose:
        print(f"  Boundary vertices: {len(boundary)}")

    # Simplify boundary if too many vertices (>500 causes slow mesh gen)
    if len(boundary) > 300:
        step = max(1, len(boundary) // 200)
        simplified = boundary[::step]
        if simplified[-1] != boundary[0]:
            simplified.append(boundary[0])
        if verbose:
            print(f"  Simplified to {len(simplified)} vertices")
        boundary = simplified

    # Generate mesh
    if verbose:
        print("\nGenerating mesh...")

    mesh = generate_rectangular_mesh(boundary, cell_size, verbose)

    # Write HDF5
    hdf_path = project_dir / f"{project_name}.g01.hdf"
    write_geometry_hdf(mesh, boundary, area_name, mannings_n,
                       hdf_path, terrain_path, verbose)

    # Write text file
    text_path = project_dir / f"{project_name}.g01"
    write_geometry_text(boundary, area_name, mesh['cell_count'],
                        cell_size, mannings_n, text_path, verbose)

    result = {
        'geometry_hdf': str(hdf_path),
        'geometry_text': str(text_path),
        'cell_count': mesh['cell_count'],
        'face_count': mesh['n_faces'],
        'facepoint_count': mesh['n_facepoints'],
        'cell_size': cell_size,
        'area_name': area_name,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("MESH GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Cells: {mesh['cell_count']:,}")
        print(f"  Faces: {mesh['n_faces']:,}")
        print(f"  Vertices: {mesh['n_facepoints']:,}")
        print(f"  Files: {hdf_path.name}, {text_path.name}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate HEC-RAS 2D mesh geometry (no GUI required)'
    )
    parser.add_argument('--boundary', '-b', required=True,
                        help='Boundary polygon (GeoJSON)')
    parser.add_argument('--terrain', '-t', help='Terrain GeoTIFF for elevation sampling')
    parser.add_argument('--project-dir', '-d', required=True, help='HEC-RAS project directory')
    parser.add_argument('--project-name', '-n', default='Project',
                        help='HEC-RAS project name (default: Project)')
    parser.add_argument('--area-name', '-a', default='FlowArea2D',
                        help='2D flow area name (default: FlowArea2D)')
    parser.add_argument('--cell-size', '-s', type=float, default=1.0,
                        help='Cell size in meters (default: 1.0)')
    parser.add_argument('--mannings', '-m', type=float, default=0.035,
                        help="Manning's n (default: 0.035)")
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    generate_hecras_geometry(
        boundary_path=args.boundary,
        project_dir=args.project_dir,
        project_name=args.project_name,
        area_name=args.area_name,
        cell_size=args.cell_size,
        mannings_n=args.mannings,
        terrain_path=args.terrain,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
