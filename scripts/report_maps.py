#!/usr/bin/env python3
"""
report_maps.py - Map generation for drainage assessment reports.

Generates overview and detail maps from analysis outputs + orthomosaic,
returns base64-encoded PNGs for embedding in HTML reports.

Requires: matplotlib, rasterio, geopandas, numpy, scipy
"""

import io
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import geopandas as gpd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# =============================================================================
# Constants
# =============================================================================

SEVERITY_COLORS = {'high': '#DC2626', 'medium': '#F59E0B', 'low': '#10B981'}
TYPE_MARKERS = {'ponding': 'o', 'chokepoint': '^', 'flat_area': 's'}
TYPE_LABELS = {'ponding': 'Ponding Zone', 'chokepoint': 'Flow Chokepoint', 'flat_area': 'Flat Area'}
STREAM_COLOR = '#00BFFF'
WATERSHED_COLOR = '#FFFFFF'


# =============================================================================
# Utilities
# =============================================================================

def fig_to_base64(fig, dpi=150) -> str:
    """Convert matplotlib figure to base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def read_ortho_overview(ortho_path: Path, max_dim=1800) -> Tuple[np.ndarray, list]:
    """Read full orthomosaic downsampled to fit in max_dim pixels.
    Returns (RGB array H,W,3 uint8, extent [left, right, bottom, top])."""
    with rasterio.open(str(ortho_path)) as src:
        scale = max_dim / max(src.height, src.width)
        out_h = int(src.height * scale)
        out_w = int(src.width * scale)

        # Read 3 RGB bands, downsampled
        bands = min(src.count, 3)
        data = src.read(
            indexes=list(range(1, bands + 1)),
            out_shape=(bands, out_h, out_w),
            resampling=Resampling.average
        )
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    # (bands, H, W) -> (H, W, bands)
    rgb = np.moveaxis(data, 0, -1)

    # Normalize to 0-255 uint8 if needed
    if rgb.dtype != np.uint8:
        vmin, vmax = np.nanpercentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 1)
        if vmax > vmin:
            rgb = np.clip((rgb - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            rgb = np.zeros_like(rgb, dtype=np.uint8)

    return rgb, extent


def read_ortho_window(ortho_path: Path, cx: float, cy: float,
                      buffer_m=50, max_px=800) -> Tuple[np.ndarray, list]:
    """Read a windowed subset of the orthomosaic. Returns (RGB H,W,3 uint8, extent)."""
    with rasterio.open(str(ortho_path)) as src:
        minx = max(cx - buffer_m, src.bounds.left)
        maxx = min(cx + buffer_m, src.bounds.right)
        miny = max(cy - buffer_m, src.bounds.bottom)
        maxy = min(cy + buffer_m, src.bounds.top)

        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        win_h = max(1, int(window.height))
        win_w = max(1, int(window.width))

        if max(win_h, win_w) > max_px:
            s = max_px / max(win_h, win_w)
            out_h, out_w = max(1, int(win_h * s)), max(1, int(win_w * s))
        else:
            out_h, out_w = win_h, win_w

        bands = min(src.count, 3)
        data = src.read(
            indexes=list(range(1, bands + 1)),
            window=window,
            out_shape=(bands, out_h, out_w),
            resampling=Resampling.bilinear
        )
        extent = [minx, maxx, miny, maxy]

    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        vmin, vmax = np.nanpercentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 1)
        if vmax > vmin:
            rgb = np.clip((rgb - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            rgb = np.zeros_like(rgb, dtype=np.uint8)

    return rgb, extent


def load_raster(path: Path) -> Tuple[np.ndarray, 'rasterio.Affine', list]:
    """Load a single-band analysis raster fully into memory.
    Returns (2D array, transform, extent [left, right, bottom, top])."""
    with rasterio.open(str(path)) as src:
        data = src.read(1)
        transform = src.transform
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        nodata = src.nodata

    # Replace nodata with NaN
    if nodata is not None:
        data = data.astype(np.float32)
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan

    return data, transform, extent


def crop_raster(data: np.ndarray, transform, bbox: list) -> Tuple[np.ndarray, list]:
    """Crop an in-memory raster to a geographic bounding box.
    bbox = [minx, maxx, miny, maxy]. Returns (cropped array, extent)."""
    minx, maxx, miny, maxy = bbox

    # Get pixel coords for corners
    row_top, col_left = rowcol(transform, minx, maxy)
    row_bot, col_right = rowcol(transform, maxx, miny)

    # Clamp
    row_top = max(0, int(row_top))
    col_left = max(0, int(col_left))
    row_bot = min(data.shape[0], int(row_bot))
    col_right = min(data.shape[1], int(col_right))

    if row_bot <= row_top or col_right <= col_left:
        return np.full((10, 10), np.nan), bbox

    cropped = data[row_top:row_bot, col_left:col_right]
    return cropped, [minx, maxx, miny, maxy]


def group_chokepoints(chokepoints: List[dict], max_dist_m=40.0) -> List[List[dict]]:
    """Group nearby chokepoints using hierarchical clustering."""
    if len(chokepoints) <= 1:
        return [chokepoints] if chokepoints else []

    from scipy.cluster.hierarchy import fcluster, linkage

    coords = np.array([[cp['centroid_x'], cp['centroid_y']] for cp in chokepoints])
    Z = linkage(coords, method='complete', metric='euclidean')
    labels = fcluster(Z, t=max_dist_m, criterion='distance')

    groups = {}
    for cp, label in zip(chokepoints, labels):
        groups.setdefault(int(label), []).append(cp)

    # Sort groups by max ratio (most severe first)
    group_list = sorted(groups.values(),
                        key=lambda g: max(cp.get('ratio', cp.get('upstream_area_sqm', 0)) for cp in g),
                        reverse=True)
    return group_list


# =============================================================================
# Cartographic Elements
# =============================================================================

def add_scale_bar(ax, length_m=None):
    """Add a scale bar to bottom-left of the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]

    if length_m is None:
        # Choose a nice round number ~20% of map width
        candidates = [10, 20, 25, 50, 100, 200, 250, 500]
        target = span_x * 0.2
        length_m = min(candidates, key=lambda c: abs(c - target))

    x0 = xlim[0] + span_x * 0.05
    y0 = ylim[0] + span_y * 0.04
    bar_h = span_y * 0.008

    # Black bar with white outline
    ax.plot([x0, x0 + length_m], [y0, y0], color='black', linewidth=4, solid_capstyle='butt')
    ax.plot([x0, x0 + length_m], [y0, y0], color='white', linewidth=2, solid_capstyle='butt')
    # Ticks at ends
    for x in [x0, x0 + length_m]:
        ax.plot([x, x], [y0 - bar_h, y0 + bar_h], color='black', linewidth=2)

    ax.text(x0 + length_m / 2, y0 + span_y * 0.015, f'{length_m}m',
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            color='white',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2',
                      edgecolor='none'))


def add_north_arrow(ax):
    """Add a north arrow to top-right of the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[1] - (xlim[1] - xlim[0]) * 0.06
    y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.04
    arrow_len = (ylim[1] - ylim[0]) * 0.06

    ax.annotate('N', xy=(x, y_top), xytext=(x, y_top - arrow_len),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                color='white',
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5),
                bbox=dict(facecolor='black', alpha=0.6,
                          boxstyle='round,pad=0.15', edgecolor='none'))


def add_legend(ax, problem_types=None):
    """Add a map legend."""
    elements = []
    if problem_types is None or 'ponding' in problem_types:
        elements.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=SEVERITY_COLORS['high'],
                               markersize=10, label='Ponding Zone'))
    if problem_types is None or 'chokepoint' in problem_types:
        elements.append(Line2D([0], [0], marker='^', color='w',
                               markerfacecolor=SEVERITY_COLORS['high'],
                               markersize=10, label='Flow Chokepoint'))
    if problem_types is None or 'flat_area' in problem_types:
        elements.append(Line2D([0], [0], marker='s', color='w',
                               markerfacecolor=SEVERITY_COLORS['low'],
                               markersize=8, label='Flat Area'))
    elements.append(Line2D([0], [0], color=STREAM_COLOR, linewidth=2,
                           label='Stream Network'))
    elements.append(Line2D([0], [0], color='#FFD700', linewidth=2,
                           label='Property Lines'))

    ax.legend(handles=elements, loc='lower right', fontsize=7,
              facecolor='white', edgecolor='gray', framealpha=0.9)


# =============================================================================
# Map Generator
# =============================================================================

class MapGenerator:
    """Generates all maps for the drainage assessment report."""

    def __init__(self, analysis_dir: Path, ortho_path: Path, config: dict):
        self.analysis_dir = Path(analysis_dir)
        self.ortho_path = Path(ortho_path)
        self.config = config
        self.community_name = config.get('community_name', 'Study Area')

        # Load analysis results
        results_path = self.analysis_dir / 'analysis_results.json'
        with open(results_path) as f:
            self.results = json.load(f)

        self.problems = self.results.get('problem_areas', [])
        self.depressions = self.results.get('depressions', [])

        # Load analysis rasters (small, fit in memory)
        self.dep_depth, self.dep_tf, self.dep_extent = self._load_if_exists('depression_depth.tif')
        self.flow_acc, self.fa_tf, self.fa_extent = self._load_if_exists('flow_accumulation.tif')
        self.slope, self.slope_tf, self.slope_extent = self._load_if_exists('slope_percent.tif')

        # Load vectors
        self.streams_gdf = self._load_gpkg('stream_network.gpkg')
        self.watersheds_gdf = self._load_gpkg('watersheds.gpkg')
        self.parcels_gdf = self._load_gpkg('parcels_utm.gpkg')

        # Cache overview ortho
        self._overview_rgb = None
        self._overview_extent = None

    def _load_if_exists(self, filename):
        path = self.analysis_dir / filename
        if path.exists():
            return load_raster(path)
        return None, None, None

    def _load_gpkg(self, filename) -> Optional[gpd.GeoDataFrame]:
        path = self.analysis_dir / filename
        if path.exists():
            try:
                return gpd.read_file(str(path))
            except Exception:
                return None
        return None

    def _get_overview_ortho(self):
        if self._overview_rgb is None:
            print("  Loading orthomosaic overview (downsampled)...")
            self._overview_rgb, self._overview_extent = read_ortho_overview(
                self.ortho_path, max_dim=1800
            )
        return self._overview_rgb, self._overview_extent

    # -----------------------------------------------------------------
    # Overview Map
    # -----------------------------------------------------------------

    def generate_overview_map(self) -> str:
        """Generate full study area overview map. Returns base64 data URI."""
        print("  Rendering overview map...")
        rgb, extent = self._get_overview_ortho()

        fig, ax = plt.subplots(1, 1, figsize=(11, 12), dpi=150)

        # 1. Orthomosaic basemap
        ax.imshow(rgb, extent=extent, origin='upper', aspect='equal')

        # 2. Depression depth overlay
        if self.dep_depth is not None:
            dep_display = np.where(self.dep_depth > 0.01, self.dep_depth, np.nan)
            ax.imshow(dep_display, extent=self.dep_extent, origin='upper',
                      cmap='YlOrRd', alpha=0.55, vmin=0, vmax=1.5,
                      aspect='equal')

        # 3. Stream network
        if self.streams_gdf is not None and len(self.streams_gdf) > 0:
            self.streams_gdf.plot(ax=ax, color=STREAM_COLOR, linewidth=0.7, alpha=0.7)

        # 4. Parcel boundaries (property lines)
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax, color='#FFD700', linewidth=0.6,
                                           alpha=0.7)

        # 5. Problem markers
        for p in self.problems:
            ax.scatter(p['centroid_x'], p['centroid_y'],
                       c=SEVERITY_COLORS.get(p['severity'], '#888'),
                       marker=TYPE_MARKERS.get(p['type'], 'o'),
                       s=100 if p['severity'] == 'high' else 60,
                       edgecolors='white', linewidths=1.0, zorder=5)

        # 6. Labels for top problems
        labeled = [p for p in self.problems if p['severity'] == 'high' and p['type'] == 'ponding']
        for p in labeled:
            ax.annotate(f"#{p['id']}",
                        (p['centroid_x'], p['centroid_y']),
                        fontsize=8, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#DC2626',
                                  alpha=0.85, edgecolor='none'),
                        xytext=(12, 12), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                        zorder=6)

        # 7. Cartographic elements
        add_scale_bar(ax)
        add_north_arrow(ax)
        add_legend(ax)

        ax.set_title(f"Drainage Assessment Overview\n{self.community_name}",
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Set extent to data bounds
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        return fig_to_base64(fig)

    # -----------------------------------------------------------------
    # Drainage Overview Maps (when no problems on parcels)
    # -----------------------------------------------------------------

    def _render_drainage_overview(self) -> str:
        """Full-extent flow accumulation + streams + parcels map."""
        print("  Rendering flow accumulation overview...")
        rgb, extent = self._get_overview_ortho()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=150)

        # Left: aerial with parcels + streams
        ax1.imshow(rgb, extent=extent, origin='upper', aspect='equal')
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax1, color='#FFD700', linewidth=0.6, alpha=0.8)
        if self.streams_gdf is not None and len(self.streams_gdf) > 0:
            self.streams_gdf.plot(ax=ax1, color=STREAM_COLOR, linewidth=1.2, alpha=0.8)
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.set_title("Aerial with Property Lines & Streams", fontsize=11, fontweight='bold')
        add_scale_bar(ax1)
        add_north_arrow(ax1)
        ax1.tick_params(labelsize=7)

        # Right: flow accumulation heatmap
        if self.flow_acc is not None:
            fa_display = np.log10(np.clip(self.flow_acc, 1, None))
            fa_display[~np.isfinite(fa_display)] = np.nan
            im = ax2.imshow(fa_display, extent=self.fa_extent, origin='upper',
                           cmap='Blues', aspect='equal',
                           vmin=0, vmax=np.nanpercentile(fa_display, 99))
            plt.colorbar(im, ax=ax2, label='Flow Accumulation (log₁₀ cells)', shrink=0.8)
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax2, color='#FFD700', linewidth=0.5, alpha=0.6)
        if self.streams_gdf is not None and len(self.streams_gdf) > 0:
            self.streams_gdf.plot(ax=ax2, color='cyan', linewidth=0.8, alpha=0.6)
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        ax2.set_title("Flow Accumulation", fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=7)

        fig.suptitle(f"Drainage Network Overview — {self.community_name}",
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig_to_base64(fig)

    def _render_depression_overview(self) -> str:
        """Full-extent depression depth map."""
        print("  Rendering depression depth overview...")
        rgb, extent = self._get_overview_ortho()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=150)

        # Left: aerial with parcels
        ax1.imshow(rgb, extent=extent, origin='upper', aspect='equal')
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax1, color='#FFD700', linewidth=0.6, alpha=0.8)
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.set_title("Aerial with Property Lines", fontsize=11, fontweight='bold')
        add_scale_bar(ax1)
        add_north_arrow(ax1)
        ax1.tick_params(labelsize=7)

        # Right: depression depth
        dep_display = np.where(self.dep_depth > 0.01, self.dep_depth, np.nan)
        im = ax2.imshow(dep_display, extent=self.dep_extent, origin='upper',
                       cmap='YlOrRd', aspect='equal', vmin=0, vmax=1.5)
        plt.colorbar(im, ax=ax2, label='Depression Depth (m)', shrink=0.8)
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax2, color='#FFD700', linewidth=0.5, alpha=0.6)
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        ax2.set_title("Depression Depth", fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=7)

        fig.suptitle(f"Terrain Depression Analysis — {self.community_name}",
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig_to_base64(fig)

    # -----------------------------------------------------------------
    # Detail Maps
    # -----------------------------------------------------------------

    def _render_detail(self, title: str, problems: List[dict],
                       analysis_raster: Optional[np.ndarray],
                       analysis_extent: Optional[list],
                       cmap: str, vmin: float, vmax: float,
                       colorbar_label: str,
                       buffer_m: float = 60,
                       log_scale: bool = False) -> str:
        """Render a dual-panel detail map for a problem area or group."""

        # Compute bounding box from all problems in group
        xs = [p['centroid_x'] for p in problems]
        ys = [p['centroid_y'] for p in problems]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 + buffer_m, buffer_m)
        half_h = max((max(ys) - min(ys)) / 2 + buffer_m, buffer_m)
        buf = max(half_w, half_h)  # Make square

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

        # ---- Left panel: Orthomosaic with markers ----
        try:
            ortho_rgb, ortho_ext = read_ortho_window(self.ortho_path, cx, cy,
                                                      buffer_m=buf, max_px=800)
            ax1.imshow(ortho_rgb, extent=ortho_ext, origin='upper', aspect='equal')
        except Exception:
            ax1.set_facecolor('#f0f0f0')
            ax1.text(0.5, 0.5, 'Orthomosaic\nnot available',
                     transform=ax1.transAxes, ha='center', va='center')

        # Parcel boundaries on left panel
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax1, color='#FFD700', linewidth=0.8, alpha=0.8)

        # Stream network on left panel
        if self.streams_gdf is not None and len(self.streams_gdf) > 0:
            self.streams_gdf.plot(ax=ax1, color=STREAM_COLOR, linewidth=1.0, alpha=0.8)

        # Problem markers on left panel
        for p in problems:
            ax1.scatter(p['centroid_x'], p['centroid_y'],
                        c=SEVERITY_COLORS.get(p['severity'], '#888'),
                        marker=TYPE_MARKERS.get(p['type'], 'o'),
                        s=120, edgecolors='white', linewidths=1.5, zorder=5)
            ax1.annotate(f"#{p['id']}",
                         (p['centroid_x'], p['centroid_y']),
                         fontsize=9, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   fc=SEVERITY_COLORS.get(p['severity'], '#888'),
                                   alpha=0.85, edgecolor='none'),
                         xytext=(10, 10), textcoords='offset points', zorder=6)

        ax1.set_title("Aerial View", fontsize=11, fontweight='bold')
        ax1.set_xlim(cx - buf, cx + buf)
        ax1.set_ylim(cy - buf, cy + buf)
        ax1.tick_params(labelsize=7)
        add_scale_bar(ax1)

        # ---- Right panel: Analysis overlay ----
        if analysis_raster is not None and analysis_extent is not None:
            bbox = [cx - buf, cx + buf, cy - buf, cy + buf]
            cropped, crop_ext = crop_raster(analysis_raster,
                                            self.dep_tf if 'depth' in colorbar_label.lower()
                                            else (self.fa_tf if 'accum' in colorbar_label.lower()
                                                  else self.slope_tf),
                                            bbox)

            display_data = cropped.copy()
            if log_scale:
                display_data = np.log10(np.clip(display_data, 1, None))
                vmin_d = 0
                vmax_d = np.nanpercentile(display_data[np.isfinite(display_data)], 98) if np.any(np.isfinite(display_data)) else 5
            else:
                vmin_d, vmax_d = vmin, vmax

            im = ax2.imshow(display_data, extent=[bbox[0], bbox[1], bbox[2], bbox[3]],
                           origin='upper', cmap=cmap, vmin=vmin_d, vmax=vmax_d,
                           aspect='equal')
            plt.colorbar(im, ax=ax2, label=colorbar_label, shrink=0.8)

            # Stream overlay on right panel too
            if self.streams_gdf is not None and len(self.streams_gdf) > 0:
                self.streams_gdf.plot(ax=ax2, color=STREAM_COLOR, linewidth=0.8, alpha=0.6)

            # Markers on right panel
            for p in problems:
                ax2.scatter(p['centroid_x'], p['centroid_y'],
                           c='white', marker=TYPE_MARKERS.get(p['type'], 'o'),
                           s=80, edgecolors='black', linewidths=1.5, zorder=5)

            ax2.set_xlim(cx - buf, cx + buf)
            ax2.set_ylim(cy - buf, cy + buf)
        else:
            ax2.text(0.5, 0.5, 'Analysis data\nnot available',
                     transform=ax2.transAxes, ha='center', va='center')

        ax2.set_title("Analysis Overlay", fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=7)

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()

        return fig_to_base64(fig)

    def generate_ponding_map(self, depression: dict) -> str:
        """Generate detail map for a ponding zone."""
        title = (f"Ponding Zone #{depression['id']} - "
                 f"Depth: {depression.get('depth_m', 0):.2f}m, "
                 f"Area: {depression.get('area_sqm', 0):.0f} m\u00b2, "
                 f"Volume: {depression.get('volume_m3', 0):.1f} m\u00b3")

        # Find matching problem_area entry
        matching = [p for p in self.problems
                    if p['type'] == 'ponding'
                    and abs(p['centroid_x'] - depression['centroid_x']) < 5
                    and abs(p['centroid_y'] - depression['centroid_y']) < 5]
        if not matching:
            matching = [{'centroid_x': depression['centroid_x'],
                         'centroid_y': depression['centroid_y'],
                         'severity': 'high', 'type': 'ponding',
                         'id': depression['id']}]

        return self._render_detail(
            title=title,
            problems=matching,
            analysis_raster=self.dep_depth,
            analysis_extent=self.dep_extent,
            cmap='YlOrRd',
            vmin=0, vmax=max(1.5, depression.get('depth_m', 1.0) * 1.2),
            colorbar_label='Depression Depth (m)',
            buffer_m=75
        )

    def generate_chokepoint_group_map(self, group: List[dict], group_num: int) -> str:
        """Generate detail map for a group of chokepoints."""
        max_ratio = max(cp.get('upstream_area_sqm', 0) for cp in group)
        title = f"Chokepoint Group {group_num} - {len(group)} Flow Convergence Points"

        return self._render_detail(
            title=title,
            problems=group,
            analysis_raster=self.flow_acc,
            analysis_extent=self.fa_extent,
            cmap='Blues',
            vmin=0, vmax=5,
            colorbar_label='Flow Accumulation (log\u2081\u2080 cells)',
            buffer_m=50,
            log_scale=True
        )

    def generate_flat_area_map(self, flat_problem: dict) -> str:
        """Generate detail map for a flat area."""
        title = (f"Flat Area #{flat_problem['id']} - "
                 f"{flat_problem.get('area_sqm', 0):.0f} m\u00b2, "
                 f"{flat_problem.get('description', '')}")

        return self._render_detail(
            title=title,
            problems=[flat_problem],
            analysis_raster=self.slope,
            analysis_extent=self.slope_extent,
            cmap='RdYlGn',
            vmin=0, vmax=5.0,
            colorbar_label='Slope (%)',
            buffer_m=60
        )

    # -----------------------------------------------------------------
    # Generate All Maps
    # -----------------------------------------------------------------

    def generate_all_maps(self) -> dict:
        """Generate all maps for the report. Returns dict with base64 images.

        Keys:
          'overview' -> base64 string
          'detail_maps' -> list of {type, title, image, problems} dicts, sorted by severity
        """
        maps = {
            'overview': None,
            'detail_maps': []
        }

        # Overview
        maps['overview'] = self.generate_overview_map()

        # If no problems detected, generate drainage overview maps
        if not self.problems:
            print("  No problem areas to map. Generating drainage overview maps...")
            maps['detail_maps'].append({
                'type': 'drainage',
                'severity': 'info',
                'title': 'Flow Accumulation & Stream Network',
                'description': ('Major drainage channels crossing the study area. '
                               'Flow accumulation shows where water concentrates during rainfall. '
                               'Property lines shown in yellow.'),
                'image': self._render_drainage_overview(),
                'problems': []
            })
            if self.dep_depth is not None:
                maps['detail_maps'].append({
                    'type': 'drainage',
                    'severity': 'info',
                    'title': 'Depression Depth Analysis',
                    'description': ('Terrain depressions that collect water. Deeper areas (red) '
                                   'represent potential ponding zones. Note: DSM-based analysis '
                                   'includes building and tree effects on terrain.'),
                    'image': self._render_depression_overview(),
                    'problems': []
                })
            return maps

        # Separate problems by type
        ponding_problems = [p for p in self.problems if p['type'] == 'ponding']
        chokepoint_problems = [p for p in self.problems if p['type'] == 'chokepoint']
        flat_problems = [p for p in self.problems if p['type'] == 'flat_area']

        # --- Ponding zones (high severity first) ---
        ponding_sorted = sorted(ponding_problems,
                                key=lambda p: ({'high': 0, 'medium': 1, 'low': 2}.get(p['severity'], 3),
                                               -p.get('area_sqm', 0)))

        for pp in ponding_sorted:
            # Find matching depression data for stats
            dep_match = None
            for d in self.depressions:
                if (abs(d['centroid_x'] - pp['centroid_x']) < 5 and
                    abs(d['centroid_y'] - pp['centroid_y']) < 5):
                    dep_match = d
                    break

            if dep_match is None:
                dep_match = {'id': pp['id'], 'centroid_x': pp['centroid_x'],
                             'centroid_y': pp['centroid_y'],
                             'depth_m': 0, 'area_sqm': pp.get('area_sqm', 0),
                             'volume_m3': 0}

            print(f"  Rendering ponding zone #{pp['id']}...")
            img = self.generate_ponding_map(dep_match)
            maps['detail_maps'].append({
                'type': 'ponding',
                'severity': pp['severity'],
                'title': f"Ponding Zone #{pp['id']}",
                'description': pp.get('description', ''),
                'image': img,
                'problems': [pp]
            })

        # --- Chokepoint groups ---
        if chokepoint_problems:
            groups = group_chokepoints(chokepoint_problems)
            for i, group in enumerate(groups):
                print(f"  Rendering chokepoint group {i+1}/{len(groups)} "
                      f"({len(group)} points)...")
                img = self.generate_chokepoint_group_map(group, i + 1)
                maps['detail_maps'].append({
                    'type': 'chokepoint',
                    'severity': 'high',
                    'title': f"Chokepoint Group {i+1} ({len(group)} convergence points)",
                    'description': f"Flow convergence area with {len(group)} identified chokepoints",
                    'image': img,
                    'problems': group
                })

        # --- Flat areas ---
        flat_sorted = sorted(flat_problems,
                             key=lambda p: -p.get('area_sqm', 0))

        for fp in flat_sorted:
            print(f"  Rendering flat area #{fp['id']}...")
            img = self.generate_flat_area_map(fp)
            maps['detail_maps'].append({
                'type': 'flat_area',
                'severity': fp['severity'],
                'title': f"Flat Area #{fp['id']}",
                'description': fp.get('description', ''),
                'image': img,
                'problems': [fp]
            })

        # Sort detail maps: high severity first, then by type priority
        type_priority = {'ponding': 0, 'chokepoint': 1, 'flat_area': 2}
        sev_priority = {'high': 0, 'medium': 1, 'low': 2}
        maps['detail_maps'].sort(
            key=lambda m: (sev_priority.get(m['severity'], 3),
                          type_priority.get(m['type'], 3))
        )

        return maps


# =============================================================================
# Storm Map Generator
# =============================================================================

class StormMapGenerator:
    """Generates maps for the storm simulation report."""

    def __init__(self, analysis_dir: Path, ortho_path: Path, config: dict):
        self.analysis_dir = Path(analysis_dir)
        self.ortho_path = Path(ortho_path)
        self.config = config
        self.community_name = config.get('community_name', 'Study Area')

        # Load storm results
        storm_path = self.analysis_dir / 'storm_results.json'
        with open(storm_path) as f:
            self.storm_data = json.load(f)

        self.flood_zones = self.storm_data.get('flood_zones', [])
        self.chokepoints = self.storm_data.get('chokepoints', [])

        # Load depth rasters
        self.depth_rasters = {}
        for scenario in self.storm_data.get('simulation', {}).get('scenarios', []):
            path = Path(scenario.get('depth_raster', ''))
            if path.exists():
                data, tf, extent = load_raster(path)
                data[data < 0] = 0
                self.depth_rasters[scenario['inches']] = (data, extent)

        # Load vectors
        self.parcels_gdf = None
        for name in ['parcels_utm.gpkg']:
            p = self.analysis_dir / name
            if p.exists():
                try:
                    self.parcels_gdf = gpd.read_file(str(p))
                except Exception:
                    pass

        self.streams_gdf = None
        p = self.analysis_dir / 'stream_network.gpkg'
        if p.exists():
            try:
                self.streams_gdf = gpd.read_file(str(p))
            except Exception:
                pass

        self._overview_rgb = None
        self._overview_extent = None

    def _get_overview_ortho(self):
        if self._overview_rgb is None:
            print("  Loading orthomosaic...")
            self._overview_rgb, self._overview_extent = read_ortho_overview(
                self.ortho_path, max_dim=1800)
        return self._overview_rgb, self._overview_extent

    @staticmethod
    def _flood_cmap():
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('flood', [
            (0.0, (0.0, 0.0, 1.0, 0.0)),
            (0.02, (0.2, 0.5, 1.0, 0.4)),
            (0.1, (0.1, 0.4, 0.9, 0.6)),
            (0.3, (0.9, 0.9, 0.0, 0.7)),
            (0.6, (1.0, 0.5, 0.0, 0.8)),
            (1.0, (0.8, 0.0, 0.0, 0.9)),
        ])

    @staticmethod
    def _depth_cmap():
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('depth', [
            (0.0, (0.9, 0.9, 0.9, 1.0)),
            (0.05, (0.7, 0.85, 1.0, 1.0)),
            (0.2, (0.2, 0.5, 1.0, 1.0)),
            (0.5, (1.0, 0.8, 0.0, 1.0)),
            (1.0, (0.8, 0.0, 0.0, 1.0)),
        ])

    def generate_progressive_flood_map(self) -> str:
        """2x2 grid showing water depth at each rainfall increment."""
        print("  Rendering progressive flood map (4 scenarios)...")
        rgb, extent = self._get_overview_ortho()
        scenarios = sorted(self.depth_rasters.keys())[:4]
        if len(scenarios) < 2:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=150)
        axes = axes.flatten()
        global_max = max(np.nanmax(d[0]) for d in self.depth_rasters.values())
        vmax = min(global_max, 1.0)
        cmap = self._flood_cmap()

        for idx, inches in enumerate(scenarios):
            ax = axes[idx]
            depth_data, depth_extent = self.depth_rasters[inches]
            ax.imshow(rgb, extent=extent, origin='upper', aspect='equal')
            if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
                self.parcels_gdf.boundary.plot(ax=ax, color='#FFD700', linewidth=0.5, alpha=0.6)
            depth_display = np.where(depth_data > 0.005, depth_data, np.nan)
            ax.imshow(depth_display, extent=depth_extent, origin='upper',
                     cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
            if self.streams_gdf is not None and len(self.streams_gdf) > 0:
                self.streams_gdf.plot(ax=ax, color='cyan', linewidth=0.5, alpha=0.4)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(f'{inches:.1f}" Rainfall', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=7)
            if idx == 0:
                add_scale_bar(ax)
                add_north_arrow(ax)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Water Depth (m)', fontsize=11)
        fig.suptitle(
            f'Progressive Storm Flooding \u2014 {self.community_name}\n'
            f'Rainfall simulation at 0.5" increments',
            fontsize=15, fontweight='bold', y=0.98)
        fig.subplots_adjust(right=0.90, hspace=0.15, wspace=0.1)
        return fig_to_base64(fig)

    def generate_chokepoint_map(self) -> str:
        """Overview of all chokepoints with flood depth at max scenario."""
        if not self.chokepoints:
            return ""
        print("  Rendering chokepoint overview...")
        rgb, extent = self._get_overview_ortho()
        max_inches = max(self.depth_rasters.keys()) if self.depth_rasters else None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=150)

        # Left: aerial + chokepoints
        ax1.imshow(rgb, extent=extent, origin='upper', aspect='equal')
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax1, color='#FFD700', linewidth=0.5, alpha=0.6)
        for cp in self.chokepoints:
            d = cp.get('depth_at_20in', 0)
            color = '#DC2626' if d > 0.30 else '#F59E0B' if d > 0.15 else '#10B981'
            sz = 120 if d > 0.30 else 90 if d > 0.15 else 60
            ax1.scatter(cp['centroid_x'], cp['centroid_y'], c=color, marker='^',
                       s=sz, edgecolors='white', linewidths=1.0, zorder=5)
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.set_title('Storm Chokepoints on Properties', fontsize=11, fontweight='bold')
        add_scale_bar(ax1)
        add_north_arrow(ax1)
        ax1.tick_params(labelsize=7)
        legend_els = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#DC2626',
                   markersize=12, label='>12" depth at 2" rain'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#F59E0B',
                   markersize=10, label='6-12" depth'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#10B981',
                   markersize=8, label='<6" depth'),
            Line2D([0], [0], color='#FFD700', linewidth=2, label='Property Lines'),
        ]
        ax1.legend(handles=legend_els, loc='lower right', fontsize=8,
                  facecolor='white', framealpha=0.9)

        # Right: flood depth at max scenario
        if max_inches and max_inches in self.depth_rasters:
            depth_data, depth_extent = self.depth_rasters[max_inches]
            depth_display = np.where(depth_data > 0.005, depth_data, np.nan)
            im = ax2.imshow(depth_display, extent=depth_extent, origin='upper',
                           cmap=self._depth_cmap(), vmin=0, vmax=0.5, aspect='equal')
            plt.colorbar(im, ax=ax2, label=f'Water Depth at {max_inches}" Rain (m)', shrink=0.8)
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=ax2, color='black', linewidth=0.4, alpha=0.5)
        for cp in self.chokepoints[:15]:
            ax2.scatter(cp['centroid_x'], cp['centroid_y'], c='red', marker='^',
                       s=80, edgecolors='white', linewidths=1.0, zorder=5)
            ax2.annotate(f"#{cp['id']}", (cp['centroid_x'], cp['centroid_y']),
                        fontsize=7, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.15', fc='red', alpha=0.8, edgecolor='none'),
                        xytext=(8, 8), textcoords='offset points', zorder=6)
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        ax2.set_title(f'Flood Depth at {max_inches:.1f}" Rainfall', fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=7)

        fig.suptitle(f'Storm Chokepoint Analysis \u2014 {self.community_name}',
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig_to_base64(fig)

    def generate_chokepoint_detail(self, cp: dict) -> str:
        """Detail showing depth progression at a single chokepoint."""
        print(f"  Rendering chokepoint #{cp['id']} detail...")
        cx, cy = cp['centroid_x'], cp['centroid_y']
        buf = 40
        scenarios = sorted(self.depth_rasters.keys())[:4]
        n = len(scenarios)

        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5), dpi=150)
        cmap = self._depth_cmap()
        vmax = max(cp.get('depth_at_20in', 0), cp.get('depth_at_15in', 0), 0.3)

        # Aerial panel
        try:
            ortho_rgb, ortho_ext = read_ortho_window(self.ortho_path, cx, cy,
                                                      buffer_m=buf, max_px=600)
            axes[0].imshow(ortho_rgb, extent=ortho_ext, origin='upper', aspect='equal')
        except Exception:
            axes[0].set_facecolor('#f0f0f0')
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            self.parcels_gdf.boundary.plot(ax=axes[0], color='#FFD700', linewidth=1, alpha=0.8)
        axes[0].scatter(cx, cy, c='red', marker='^', s=120, edgecolors='white',
                       linewidths=2, zorder=5)
        axes[0].set_xlim(cx - buf, cx + buf)
        axes[0].set_ylim(cy - buf, cy + buf)
        axes[0].set_title('Aerial', fontsize=10, fontweight='bold')
        add_scale_bar(axes[0])
        axes[0].tick_params(labelsize=6)

        # Depth panels for each scenario
        for idx, inches in enumerate(scenarios):
            ax = axes[idx + 1]
            if inches in self.depth_rasters:
                depth_data, depth_extent = self.depth_rasters[inches]
                # Manual crop using array indexing
                res = abs(depth_extent[1] - depth_extent[0]) / depth_data.shape[1]
                col_min = max(0, int((cx - buf - depth_extent[0]) / res))
                col_max = min(depth_data.shape[1], int((cx + buf - depth_extent[0]) / res))
                row_min = max(0, int((depth_extent[3] - (cy + buf)) / res))
                row_max = min(depth_data.shape[0], int((depth_extent[3] - (cy - buf)) / res))
                cropped = depth_data[row_min:row_max, col_min:col_max]
                crop_ext = [cx - buf, cx + buf, cy - buf, cy + buf]
                depth_display = np.where(cropped > 0.005, cropped, np.nan)
                ax.imshow(depth_display, extent=crop_ext, origin='upper',
                         cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
            if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
                self.parcels_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.5)
            ax.scatter(cx, cy, c='red', marker='^', s=80, edgecolors='white',
                      linewidths=1.5, zorder=5)
            ax.set_xlim(cx - buf, cx + buf)
            ax.set_ylim(cy - buf, cy + buf)

            # Get depth at this scenario from chokepoint data
            key_map = {0.5: 'depth_at_05in', 1.0: 'depth_at_10in',
                       1.5: 'depth_at_15in', 2.0: 'depth_at_20in'}
            d = cp.get(key_map.get(inches, ''), 0)
            ax.set_title(f'{inches:.1f}" Rain\n{d:.2f}m ({d*39.37:.0f}")',
                        fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=6)

        esc = cp.get('escalation_rate', 0)
        fig.suptitle(f"Chokepoint #{cp['id']} \u2014 Escalation: {esc:.3f}m per inch",
                     fontsize=11, fontweight='bold', y=1.04)
        fig.tight_layout()
        return fig_to_base64(fig)

    def generate_all_maps(self) -> dict:
        """Generate all storm maps."""
        maps = {
            'progressive_flood': None,
            'chokepoint_overview': None,
            'chokepoint_details': []
        }
        if self.depth_rasters:
            maps['progressive_flood'] = self.generate_progressive_flood_map()
        if self.chokepoints:
            maps['chokepoint_overview'] = self.generate_chokepoint_map()
            for cp in self.chokepoints[:10]:
                img = self.generate_chokepoint_detail(cp)
                maps['chokepoint_details'].append({
                    'id': cp['id'],
                    'severity': cp.get('severity', 'high'),
                    'description': cp.get('description', ''),
                    'image': img
                })
        return maps
