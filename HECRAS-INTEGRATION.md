# HEC-RAS 2D Integration Guide
## Drone Drainage Analysis Pipeline

### Overview

HEC-RAS 6.6 provides GPU-accelerated 2D shallow water simulation for
physics-based flood modeling. It replaces the GRASS r.sim.water approach
with industry-standard, grant-accepted results.

### Why HEC-RAS over GRASS r.sim.water

| Feature | GRASS r.sim.water | HEC-RAS 2D |
|---------|-------------------|------------|
| GPU acceleration | No (CPU only) | Yes (OpenCL) |
| Time-series output | No (final state only) | Yes (animated) |
| Grant acceptance | Not standard | Industry standard (TWDB, FEMA) |
| Computation speed | ~4 min/scenario at 1m | Seconds/scenario with GPU |
| Results format | GeoTIFF snapshots | HDF5 time-series + GeoTIFF export |
| Visualization | Manual in QGIS | Built-in RAS Mapper + QGIS mesh |

### Installation

**HEC-RAS 6.6:**
- Location: `C:\Program Files (x86)\HEC\HEC-RAS\6.6\`
- Installed via: `HEC-RAS_66_Setup.exe /S` (silent install)
- Download: https://github.com/HydrologicEngineeringCenter/hec-downloads/releases

**Python packages:**
- `rashdf` — Read HEC-RAS HDF5 output (by FEMA FFRD team)
- `h5py` — Direct HDF5 access
- Install: `pip install rashdf h5py`

### GPU Setup

An NVIDIA RTX 3500 Ada (12GB VRAM) supports HEC-RAS GPU via OpenCL:
- OpenCL comes with NVIDIA drivers (no separate SDK needed)
- In HEC-RAS: Plan > Options > 2D Computation Options > select your GPU
- Single precision (float32) on GPU vs double (float64) on CPU
- For residential drainage at 0.5m resolution, GPU is 5-10x faster

### Pipeline Integration

```
Stage 3c: HEC-RAS Storm Simulation
  Input:  terrain_30cm.tif (DSM resampled from 1.7cm)
  Config: sample_site.json (scenarios, soil params)
  Output: Depth/velocity rasters per scenario, time-series animation

Script: 03c_hecras_setup.py
  - Creates project files (.prj, .g01, .p01, .u01)
  - Generates rainfall hyetographs (SCS Type II)
  - Provides GUI setup instructions for one-time mesh creation
  - Runs headless via: Ras.exe -c project.prj
  - Extracts results from HDF5 output
```

### One-Time GUI Setup (Required)

HEC-RAS mesh generation requires the GUI. This is a one-time process per project:

1. Open project: `SampleSite_Drainage.prj`
2. Import terrain in RAS Mapper
3. Draw 2D flow area polygon (land only, exclude lake)
4. Generate computational mesh (0.5m cell size)
5. Add breaklines along roads/buildings
6. Set Manning's n (0.035 default)
7. Configure downstream boundary (normal depth)
8. Select GPU device

After this, all simulation runs are automated via command line.

### Automation (after GUI setup)

```python
# Run simulation headlessly
from subprocess import run
run(["C:/Program Files (x86)/HEC/HEC-RAS/6.6/Ras.exe", "-c", "project.prj"])

# Read results
import rashdf
# or h5py for direct HDF5 access
```

### File Structure

```
sample-site/hecras/
├── SampleSite_Drainage.prj      # Project file
├── SampleSite_Drainage.g01      # Geometry (text)
├── SampleSite_Drainage.g01.hdf  # Geometry (HDF5, after mesh creation)
├── SampleSite_Drainage.p01      # Plan
├── SampleSite_Drainage.p01.hdf  # Results (HDF5, after simulation)
├── SampleSite_Drainage.u01      # Unsteady flow / precipitation
├── Terrain/
│   └── terrain.tif              # 30cm DSM
├── SETUP_INSTRUCTIONS.txt       # Step-by-step GUI setup
└── setup_metadata.json          # Project metadata
```

### Rainfall Scenarios

| Scenario | Total (in) | Duration | Distribution |
|----------|-----------|----------|-------------|
| Light    | 0.5       | 1 hour   | SCS Type II |
| Moderate | 1.0       | 1 hour   | SCS Type II |
| Heavy    | 1.5       | 1 hour   | SCS Type II |
| Severe   | 2.0       | 1 hour   | SCS Type II |

For grant applications, also run:
- 2-year, 10-year, 25-year, 100-year return periods
- Using NOAA Atlas 14 Volume 11 (Texas) precipitation data
- 1-hour, 6-hour, 12-hour, 24-hour durations
