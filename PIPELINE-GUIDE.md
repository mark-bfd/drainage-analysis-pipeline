# Drone Drainage Analysis Pipeline
## Setup & Operations Guide (Windows workstation)

This document guides AI-assisted operation of the complete workflow — from raw drone imagery to a delivered drainage assessment report. Every stage is designed for AI-assisted automation.

---

## System Requirements

### Software to Install
```powershell
# Python GIS packages (run in PowerShell)
pip install numpy rasterio gdal scipy shapely fiona geopandas pyproj laspy pdal-python reportlab weasyprint

# Verify QGIS installation (already at C:\Program Files\QGIS 3.40.15\)
# GRASS GIS is bundled inside QGIS on Windows — no separate install needed
# GRASS binaries: C:\Program Files\QGIS 3.40.15\apps\grass\
```

### Environment Variables (add to system PATH)
```
C:\Program Files\QGIS 3.40.15\bin
C:\Program Files\QGIS 3.40.15\apps\grass\bin
C:\Program Files\QGIS 3.40.15\apps\Python312
C:\Program Files\QGIS 3.40.15\apps\Python312\Scripts
```

### Project Structure
```
C:\path\to\drainage-analysis-pipeline\
├── scripts\          # Python analysis scripts
├── config\           # Per-community configuration files
├── templates\        # Report templates and branding assets
├── test-data\        # Sample/test datasets
└── output\           # Generated reports and analysis results
```

### GitHub Repository
- Work from the repository root
- Commit all scripts, templates, and configs to the repo

---

## THE PIPELINE: 5 Stages

---

## Stage 1: Drone Data Collection (Field)

**What happens:** The operator flies the target area with a survey drone (e.g. DJI Mavic 3 Enterprise).

**AI role at this stage:** Pre-flight planning optimization.

**Script to build: `scripts/01_flight_plan.py`**
- Input: community boundary (GeoJSON or lat/lon bounds), target GSD (ground sample distance)
- Output: recommended flight parameters (altitude, overlap, grid pattern)
- Calculate: estimated flight time, number of images, storage needed
- Factor in: FAA Part 107 constraints, terrain elevation variation, no-fly zones

**Key parameters:**
- Target GSD: 2-3 cm/pixel for drainage assessment
- Front overlap: 80%
- Side overlap: 70%
- Flight altitude: calculated from GSD and sensor specs

---

## Stage 2: Photogrammetry Processing (Pix4Dmatic)

**What happens:** Raw drone images are processed into 3D models.

**Software:** Pix4Dmatic

**Required exports from Pix4D (GeoTIFF format, WGS84 or UTM):**

| Output | File | Purpose |
|--------|------|---------|
| Digital Surface Model (DSM) | `*-dsm.tif` | Elevation including buildings/trees |
| Digital Terrain Model (DTM) | `*-dtm.tif` | Bare ground elevation (critical for drainage) |
| Orthomosaic | `*-orthomosaic.tif` | Georeferenced aerial photo |
| Point Cloud | `*-dense_point_cloud.laz` | 3D point data for detailed analysis |
| Quality Report | `*.pdf` | Processing accuracy metrics |

**AI role at this stage:** Quality validation.

**Script to build: `scripts/02_validate_pix4d.py`**
- Input: Pix4D export directory
- Validate: all required files exist, CRS is consistent, resolution meets spec
- Check: DSM/DTM have no-data holes, orthomosaic coverage is complete
- Report: GSD achieved, area covered, coordinate system, file sizes
- Flag: any quality issues before proceeding

**If DSM exists but DTM does not:**
- Generate DTM from DSM using ground classification of the point cloud
- Use PDAL or GRASS `r.fillnulls` to create bare-earth model
- Or use Bulldozer (open-source DTM extraction from DSM): `pip install bulldozer-dtm`

---

## Stage 3: Hydrological Analysis (QGIS/GRASS + Python)

**What happens:** The DSM/DTM is processed through a complete hydrological analysis pipeline.

**This is where AI automation provides the biggest advantage.** A human GIS analyst would run these tools one at a time, manually setting parameters and visually inspecting each step. The AI pipeline runs everything in sequence, processes every pixel, and tests multiple parameter configurations.

### Script to build: `scripts/03_hydro_analysis.py`

**This is the core script. It should:**

#### Step 3a: Terrain Preprocessing
```
Input: DTM (or DSM if no DTM available)
1. Fill sinks/depressions using Wang & Liu algorithm (GRASS r.fill.dir)
   - SAVE the depression map before filling — these are potential ponding areas
2. Generate a "depression inventory":
   - Location (centroid lat/lon)
   - Depth (max depth of depression)
   - Area (sq meters)
   - Volume (cubic meters of water it holds)
   - Whether it connects to an outflow path
3. Smooth/filter noise if needed (but preserve real terrain features)
Output: filled DTM, depression inventory (GeoPackage + JSON)
```

#### Step 3b: Flow Direction & Accumulation
```
Input: filled DTM
1. Compute flow direction (D8 algorithm via GRASS r.watershed)
2. Compute flow accumulation (number of upstream cells draining through each cell)
3. Extract stream network at multiple thresholds:
   - Low threshold (more streams): shows minor drainage paths
   - Medium threshold: shows primary drainage network
   - High threshold: shows only major channels
Output: flow direction raster, flow accumulation raster, stream networks (GeoPackage)
```

#### Step 3c: Watershed Delineation
```
Input: flow direction, flow accumulation
1. Identify pour points (drainage outlets) — where streams exit the study area
2. Delineate watershed/catchment for each pour point
3. Calculate watershed statistics:
   - Area (acres)
   - Average slope
   - Longest flow path
   - Time of concentration (Kirpich or SCS method)
Output: watershed polygons with statistics (GeoPackage + JSON)
```

#### Step 3d: Problem Detection
```
Input: all previous outputs + orthomosaic
1. PONDING ZONES: Depressions exceeding thresholds (>6 inches deep, >500 sq ft area)
2. CHOKEPOINTS: Cells where flow accumulation spikes suddenly (high upstream area converging)
3. FLAT AREAS: Regions with slope < 0.5% and no clear outflow path
4. ROAD CROSSINGS: Where computed flow paths intersect road surfaces
   - Cross-reference flow paths with orthomosaic to identify roads
   - Each crossing is a potential culvert location — flag for field verification
5. FLOW OBSTRUCTIONS: Where flow paths are blocked by structures, berms, or fill
Output: problem areas inventory (GeoPackage + JSON) ranked by severity
```

#### Step 3e: Storm Scenario Modeling
```
Input: watershed data + NOAA Atlas 14 + SSURGO soil data
1. Download NOAA Atlas 14 precipitation frequency estimates for the site
   - Source: https://hdsc.nws.noaa.gov/pfds/
   - Get depths for: 2-yr, 5-yr, 10-yr, 25-yr, 50-yr, 100-yr return periods
   - Durations: 1-hour, 6-hour, 12-hour, 24-hour
2. Download SSURGO soil data for the area
   - Source: https://websoilsurvey.nrcs.usda.gov/
   - Extract hydrologic soil group (A/B/C/D) for curve number estimation
3. For each storm scenario:
   - Compute runoff depth using SCS Curve Number method
   - Calculate peak discharge for each watershed (SCS Unit Hydrograph)
   - Determine which depressions overflow at each return period
   - Map flood extent/depth for each scenario
Output: scenario results (GeoPackage + JSON), flood depth rasters per scenario
```

### Key GRASS GIS Commands Reference
```bash
# These run inside QGIS's GRASS environment or via subprocess
r.fill.dir        # Fill sinks, get flow direction
r.watershed        # Flow accumulation, watershed delineation, stream extraction
r.stream.extract   # Stream network from flow accumulation
r.water.outlet     # Delineate basin from pour point
r.drain            # Trace flow path from a point
r.lake             # Model water level in a depression
r.slope.aspect     # Compute slope and aspect from DEM
v.watershed        # Vector-based watershed tools
```

### QGIS Python (PyQGIS) Integration
```python
# Initialize QGIS in standalone mode (no GUI needed)
from qgis.core import QgsApplication
qgs = QgsApplication([], False)
qgs.initQgis()

# Access GRASS through Processing
import processing
from processing.core.Processing import Processing
Processing.initialize()

# Example: run GRASS r.watershed
result = processing.run("grass7:r.watershed", {
    'elevation': '/path/to/dtm.tif',
    'threshold': 5000,
    'accumulation': '/path/to/output/flow_acc.tif',
    'drainage': '/path/to/output/flow_dir.tif',
    'stream': '/path/to/output/streams.tif',
    'basin': '/path/to/output/basins.tif',
    'GRASS_REGION_PARAMETER': None,
    'GRASS_REGION_CELLSIZE_PARAMETER': 0
})
```

---

## Stage 4: Report Generation (Claude API + Python)

**What happens:** Analysis outputs are assembled into a professional report.

### Script to build: `scripts/04_generate_report.py`

**This script:**
1. Reads all JSON statistics from Stage 3
2. Auto-generates map images from rasters/vectors (matplotlib or QGIS print layouts)
3. Assembles a data package for Claude API
4. Calls Claude API to generate narrative sections
5. Combines narrative + maps + data tables into a branded PDF

### Report Structure:
```
1. EXECUTIVE SUMMARY
   - 1-page overview: area assessed, key findings, top 3-5 recommendations
   - Generated by Claude from the analysis statistics

2. SITE OVERVIEW
   - Location map, area boundaries, terrain summary
   - Auto-generated from orthomosaic + watershed boundaries

3. METHODOLOGY
   - Data collection specs (drone, flight parameters, GSD)
   - Processing tools and parameters used
   - Analysis methods with citations
   - Limitations and disclaimers

4. CURRENT CONDITIONS
   - Orthomosaic overview with annotation
   - Elevation profile and slope analysis
   - Existing visible drainage infrastructure

5. DRAINAGE ANALYSIS
   - Flow direction and accumulation maps
   - Watershed boundaries with statistics table
   - Stream network map at multiple scales

6. PROBLEM AREA INVENTORY
   - Ranked table: location, type, severity, description
   - For each problem area:
     - Map cutout showing the issue
     - Orthomosaic photo of the location
     - Upstream catchment area
     - Estimated flood depth at various return periods
   - Generated by Claude from problem detection JSON

7. STORM SCENARIO ANALYSIS
   - Results for 2-yr, 10-yr, 25-yr, 100-yr events
   - Flood extent maps for each scenario
   - Table: which problems activate at which return period
   - "At a 25-year storm, 12 of 18 identified depressions overflow"

8. RECOMMENDATIONS
   - Prioritized action items (assessment-level, not engineering design)
   - Estimated urgency (immediate, short-term, long-term)
   - Whether each item requires engineering follow-up
   - Relevance to TWDB FIF grant eligibility
   - Generated by Claude from the full analysis context

9. GRANT DOCUMENTATION SUPPORT
   - Summary formatted for TWDB FIF application narrative
   - Technical data tables in grant-required format
   - Before/after potential with cost-benefit framing

10. DATA & APPENDIX
    - Full-resolution maps
    - Complete depression inventory table
    - Watershed statistics
    - Storm scenario detail tables
    - Data sources and accuracy statements
```

### Claude API Integration for Report Writing
```python
import anthropic

client = anthropic.Anthropic()

# Send analysis data to Claude for narrative generation
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": f"""You are writing a professional drainage assessment report for {community_name}.

Analysis data:
{json.dumps(analysis_stats, indent=2)}

Write the Executive Summary section (1 page). Include:
- Total area assessed
- Number of problem areas identified, ranked by severity
- Top 3 most critical findings with specific locations
- Key recommendation
- Relevance to TWDB FIF funding eligibility

Tone: professional, data-driven, written for a city council audience.
Do not include engineering design recommendations (pipe sizes, materials).
Focus on assessment findings and planning-level recommendations."""
    }]
)
```

### PDF Generation
```python
# Use WeasyPrint for HTML→PDF with CSS styling
# Or ReportLab for programmatic PDF construction
# Template in templates/report-template.html with your company branding
```

---

## Stage 5: Delivery & Archival

### Script to build: `scripts/05_package_delivery.py`
- Packages: final PDF report + all GIS deliverables (GeoPackage, GeoTIFF)
- Creates a client-ready ZIP archive
- Uploads to Google Drive shared folder (if API configured)
- Archives project data to network storage

### GIS Deliverables (included with report):
```
deliverables/
├── report/
│   └── Drainage_Assessment_{community}_{date}.pdf
├── gis/
│   ├── orthomosaic.tif
│   ├── dsm.tif
│   ├── dtm.tif
│   ├── flow_accumulation.tif
│   ├── watersheds.gpkg
│   ├── stream_network.gpkg
│   ├── problem_areas.gpkg
│   ├── depression_inventory.gpkg
│   └── flood_scenarios/
│       ├── flood_depth_2yr.tif
│       ├── flood_depth_10yr.tif
│       ├── flood_depth_25yr.tif
│       └── flood_depth_100yr.tif
└── data/
    ├── analysis_summary.json
    ├── problem_areas.csv
    └── watershed_statistics.csv
```

---

## CLI Usage (Target)

Once built, the full pipeline runs as:
```bash
python scripts/run_pipeline.py \
  --input "C:\path\to\pix4d\exports" \
  --community "Sample Lakeside" \
  --config config/sample_site.json \
  --storms 2,10,25,100 \
  --output "C:\path\to\output"
```

Or stage by stage:
```bash
python scripts/02_validate_pix4d.py --input "C:\path\to\pix4d\exports"
python scripts/03_hydro_analysis.py --dtm dtm.tif --ortho orthomosaic.tif --config config/sample_site.json
python scripts/04_generate_report.py --data output/analysis/ --community "Sample Lakeside"
python scripts/05_package_delivery.py --project output/ --archive \\NAS\sample-site
```

---

## Competitive Advantage Summary

| What AI Does Better | Why It Matters |
|---|---|
| Processes every pixel exhaustively | Finds problems humans skip when sampling |
| Runs 4+ storm scenarios in minutes | Humans typically model 1-2 |
| Generates depression inventory automatically | Catalogs every low point, not just obvious ones |
| Produces draft report same-day | Traditional firms take weeks to write |
| Repeatable identical methodology | Defensible, consistent results across communities |
| Annual change detection trivial | New revenue stream impossible for manual analysis |

## Honest Limitations (stated in every report)

- Surface analysis only — subsurface pipes/culverts not visible from aerial data
- Assessment-level, not engineering design — does not replace PE-stamped plans
- Elevation accuracy limited by photogrammetry (typically ±2-5cm with GCPs)
- Soil infiltration estimated from SSURGO database, not field-tested
- Recommendations are for planning and grant documentation purposes
- Field verification of key findings recommended before action

---

## Getting Started: Sample Site Test Case

1. Export DSM + DTM + orthomosaic from Pix4Dmatic as GeoTIFF
2. Place files in `C:\path\to\projects\sample-site\Pix4D_Output\`
3. Install Python packages (see requirements above)
4. Run `scripts/02_validate_pix4d.py` to verify data quality
5. Run `scripts/03_hydro_analysis.py` on the site DTM
6. Compare results to field-verified ground truth (known ponding/flooding locations)
7. Iterate thresholds and parameters until results match reality
8. Generate the report — this becomes the portfolio piece and sales demo
