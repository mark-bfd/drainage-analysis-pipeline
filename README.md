# Drone Drainage Analysis Pipeline

**Part 5 of a 5-part build-sequence series — full-pipeline engineering: geospatial + simulation + automated reporting.**

An end-to-end automated pipeline that turns drone photogrammetry output into a client-ready drainage assessment: validated terrain → hydrological analysis → 2D storm simulation → maps and a narrative report. **Impact:** what took a dedicated engineer two weeks of manual tool-driving runs in two days of monitored processing, with identical repeatable methodology every time.

## What it does

1. **Validate** Pix4D photogrammetry exports (DSM/DTM, orthomosaic, point cloud) — CRS consistency, GSD, coverage, no-data holes. 10-point quality gate before anything downstream runs.
2. **Analyze** terrain hydrology with GRASS GIS (via QGIS's bundled GRASS 8.4): sink filling, D8 flow direction/accumulation, watershed delineation, stream extraction, depression inventory, chokepoint and flat-area detection — with parcel-boundary filtering so detections land on properties, not open water.
3. **Simulate** storm scenarios three ways:
   - **GRASS `r.sim.water`** — physics-based overland flow, CPU-only (~4 min/scenario at 1 m resolution)
   - **ANUGA** — open-source shallow-water solver, fully automated mesh + boundary setup
   - **HEC-RAS 2D** — the industry/grant-standard solver, driven programmatically (see below)
4. **Report** — automated map rendering (progressive flood depth maps, chokepoint detail cutouts, parcel overlays) plus optional LLM-generated narrative sections, assembled into a branded PDF.
5. **Package** — client deliverable folder (PDF + GeoTIFFs + GeoPackages), manifest, ZIP, archive.

## Architecture

```
 Pix4D exports (DSM / DTM / orthomosaic / point cloud)
        │
        ▼
 ┌─────────────────────┐
 │ 02_validate_pix4d   │  quality gate: CRS, GSD, coverage, holes
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 03_hydro_analysis   │  GRASS GIS: flow, watersheds, depressions,
 │                     │  chokepoints + parcel-mask filtering
 └─────────┬───────────┘
           ▼
 ┌─────────────────────────────────────────────────┐
 │ Storm simulation (three interchangeable paths)  │
 │  03b storm_simulation   GRASS r.sim.water (CPU) │
 │  03d anuga_storm        ANUGA shallow-water     │
 │  03a + 03c HEC-RAS 2D   programmatic mesh + prj │
 └─────────┬───────────────────────────────────────┘
           ▼
 ┌─────────────────────┐     ┌─────────────────┐
 │ 04_generate_report  │────▶│ report_maps.py  │
 │ HTML → PDF          │     │ map rendering   │
 └─────────┬───────────┘     └─────────────────┘
           ▼
 ┌─────────────────────┐
 │ 05_package_delivery │  deliverables ZIP + archive
 └─────────────────────┘

 run_pipeline.py orchestrates the full chain.
```

## The interesting hard part: programmatic HEC-RAS 2D mesh generation

HEC-RAS is the solver grant programs actually accept — but its 2D mesh generation is locked behind the RAS Mapper GUI, which makes an automated pipeline impossible out of the box.

`03a_hecras_mesh_gen.py` generates the mesh **without the GUI** by writing the geometry HDF5 (`.g01.hdf`) directly, using a schema reverse-engineered from HEC-RAS 6.6's own example projects: cell centers, face points, face-cell connectivity tables, perimeter identification, and per-cell terrain elevation sampling, all as a regular grid clipped to an arbitrary boundary polygon. In test runs it produced a **313,000-cell mesh in 6.6 seconds** with pure Python (numpy + h5py).

Status caveat, honestly stated: the HDF5 mesh data verifies structurally against HEC-RAS's reference schema; the companion `.g01` plain-text geometry format still has formatting quirks under validation against HEC-RAS 6.6. The GRASS and ANUGA simulation paths are fully working end-to-end.

## Stack

- **Python** (two environments — QGIS's Python 3.12 needs numpy<2, ANUGA needs numpy≥2; `run_with_qgis_python.bat` bootstraps the QGIS one)
- **Pix4Dmatic** — photogrammetry (upstream of this repo)
- **QGIS 3.40 + GRASS GIS 8.4** — terrain and hydrological analysis (`r.watershed`, `r.fill.dir`, `r.sim.water`)
- **ANUGA** — open-source 2D shallow-water simulation
- **HEC-RAS 6.6** — grant-standard 2D hydraulic modeling, driven headlessly (`Ras.exe -c`), GPU via OpenCL
- **GDAL / rasterio / geopandas / shapely / scipy / h5py / rashdf** — geospatial + HDF5 plumbing
- **matplotlib** — automated map rendering; **WeasyPrint** — HTML→PDF reports
- **Anthropic API** (optional) — narrative report sections; falls back to templated text

## Notes on this public copy

- `config/sample_site.json` is **synthetic** — a fictional site with placeholder coordinates and round-number values, provided to document the config schema. It is not a real community or client.
- Client deliverables, site data, and client-specific configuration are excluded. The engineering is all here; the data isn't.

## Series context

This is the capstone of a 5-part build sequence. Parts 1–4 covered the component skills; this repo is where they compose: drone data handling, GIS automation, numerical simulation, format reverse-engineering, and automated document generation in one continuous, hands-off pipeline.

## License

MIT — see [LICENSE](LICENSE).
