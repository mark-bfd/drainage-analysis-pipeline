#!/usr/bin/env python3
"""
04_generate_report.py - Report Generation with Claude API
Drone Drainage Analysis Pipeline

Generates professional drainage assessment reports from analysis data.
Uses Claude API for narrative generation.

Usage:
    python 04_generate_report.py --data ./output --community "Sample Lakeside" --output report.pdf
    python 04_generate_report.py --data ./output --config config/sample_site.json

Requires: anthropic, weasyprint or reportlab
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add scripts dir to path for report_maps import
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime
from typing import Dict, List, Optional, Any
from string import Template
import base64

# Optional imports
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic package not installed. Narrative generation disabled.")

try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except (ImportError, OSError):
    HAS_WEASYPRINT = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# =============================================================================
# Report Templates
# =============================================================================

EXECUTIVE_SUMMARY_PROMPT = """You are writing a professional drainage assessment report for {community_name}.

Analysis data:
{analysis_json}

Write the Executive Summary section (approximately 300-400 words). Include:
- Total area assessed and data collection method (drone photogrammetry)
- Number of problem areas identified, broken down by severity
- Top 3-5 most critical findings with specific descriptions
- Key recommendation for next steps
- Brief mention of TWDB FIF funding eligibility if applicable

Tone: Professional, data-driven, written for a city council or HOA board audience.
Do not include engineering design recommendations (pipe sizes, materials, etc.).
Focus on assessment findings and planning-level recommendations.

Format: Plain text paragraphs, no markdown headers."""

PROBLEM_AREAS_PROMPT = """You are writing the Problem Area Inventory section of a drainage assessment report.

Problem areas identified:
{problems_json}

For each HIGH and MEDIUM severity problem, write a brief 2-3 sentence description that:
- States the type of problem (ponding zone, chokepoint, flat area)
- Describes the location and size
- Explains why it's a concern

Format as a numbered list. Be specific and technical but accessible to non-engineers.
Include all HIGH severity items and the most significant MEDIUM severity items (up to 10 total)."""

RECOMMENDATIONS_PROMPT = """You are writing the Recommendations section of a drainage assessment report.

Analysis summary:
{analysis_json}

Problem areas:
{problems_json}

Known issues from client:
{known_issues}

Write 5-8 prioritized recommendations. For each:
- State the recommended action clearly
- Indicate urgency (immediate, short-term, long-term)
- Note whether engineering follow-up is needed
- Estimate relative cost level if appropriate (low/medium/high)

Important:
- These are planning-level recommendations, not engineering designs
- Do not specify pipe sizes, materials, or construction details
- Focus on what should be studied further or what requires attention
- If relevant, mention potential TWDB FIF grant eligibility

Format: Numbered list with clear action items."""


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {
            size: letter;
            margin: 1in;
            @top-right {
                content: "[Your Company]";
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
            }
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
        }
        h1 {
            color: #1a365d;
            font-size: 24pt;
            border-bottom: 3px solid #2c5282;
            padding-bottom: 10px;
            margin-top: 0;
        }
        h2 {
            color: #2c5282;
            font-size: 16pt;
            margin-top: 30px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }
        h3 {
            color: #2d3748;
            font-size: 13pt;
            margin-top: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header img {
            max-width: 200px;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 14pt;
            color: #666;
            margin-top: 10px;
        }
        .date {
            font-size: 11pt;
            color: #888;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #2c5282;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .severity-high {
            background-color: #fed7d7;
            color: #c53030;
            font-weight: bold;
        }
        .severity-medium {
            background-color: #fefcbf;
            color: #b7791f;
        }
        .severity-low {
            background-color: #c6f6d5;
            color: #276749;
        }
        .stat-box {
            display: inline-block;
            background: #edf2f7;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24pt;
            font-weight: bold;
            color: #2c5282;
        }
        .stat-label {
            font-size: 10pt;
            color: #666;
            text-transform: uppercase;
        }
        .disclaimer {
            background: #fff5f5;
            border-left: 4px solid #c53030;
            padding: 15px;
            margin: 20px 0;
            font-size: 10pt;
        }
        .page-break {
            page-break-before: always;
        }
        .map-container {
            text-align: center;
            margin: 25px 0;
            page-break-inside: avoid;
        }
        .map-container img {
            max-width: 100%;
            border: 1px solid #ccc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 4px;
        }
        .map-caption {
            font-size: 10pt;
            color: #666;
            margin-top: 8px;
            font-style: italic;
        }
        .detail-section {
            margin: 30px 0;
            page-break-inside: avoid;
        }
        .detail-section h3 {
            border-left: 4px solid #2c5282;
            padding-left: 12px;
        }
        ul, ol {
            margin-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Drainage Assessment Report</h1>
        <div class="subtitle">$community_name</div>
        <div class="date">Prepared: $date</div>
        <div class="date">[Your Company]</div>
    </div>

    <h2>Executive Summary</h2>
    $executive_summary

    <div style="text-align: center; margin: 30px 0;">
        <div class="stat-box">
            <div class="stat-value">$total_problems</div>
            <div class="stat-label">Problem Areas</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">$high_severity</div>
            <div class="stat-label">High Severity</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">$depressions</div>
            <div class="stat-label">Ponding Zones</div>
        </div>
    </div>

    $overview_map_section

    <div class="page-break"></div>

    <h2>Methodology</h2>
    <h3>Data Collection</h3>
    <p>Aerial imagery was collected using a DJI Mavic 3 Enterprise drone platform.
    Flight parameters were optimized for drainage assessment with a target ground
    sample distance (GSD) of 2-3 cm/pixel. Image overlap was set to 80% frontal
    and 70% lateral to ensure complete terrain coverage.</p>

    <h3>Processing</h3>
    <p>Raw imagery was processed using Pix4Dmatic photogrammetry software to generate:</p>
    <ul>
        <li>Digital Surface Model (DSM) - includes all surface features</li>
        <li>Digital Terrain Model (DTM) - bare earth elevation</li>
        <li>Orthomosaic - georeferenced aerial photograph</li>
    </ul>

    <h3>Hydrological Analysis</h3>
    <p>The terrain model was analyzed using industry-standard GIS methods:</p>
    <ul>
        <li>Depression filling and sink identification</li>
        <li>D8 flow direction and accumulation modeling</li>
        <li>Stream network extraction at multiple thresholds</li>
        <li>Watershed delineation</li>
        <li>Problem area detection and classification</li>
    </ul>

    <h2>Problem Area Inventory</h2>
    $problem_areas_section

    <h3>Problem Summary by Type</h3>
    <table>
        <tr>
            <th>Type</th>
            <th>Count</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Ponding Zones</td>
            <td>$ponding_count</td>
            <td>Terrain depressions that collect and hold water</td>
        </tr>
        <tr>
            <td>Flow Chokepoints</td>
            <td>$chokepoint_count</td>
            <td>Locations where drainage capacity suddenly decreases</td>
        </tr>
        <tr>
            <td>Flat Areas</td>
            <td>$flat_area_count</td>
            <td>Low-slope areas with poor natural drainage</td>
        </tr>
    </table>

    $detail_maps_section

    <div class="page-break"></div>

    <h2>Recommendations</h2>
    $recommendations_section

    <h2>Limitations</h2>
    <div class="disclaimer">
        <strong>Important Disclaimers:</strong>
        <ul>
            <li>This assessment is based on surface analysis only. Subsurface infrastructure
                (pipes, culverts, inlets) is not visible from aerial imagery.</li>
            <li>This is an assessment-level study, not an engineering design.
                Specific solutions require licensed professional engineer review.</li>
            <li>Elevation accuracy is limited by photogrammetry methods (typically ±2-5 cm with GCPs).</li>
            <li>Soil infiltration rates are estimated from regional databases, not field-tested.</li>
            <li>Field verification of key findings is recommended before action.</li>
        </ul>
    </div>

    <h2>Data & Deliverables</h2>
    <p>The following GIS data products accompany this report:</p>
    <table>
        <tr>
            <th>File</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>dem_filled.tif</td>
            <td>Depression-filled terrain model</td>
        </tr>
        <tr>
            <td>flow_accumulation.tif</td>
            <td>Flow accumulation raster</td>
        </tr>
        <tr>
            <td>depression_depth.tif</td>
            <td>Depression depth map</td>
        </tr>
        <tr>
            <td>slope_percent.tif</td>
            <td>Terrain slope in percent</td>
        </tr>
        <tr>
            <td>streams_*.tif</td>
            <td>Extracted stream networks</td>
        </tr>
        <tr>
            <td>analysis_results.json</td>
            <td>Complete analysis data in JSON format</td>
        </tr>
    </table>

    <div style="margin-top: 50px; padding-top: 20px; border-top: 2px solid #ccc; font-size: 10pt; color: #666;">
        <p><strong>[Your Company]</strong><br>
        Professional Drone Mapping &amp; GIS Analysis<br>
        This report was generated with AI-assisted analysis.</p>
    </div>
</body>
</html>
"""


# =============================================================================
# Claude API Integration
# =============================================================================

def generate_narrative_with_claude(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate narrative text using Claude API."""
    if not HAS_ANTHROPIC:
        return None

    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"  Warning: Claude API failed ({e}), using template text")
        return None


def _fmt_area(sqm: float) -> str:
    """Format area in human-readable units."""
    if sqm > 4046.86:  # > 1 acre
        return f"{sqm/4046.86:.1f} acres"
    return f"{sqm:,.0f} sq ft" if sqm > 10 else f"{sqm:.1f} m²"


def generate_executive_summary(analysis_data: dict, config: dict, api_key: Optional[str] = None) -> str:
    """Generate executive summary — Claude API with template fallback."""
    community_name = config.get('community_name', 'the study area')
    stats = analysis_data.get('statistics', {})
    problems = analysis_data.get('problem_areas', [])
    depressions = analysis_data.get('depressions', [])
    input_stats = stats.get('input', {})
    problem_stats = stats.get('problems', {})

    # Try Claude API first
    if api_key:
        summary_data = {
            'statistics': stats,
            'problem_count': len(problems),
            'high_severity_count': problem_stats.get('high_severity', 0),
            'depression_count': len(depressions),
            'total_depression_volume_m3': sum(d.get('volume_m3', 0) for d in depressions)
        }
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            community_name=community_name,
            analysis_json=json.dumps(summary_data, indent=2)
        )
        result = generate_narrative_with_claude(prompt, api_key)
        if result:
            return result

    # Template-driven fallback
    elev_range = input_stats.get('elevation_range', [0, 0])
    total_volume = sum(d.get('volume_m3', 0) for d in depressions)
    high_count = problem_stats.get('high_severity', 0)
    med_count = problem_stats.get('medium_severity', 0)
    low_count = problem_stats.get('low_severity', 0)
    total_count = problem_stats.get('total', 0)
    by_type = problem_stats.get('by_type', {})

    location = config.get('location', {})
    county = location.get('county', '')
    state = location.get('state', 'Texas')
    lake = location.get('lake', '')

    loc_desc = f"{community_name}"
    if county:
        loc_desc += f", located in {county} County, {state}"
    if lake:
        loc_desc += f" on {lake}"

    known_issues = config.get('known_issues', [])
    known_text = ""
    if known_issues:
        known_text = (
            " These findings are consistent with reported flooding history in the community, "
            "including documented flood damage events."
        )

    grant = config.get('grant_context', {})
    grant_text = ""
    if grant.get('target_program'):
        grant_text = (
            f"\n\nThe findings of this assessment support an application to the "
            f"{grant['target_program']} program for flood mitigation planning and "
            f"infrastructure improvements."
        )

    return (
        f"[Your Company] conducted an AI-enhanced drainage assessment of "
        f"{loc_desc}. High-resolution aerial imagery was captured via drone "
        f"photogrammetry at {input_stats.get('pixel_size_m', 0)*100:.1f} cm ground "
        f"sample distance and processed into a Digital Surface Model covering the "
        f"full study area.\n\n"
        f"The terrain analysis identified a total of {total_count} drainage concern areas: "
        f"{high_count} rated high severity, {med_count} medium severity, and {low_count} "
        f"low severity. These include {by_type.get('ponding', 0)} ponding zones "
        f"(terrain depressions that collect water), {by_type.get('chokepoint', 0)} flow "
        f"chokepoints (locations where drainage capacity decreases sharply), and "
        f"{by_type.get('flat_area', 0)} flat areas with poor natural drainage.\n\n"
        f"A total of {len(depressions)} significant terrain depressions were identified "
        f"with a combined water storage volume of {total_volume:,.0f} cubic meters. "
        f"The elevation across the study area ranges from {elev_range[0]:.1f}m to "
        f"{elev_range[1]:.1f}m, with {input_stats.get('crs_epsg', 'UTM')} coordinate "
        f"reference system used for all analysis.{known_text}{grant_text}"
    )


def generate_problem_descriptions(problem_areas: list, api_key: Optional[str] = None) -> str:
    """Generate problem area descriptions — Claude API with template fallback."""
    significant = [p for p in problem_areas if p.get('severity') in ['high', 'medium']][:15]

    if not significant:
        return "<p>No significant problem areas identified in the study area.</p>"

    # Try Claude API
    if api_key:
        prompt = PROBLEM_AREAS_PROMPT.format(problems_json=json.dumps(significant, indent=2))
        result = generate_narrative_with_claude(prompt, api_key)
        if result:
            lines = result.strip().split('\n')
            items = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line[0].isdigit():
                        line = line.lstrip('0123456789.)-: ')
                    items.append(f"<li>{line}</li>")
            if items:
                return f"<ol>{''.join(items)}</ol>"

    # Template fallback: build detailed problem table
    rows = []
    for i, p in enumerate(significant):
        sev = p.get('severity', 'low')
        sev_class = f"severity-{sev}"
        ptype = p.get('type', 'unknown').replace('_', ' ').title()
        desc = p.get('description', '')
        area = p.get('area_sqm', 0)

        rows.append(
            f"<tr>"
            f"<td>{i+1}</td>"
            f"<td class='{sev_class}'>{sev.upper()}</td>"
            f"<td>{ptype}</td>"
            f"<td>{desc}</td>"
            f"<td>{_fmt_area(area * 10.764)}</td>"  # sqm to sqft
            f"</tr>"
        )

    return (
        "<table>"
        "<tr><th>#</th><th>Severity</th><th>Type</th><th>Description</th><th>Area</th></tr>"
        + ''.join(rows) +
        "</table>"
    )


def generate_recommendations(analysis_data: dict, config: dict, api_key: Optional[str] = None) -> str:
    """Generate recommendations — Claude API with template fallback."""
    problem_areas = analysis_data.get('problem_areas', [])
    known_issues = config.get('known_issues', [])

    # Try Claude API
    if api_key:
        analysis_summary = {
            'total_problems': len(problem_areas),
            'by_severity': {s: len([p for p in problem_areas if p.get('severity') == s])
                           for s in ['high', 'medium', 'low']},
            'by_type': {t: len([p for p in problem_areas if p.get('type') == t])
                       for t in ['ponding', 'chokepoint', 'flat_area']},
            'grant_context': config.get('grant_context', {})
        }
        top_problems = [p for p in problem_areas if p.get('severity') == 'high'][:5]
        prompt = RECOMMENDATIONS_PROMPT.format(
            analysis_json=json.dumps(analysis_summary, indent=2),
            problems_json=json.dumps(top_problems, indent=2),
            known_issues='\n'.join(f"- {i}" for i in known_issues) if known_issues else "None"
        )
        result = generate_narrative_with_claude(prompt, api_key)
        if result:
            lines = result.strip().split('\n')
            items = [f"<li>{l.strip().lstrip('0123456789.)-: ')}</li>"
                     for l in lines if l.strip() and not l.strip().startswith('#') and len(l.strip()) > 5]
            if items:
                return f"<ol>{''.join(items)}</ol>"

    # Template fallback
    has_ponding = any(p.get('type') == 'ponding' for p in problem_areas)
    has_chokepoints = any(p.get('type') == 'chokepoint' for p in problem_areas)
    has_flat = any(p.get('type') == 'flat_area' for p in problem_areas)
    grant = config.get('grant_context', {})

    recs = []
    recs.append(
        "<strong>Field Verification (Immediate):</strong> Conduct on-site verification of the "
        "highest-severity problem areas identified in this assessment. Compare mapped ponding "
        "zones and flow paths against observed drainage patterns during rainfall events."
    )

    if has_ponding:
        recs.append(
            "<strong>Ponding Zone Engineering Study (Short-term):</strong> Commission a "
            "licensed professional engineer to evaluate the identified ponding zones and "
            "design appropriate drainage improvements. This study should include subsurface "
            "infrastructure survey (existing pipes, culverts, inlets) which was not visible "
            "from aerial imagery."
        )

    if has_chokepoints:
        recs.append(
            "<strong>Chokepoint Capacity Analysis (Short-term):</strong> Evaluate identified "
            "flow chokepoints for undersized culverts, blocked channels, or inadequate drainage "
            "infrastructure. These locations represent the highest risk for upstream flooding "
            "during storm events."
        )

    if has_flat:
        recs.append(
            "<strong>Grading Assessment for Flat Areas (Medium-term):</strong> Flat areas with "
            "slopes below 0.5% lack adequate natural drainage. Evaluate whether regrading, "
            "swales, or French drains can improve water conveyance in these zones."
        )

    recs.append(
        "<strong>Storm Scenario Modeling (Short-term):</strong> Expand this assessment with "
        "NOAA Atlas 14 rainfall frequency data and SCS Curve Number modeling to determine "
        "which problem areas activate at different storm return periods (2-year, 10-year, "
        "25-year, 100-year). This data supports infrastructure sizing decisions."
    )

    recs.append(
        "<strong>DTM Generation from Point Cloud (Short-term):</strong> This analysis used "
        "a Digital Surface Model which includes buildings and trees. Generating a bare-earth "
        "Digital Terrain Model from the point cloud data will improve accuracy of flow path "
        "modeling and depression identification."
    )

    if grant.get('target_program'):
        recs.append(
            f"<strong>{grant['target_program']} Grant Application (Medium-term):</strong> "
            f"The findings of this assessment, combined with an engineering study, provide "
            f"the technical documentation required for a {grant['target_program']} application. "
            f"Eligible activities include drainage assessment and flood mitigation planning, "
            f"with grants up to ${grant.get('max_grant_amount', 0):,} "
            f"({grant.get('match_requirement', '50%')} match required)."
        )

    recs.append(
        "<strong>Annual Change Detection (Long-term):</strong> Repeat drone surveys annually "
        "to detect terrain changes, new construction impacts, and drainage pattern shifts. "
        "This creates a time-series dataset for monitoring and maintenance planning."
    )

    return "<ol>" + "".join(f"<li>{r}</li>" for r in recs) + "</ol>"


# =============================================================================
# Report Generation
# =============================================================================

def generate_html_report(analysis_data: dict, config: dict,
                         api_key: Optional[str] = None,
                         ortho_path: Optional[Path] = None,
                         analysis_dir: Optional[Path] = None) -> str:
    """Generate complete HTML report with optional visual maps."""

    community_name = config.get('community_name', 'Study Area')
    problem_areas = analysis_data.get('problem_areas', [])
    stats = analysis_data.get('statistics', {}).get('problems', {})

    # Generate narratives
    print("Generating executive summary...")
    executive_summary = generate_executive_summary(analysis_data, config, api_key)
    executive_summary = f"<p>{executive_summary.replace(chr(10), '</p><p>')}</p>"

    print("Generating problem descriptions...")
    problem_descriptions = generate_problem_descriptions(problem_areas, api_key)

    print("Generating recommendations...")
    recommendations = generate_recommendations(analysis_data, config, api_key)

    # Generate maps if orthomosaic is available
    overview_map_html = ""
    detail_maps_html = ""

    if ortho_path and analysis_dir and ortho_path.exists():
        try:
            # Check if storm simulation data exists
            storm_results_path = analysis_dir / 'storm_results.json'
            has_storm = storm_results_path.exists()

            if has_storm:
                from report_maps import StormMapGenerator
                print("\nGenerating storm simulation maps...")
                storm_gen = StormMapGenerator(analysis_dir, ortho_path, config)
                storm_maps = storm_gen.generate_all_maps()

                fig_num = 1

                # Progressive flood map
                if storm_maps.get('progressive_flood'):
                    overview_map_html = (
                        '<div class="map-container">'
                        f'<img src="{storm_maps["progressive_flood"]}" alt="Progressive Flooding">'
                        f'<div class="map-caption">Figure {fig_num}: Progressive storm flooding '
                        'at 0.5" rainfall increments. Water depth shown from blue (shallow) '
                        'to red (deep). Yellow lines indicate property boundaries.</div>'
                        '</div>'
                    )
                    fig_num += 1

                # Chokepoint overview + details
                sections = []
                if storm_maps.get('chokepoint_overview'):
                    sections.append('<div class="page-break"></div>')
                    sections.append('<h2>Storm Chokepoint Analysis</h2>')
                    sections.append(
                        '<p>Chokepoints are locations where water depth escalates '
                        'rapidly with increasing rainfall. These are areas where '
                        'structures, terrain, or infrastructure cause water to back up '
                        'during storm events.</p>')
                    sections.append(
                        '<div class="map-container">'
                        f'<img src="{storm_maps["chokepoint_overview"]}" '
                        'alt="Chokepoint Overview">'
                        f'<div class="map-caption">Figure {fig_num}: Storm chokepoints '
                        'identified on property land. Left: aerial with markers. '
                        'Right: water depth at maximum rainfall scenario.</div>'
                        '</div>')
                    fig_num += 1

                if storm_maps.get('chokepoint_details'):
                    sections.append('<div class="page-break"></div>')
                    sections.append('<h2>Chokepoint Detail: Depth Progression</h2>')
                    sections.append(
                        '<p>Each chokepoint below shows how water depth increases '
                        'with rainfall intensity. The escalation rate indicates how '
                        'quickly conditions worsen per additional inch of rain.</p>')

                    for det in storm_maps['chokepoint_details']:
                        sev = det.get('severity', 'high')
                        sections.append(
                            f'<div class="detail-section">'
                            f'<h3><span class="severity-{sev}">{sev.upper()}</span> '
                            f'Chokepoint #{det["id"]}</h3>'
                            f'<p>{det.get("description", "")}</p>'
                            f'<div class="map-container">'
                            f'<img src="{det["image"]}" alt="Chokepoint #{det["id"]}">'
                            f'<div class="map-caption">Figure {fig_num}: '
                            f'Chokepoint #{det["id"]} depth progression</div>'
                            f'</div></div>')
                        fig_num += 1

                if sections:
                    detail_maps_html = '\n'.join(sections)

                total_maps = fig_num - 1
                print(f"  Generated {total_maps} storm maps")

            else:
                # Fall back to standard hydro maps
                from report_maps import MapGenerator
                print("\nGenerating visual maps...")
                map_gen = MapGenerator(analysis_dir, ortho_path, config)
                all_maps = map_gen.generate_all_maps()

                if all_maps.get('overview'):
                    overview_map_html = (
                        '<div class="map-container">'
                        f'<img src="{all_maps["overview"]}" alt="Overview Map">'
                        '<div class="map-caption">Figure 1: Study area overview.</div>'
                        '</div>')

                if all_maps.get('detail_maps'):
                    sections = ['<div class="page-break"></div>',
                                '<h2>Detailed Maps</h2>']
                    for i, dm in enumerate(all_maps['detail_maps']):
                        sections.append(
                            f'<div class="detail-section">'
                            f'<div class="map-container">'
                            f'<img src="{dm["image"]}" alt="{dm["title"]}">'
                            f'<div class="map-caption">Figure {i+2}: {dm["title"]}</div>'
                            f'</div></div>')
                    detail_maps_html = '\n'.join(sections)

        except Exception as e:
            print(f"  Warning: Map generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Fill template using Template (safe with CSS braces)
    tmpl = Template(HTML_TEMPLATE)
    html = tmpl.safe_substitute(
        community_name=community_name,
        date=datetime.now().strftime("%B %d, %Y"),
        executive_summary=executive_summary,
        total_problems=stats.get('total', 0),
        high_severity=stats.get('high_severity', 0),
        depressions=len(analysis_data.get('depressions', [])),
        problem_areas_section=problem_descriptions,
        ponding_count=stats.get('by_type', {}).get('ponding', 0),
        chokepoint_count=stats.get('by_type', {}).get('chokepoint', 0),
        flat_area_count=stats.get('by_type', {}).get('flat_area', 0),
        recommendations_section=recommendations,
        overview_map_section=overview_map_html,
        detail_maps_section=detail_maps_html
    )

    return html


def save_report_html(html: str, output_path: Path):
    """Save HTML report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report saved: {output_path}")


def save_report_pdf(html: str, output_path: Path):
    """Convert HTML to PDF."""
    if HAS_WEASYPRINT:
        print("Converting to PDF with WeasyPrint...")
        HTML(string=html).write_pdf(output_path)
        print(f"PDF report saved: {output_path}")
    elif HAS_REPORTLAB:
        print("WeasyPrint not available. PDF generation requires WeasyPrint.")
        print("Install with: pip install weasyprint")
        # Could implement reportlab fallback here
    else:
        print("PDF generation requires weasyprint or reportlab.")
        print("Install with: pip install weasyprint")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate drainage assessment report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python 04_generate_report.py --data ./output --community "Sample Lakeside" --output report.pdf
  python 04_generate_report.py --data ./output --config config/sample_site.json
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to analysis output directory (contains analysis_results.json)')
    parser.add_argument('--config', '-c',
                        help='Path to community config JSON')
    parser.add_argument('--community',
                        help='Community name (if not using config)')
    parser.add_argument('--output', '-o',
                        help='Output file path (default: <data>/report.pdf)')
    parser.add_argument('--format', choices=['pdf', 'html', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--ortho',
                        help='Path to orthomosaic GeoTIFF (enables visual maps)')
    parser.add_argument('--api-key',
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--skip-narrative', action='store_true',
                        help='Skip Claude API narrative generation')

    args = parser.parse_args()

    # Load analysis data
    data_dir = Path(args.data)
    results_path = data_dir / 'analysis_results.json'

    if not results_path.exists():
        print(f"ERROR: Analysis results not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        analysis_data = json.load(f)

    # Load or build config
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

    if args.community:
        config['community_name'] = args.community
    elif 'community_name' not in config:
        config['community_name'] = 'Study Area'

    # Determine output paths
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = data_dir / f"report_{config['community_name'].replace(' ', '_')}"

    # Generate report
    api_key = None if args.skip_narrative else args.api_key

    print(f"\nGenerating report for: {config['community_name']}")
    print(f"{'='*50}\n")

    # Resolve orthomosaic path
    ortho_path = None
    if args.ortho:
        ortho_path = Path(args.ortho)
    elif config.get('pix4d_output'):
        # Try to auto-detect orthomosaic from pix4d output dir
        pix4d_dir = Path(config['pix4d_output'])
        if pix4d_dir.exists():
            # Prefer files with "orthomosaic" in name, then "ortho" but not "dsm"
            ortho_files = list(pix4d_dir.glob('*orthomosaic*.*tif*'))
            if not ortho_files:
                ortho_files = [f for f in pix4d_dir.glob('*ortho*.*tif*')
                               if 'dsm' not in f.name.lower() and 'dtm' not in f.name.lower()]
            if ortho_files:
                ortho_path = ortho_files[0]
                print(f"Auto-detected orthomosaic: {ortho_path.name}")

    if ortho_path and ortho_path.exists():
        print(f"Orthomosaic: {ortho_path}")
        print("Visual maps will be generated.\n")
    else:
        print("No orthomosaic found. Report will be text-only.\n")

    html = generate_html_report(analysis_data, config, api_key,
                                ortho_path=ortho_path,
                                analysis_dir=data_dir)

    # Save outputs
    if args.format in ['html', 'both']:
        html_path = output_base.with_suffix('.html') if not str(output_base).endswith('.html') else output_base
        save_report_html(html, html_path)

    if args.format in ['pdf', 'both']:
        pdf_path = output_base.with_suffix('.pdf') if not str(output_base).endswith('.pdf') else output_base
        save_report_pdf(html, pdf_path)

    print("\nReport generation complete!")


if __name__ == '__main__':
    main()
