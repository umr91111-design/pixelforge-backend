
"""Safe SVG topology analysis and repair planning for PixelForge AI.

This version:
- Detects overlaps between simple closed polygons.
- Detects shared boundaries.
- Detects nearby gaps.
- Produces repair candidates for later geometry optimization.
- Never changes SVG geometry automatically.

Only simple closed line-only paths without transforms are analyzed.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from lxml import etree
from shapely.geometry import Polygon
from svgpathtools import Line, parse_path


logger = logging.getLogger("pixelforge.topology")

MAX_SVG_BYTES = 20 * 1024 * 1024
MAX_PATHS = 500
MAX_SEGMENTS_PER_PATH = 1000
MAX_TOTAL_SEGMENTS = 10000

DEFAULT_GAP_TOLERANCE = 1.5
MIN_OVERLAP_AREA = 0.01
MIN_SHARED_BOUNDARY_LENGTH = 0.01
MIN_POLYGON_AREA = 1e-8


def _parse_svg(svg_text: str) -> etree._Element:
    """Parse SVG safely and reject active content."""

    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG input is empty.")

    if len(svg_text.encode("utf-8")) > MAX_SVG_BYTES:
        raise ValueError("SVG exceeds the allowed size.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )

    try:
        root = etree.fromstring(
            svg_text.encode("utf-8"),
            parser=parser,
        )
    except (etree.XMLSyntaxError, ValueError, UnicodeError) as exc:
        raise ValueError("Invalid SVG XML.") from exc

    if not isinstance(root.tag, str):
        raise ValueError("Invalid SVG root.")

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("Document root is not SVG.")

    if root.getroottree().docinfo.doctype:
        raise ValueError("SVG documents with DOCTYPE are not allowed.")

    blocked = {"script", "foreignobject"}

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        name = etree.QName(element).localname.lower()

        if name in blocked:
            raise ValueError(
                "SVG contains prohibited active content."
            )

        for attribute in element.attrib:
            local_name = etree.QName(attribute).localname.lower()

            if local_name.startswith("on") and len(local_name) > 2:
                raise ValueError(
                    "SVG contains an event handler."
                )

    return root


def _has_transform(element: etree._Element) -> bool:
    """Return True if this element or an ancestor has a transform."""

    current = element

    while current is not None:
        if current.get("transform"):
            return True

        current = current.getparent()

    return False


def _path_to_polygon(path_data: str) -> Polygon | None:
    """Convert a simple closed line-only path into a valid polygon."""

    if not path_data:
        return None

    try:
        path = parse_path(path_data)
    except Exception:
        return None

    if not path or len(path) > MAX_SEGMENTS_PER_PATH:
        return None

    if not path.isclosed():
        return None

    if not all(isinstance(segment, Line) for segment in path):
        return None

    points = []

    for segment in path:
        start = segment.start
        end = segment.end

        coordinates = (
            start.real,
            start.imag,
            end.real,
            end.imag,
        )

        if not all(np.isfinite(value) for value in coordinates):
            return None

        if abs(end - start) <= 1e-9:
            return None

        points.append((
            float(start.real),
            float(start.imag),
        ))

    if len(points) < 3:
        return None

    try:
        polygon = Polygon(points)
    except Exception:
        return None

    if (
        polygon.is_empty
        or polygon.area <= MIN_POLYGON_AREA
        or not polygon.is_valid
        or not polygon.exterior.is_simple
    ):
        return None

    return polygon


def _inspect_paths(root: etree._Element) -> tuple:
    """Collect supported polygons and count skipped paths."""

    polygons = []
    skipped_paths = 0
    inspected_paths = 0
    total_segments = 0

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        if etree.QName(element).localname.lower() != "path":
            continue

        if inspected_paths >= MAX_PATHS:
            skipped_paths += 1
            continue

        inspected_paths += 1

        if _has_transform(element):
            skipped_paths += 1
            continue

        path_data = element.get("d", "")

        if not path_data:
            skipped_paths += 1
            continue

        try:
            parsed = parse_path(path_data)
            segment_count = len(parsed)

            if (
                segment_count == 0
                or segment_count > MAX_SEGMENTS_PER_PATH
                or total_segments + segment_count > MAX_TOTAL_SEGMENTS
            ):
                skipped_paths += 1
                continue

            total_segments += segment_count

            polygon = _path_to_polygon(path_data)

            if polygon is None:
                skipped_paths += 1
                continue

            polygons.append({
                "index": len(polygons),
                "element_id": element.get("id"),
                "polygon": polygon,
            })

        except Exception:
            skipped_paths += 1
            logger.exception(
                "Topology analysis skipped an SVG path."
            )

    return polygons, inspected_paths, skipped_paths


def _path_reference(path_info: dict) -> dict:
    """Return a stable reference to an analyzed path."""

    result = {
        "path_index": path_info["index"],
    }

    if path_info.get("element_id"):
        result["element_id"] = path_info["element_id"]

    return result


def _analyze_polygon_pairs(
    polygons: list,
    gap_tolerance: float,
) -> tuple:
    """Inspect polygon pairs and produce diagnostic repair candidates."""

    overlaps = []
    shared_boundaries = []
    nearby_gaps = []
    repair_candidates = []

    pair_checks = 0

    for first, second in combinations(polygons, 2):
        pair_checks += 1

        polygon_a = first["polygon"]
        polygon_b = second["polygon"]

        ref_a = _path_reference(first)
        ref_b = _path_reference(second)

        # First use bounding boxes to avoid unnecessary intersections.
        envelopes_intersect = (
            polygon_a.envelope.intersects(polygon_b.envelope)
        )

        if envelopes_intersect:
            intersection = polygon_a.intersection(polygon_b)

            overlap_area = float(intersection.area)

            shared_length = float(
                polygon_a.boundary.intersection(
                    polygon_b.boundary
                ).length
            )

        else:
            overlap_area = 0.0
            shared_length = 0.0

        distance = float(polygon_a.distance(polygon_b))

        has_overlap = overlap_area > MIN_OVERLAP_AREA
        has_shared_boundary = (
            shared_length > MIN_SHARED_BOUNDARY_LENGTH
        )

        if has_overlap:
            overlap_info = {
                "path_a": first["index"],
                "path_b": second["index"],
                "overlap_area": round(overlap_area, 4),
            }

            overlaps.append(overlap_info)

            # Overlap is not automatically an error.
            # It may be intentional artwork layering.
            repair_candidates.append({
                "type": "overlap_review",
                "path_a": ref_a,
                "path_b": ref_b,
                "measured_overlap_area": round(
                    overlap_area, 4
                ),
                "automatic_repair_allowed": False,
                "reason": (
                    "Overlapping paths may represent intentional "
                    "layering or separate colored objects."
                ),
            })

        if has_shared_boundary:
            shared_boundaries.append({
                "path_a": first["index"],
                "path_b": second["index"],
                "shared_length": round(shared_length, 4),
            })

        is_nearby_gap = (
            not has_overlap
            and not has_shared_boundary
            and 0 < distance <= gap_tolerance
        )

        if is_nearby_gap:
            gap_info = {
                "path_a": first["index"],
                "path_b": second["index"],
                "distance": round(distance, 4),
            }

            nearby_gaps.append(gap_info)

            # A small gap might be intentional.
            # Do not snap or merge these shapes automatically.
            repair_candidates.append({
                "type": "gap_review",
                "path_a": ref_a,
                "path_b": ref_b,
                "measured_distance": round(distance, 4),
                "automatic_repair_allowed": False,
                "reason": (
                    "Nearby paths may be separate objects. "
                    "Their intended connection is unknown."
                ),
            })

    return (
        overlaps,
        shared_boundaries,
        nearby_gaps,
        repair_candidates,
        pair_checks,
    )


def analyze_svg_topology(
    svg_text: str,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
) -> dict:
    """Analyze SVG topology without modifying its geometry.

    Supported geometry:
    - Simple closed polygons.
    - Straight line segments.
    - Paths without SVG transforms.

    Curves, transformed paths and invalid polygons are skipped.

    The result includes repair candidates for review, but no
    automatic geometry modification is performed.
    """

    if (
        isinstance(gap_tolerance, bool)
        or not isinstance(gap_tolerance, (int, float))
        or not np.isfinite(gap_tolerance)
        or gap_tolerance < 0
        or gap_tolerance > 100
    ):
        raise ValueError(
            "Gap tolerance must be between 0 and 100."
        )

    root = _parse_svg(svg_text)

    (
        polygons,
        inspected_paths,
        skipped_paths,
    ) = _inspect_paths(root)

    (
        overlaps,
        shared_boundaries,
        nearby_gaps,
        repair_candidates,
        pair_checks,
    ) = _analyze_polygon_pairs(
        polygons,
        float(gap_tolerance),
    )

    return {
        "topology_engine": "shapely",
        "analysis_only": True,
        "geometry_modified": False,

        "repair_planning_enabled": True,
        "automatic_repair_enabled": False,

        "paths_found": inspected_paths,
        "valid_polygons": len(polygons),
        "skipped_paths": skipped_paths,
        "pair_checks": pair_checks,

        "overlaps_found": len(overlaps),
        "shared_boundaries_found": len(shared_boundaries),
        "nearby_gaps_found": len(nearby_gaps),

        "overlaps": overlaps,
        "shared_boundaries": shared_boundaries,
        "nearby_gaps": nearby_gaps,

        "repair_candidates_found": len(repair_candidates),
        "repair_candidates": repair_candidates,

        "gap_tolerance": float(gap_tolerance),

        "limitations": [
            "Only closed line-only paths are analyzed.",
            "Paths with SVG transforms are skipped.",
            "Curved paths are skipped.",
            "Complex SVG path relationships are not repaired.",
            "Small gaps may be intentional.",
            "Overlapping paths may be intentional.",
            "No automatic geometry repair is performed.",
        ],
    }
