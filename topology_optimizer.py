"""Safe topology analysis for PixelForge AI SVG output.

This module detects:
- Overlapping closed polygon paths.
- Shared boundaries between neighboring paths.
- Small gaps between nearby paths.

It does not modify SVG geometry. Automatic repair will be
implemented only after detection is tested.
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
            raise ValueError("SVG contains prohibited active content.")

        for attribute in element.attrib:
            local_name = etree.QName(attribute).localname.lower()

            if local_name.startswith("on") and len(local_name) > 2:
                raise ValueError("SVG contains an event handler.")

    return root


def _has_transform(element: etree._Element) -> bool:
    """Skip paths with transforms until transform handling is supported."""

    current = element

    while current is not None:
        if current.get("transform"):
            return True

        current = current.getparent()

    return False


def _path_to_polygon(path_data: str) -> Polygon | None:
    """Convert a simple closed line-only SVG path into a polygon."""

    if not path_data:
        return None

    path = parse_path(path_data)

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

        points.append((float(start.real), float(start.imag)))

    if len(points) < 3:
        return None

    polygon = Polygon(points)

    if (
        polygon.is_empty
        or polygon.area <= 0
        or not polygon.is_valid
        or not polygon.exterior.is_simple
    ):
        return None

    return polygon


def analyze_svg_topology(
    svg_text: str,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
) -> dict:
    """Analyze polygon relationships without changing the SVG.

    Gap tolerance is measured in SVG coordinate units.

    Only simple, closed, line-only paths without SVG transforms
    are analyzed. Curves, transformed paths, and invalid polygons
    are safely skipped.
    """

    if (
        not isinstance(gap_tolerance, (int, float))
        or not np.isfinite(gap_tolerance)
        or gap_tolerance < 0
        or gap_tolerance > 100
    ):
        raise ValueError("Gap tolerance must be between 0 and 100.")

    root = _parse_svg(svg_text)

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

        try:
            if not path_data:
                skipped_paths += 1
                continue

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
            logger.exception("Topology analysis skipped an SVG path.")

    overlaps = []
    shared_boundaries = []
    nearby_gaps = []

    pair_checks = 0

    for first, second in combinations(polygons, 2):
        pair_checks += 1

        polygon_a = first["polygon"]
        polygon_b = second["polygon"]

        # Avoid expensive geometry operations for distant paths.
        if not polygon_a.envelope.intersects(polygon_b.envelope):
            distance = polygon_a.distance(polygon_b)

            if 0 < distance <= gap_tolerance:
                nearby_gaps.append({
                    "path_a": first["index"],
                    "path_b": second["index"],
                    "distance": round(float(distance), 4),
                })

            continue

        intersection = polygon_a.intersection(polygon_b)

        if intersection.area > MIN_OVERLAP_AREA:
            overlaps.append({
                "path_a": first["index"],
                "path_b": second["index"],
                "overlap_area": round(float(intersection.area), 4),
            })

        shared_length = polygon_a.boundary.intersection(
            polygon_b.boundary
        ).length

        if shared_length > MIN_SHARED_BOUNDARY_LENGTH:
            shared_boundaries.append({
                "path_a": first["index"],
                "path_b": second["index"],
                "shared_length": round(float(shared_length), 4),
            })

        distance = polygon_a.distance(polygon_b)

        if (
            intersection.area <= MIN_OVERLAP_AREA
            and shared_length <= MIN_SHARED_BOUNDARY_LENGTH
            and 0 < distance <= gap_tolerance
        ):
            nearby_gaps.append({
                "path_a": first["index"],
                "path_b": second["index"],
                "distance": round(float(distance), 4),
            })

    return {
        "topology_engine": "shapely",
        "analysis_only": True,
        "geometry_modified": False,
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
        "gap_tolerance": float(gap_tolerance),
        "limitations": [
            "Only closed line-only paths are analyzed.",
            "Paths with SVG transforms are skipped.",
            "Curved paths are skipped.",
            "No automatic geometry repair is performed.",
        ],
    }
