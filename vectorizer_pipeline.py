"""Unified CPU-only analysis and topology-aware SVG refinement pipeline."""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from lxml import etree
from PIL import Image, UnidentifiedImageError
from shapely.geometry import Polygon
from skimage import color, filters, morphology
from svgpathtools import parse_path, Line, CubicBezier, Path

logger = logging.getLogger("pixelforge.pipeline")

MAX_SVG_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 1600
MAX_PATHS_TO_INSPECT = 10000
MAX_PATH_DATA_BYTES = 256 * 1024
MAX_TOTAL_PATH_DATA_BYTES = 5 * 1024 * 1024
MAX_SEGMENTS_PER_PATH = 1000
MAX_TOTAL_SEGMENTS = 10000
MAX_CONTOURS_TO_PROCESS = 5000
MAX_CONTOUR_POINTS = 2000

SHARED_EDGE_PRECISION = 6


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Safely decode an uploaded image."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "Unable to decode the uploaded image."
        ) from exc


def segment_image(image_bytes: bytes) -> dict[str, Any]:
    """Analyse image regions using scikit-image."""
    image = _decode_image(image_bytes)
    array = np.asarray(image)
    gray = color.rgb2gray(array)

    try:
        threshold = float(filters.threshold_otsu(gray))
    except ValueError:
        threshold = 0.5

    mask = gray < threshold

    cleaned = morphology.remove_small_objects(
        mask,
        min_size=12,
        connectivity=2,
    )

    return {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "threshold": threshold,
        "foreground_pixels": int(np.count_nonzero(cleaned)),
        "segmentation_method": "otsu_threshold",
        "analysis_engine": "scikit-image",
    }


def analyze_contour_geometry(
    image_bytes: bytes,
) -> dict[str, Any]:
    """Analyse contours and polygon validity using OpenCV and Shapely."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            "Unable to decode image for contour analysis."
        )

    original_height, original_width = image.shape

    scale = min(
        1.0,
        MAX_IMAGE_SIDE / max(original_height, original_width),
    )

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                max(1, round(original_width * scale)),
                max(1, round(original_height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_polygons = 0
    invalid_polygons = 0
    skipped_contours = 0
    total_area = 0.0

    scale_x = original_width / image.shape[1]
    scale_y = original_height / image.shape[0]

    for contour in contours[:MAX_CONTOURS_TO_PROCESS]:
        if cv2.contourArea(contour) < 20:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, 0.01 * perimeter)

        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        points = simplified.reshape(-1, 2)

        if len(points) < 3:
            continue

        if len(points) > MAX_CONTOUR_POINTS:
            skipped_contours += 1
            continue

        try:
            polygon = Polygon(points)

            if polygon.is_empty or polygon.area <= 0:
                continue

            if polygon.is_valid:
                valid_polygons += 1
                total_area += (
                    float(polygon.area) * scale_x * scale_y
                )
            else:
                invalid_polygons += 1

        except Exception:
            skipped_contours += 1
            logger.exception("Polygon analysis failed.")

    skipped_contours += max(
        0,
        len(contours) - MAX_CONTOURS_TO_PROCESS,
    )

    return {
        "contours_found": len(contours),
        "valid_polygons": valid_polygons,
        "invalid_polygons": invalid_polygons,
        "skipped_contours": skipped_contours,
        "polygon_area_pixels": round(total_area, 2),
        "geometry_engine": "shapely",
        "contour_engine": "opencv",
    }


def _path_has_finite_coordinates(path: Path) -> bool:
    """Check that sampled points along every SVG segment are finite."""
    try:
        if not path:
            return False

        for segment in path:
            for position in (0.0, 0.25, 0.5, 0.75, 1.0):
                point = segment.point(position)

                if not (
                    np.isfinite(point.real)
                    and np.isfinite(point.imag)
                ):
                    return False

        return True

    except Exception:
        logger.exception("SVG path coordinate validation failed.")
        return False


def validate_and_inspect_svg(
    svg_text: str,
) -> dict[str, Any]:
    """Securely parse SVG and inspect a bounded number of paths."""
    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG output is empty.")

    encoded = svg_text.encode("utf-8")

    if len(encoded) > MAX_SVG_BYTES:
        raise ValueError("SVG output exceeds the allowed size.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )

    try:
        root = etree.fromstring(encoded, parser=parser)

        if root.getroottree().docinfo.doctype:
            raise ValueError(
                "SVG documents with DOCTYPE are not allowed."
            )

    except etree.XMLSyntaxError as exc:
        raise ValueError(
            "Generated SVG is not valid XML."
        ) from exc

    if not isinstance(root.tag, str):
        raise ValueError("SVG root element is invalid.")

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("The generated document is not an SVG.")

    svg_namespace = "http://www.w3.org/2000/svg"

    path_count = 0
    inspected_paths = 0
    invalid_paths = 0
    skipped_paths = 0
    total_path_length = 0.0
    total_path_data_bytes = 0
    total_segments = 0

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        if etree.QName(element).localname.lower() != "path":
            continue

        path_count += 1

        if inspected_paths >= MAX_PATHS_TO_INSPECT:
            skipped_paths += 1
            continue

        path_data = element.get("d", "")

        if not path_data:
            invalid_paths += 1
            continue

        path_size = len(path_data.encode("utf-8"))

        if path_size > MAX_PATH_DATA_BYTES:
            skipped_paths += 1
            continue

        if (
            total_path_data_bytes + path_size
            > MAX_TOTAL_PATH_DATA_BYTES
        ):
            skipped_paths += 1
            continue

        try:
            path = parse_path(path_data)

            segment_count = len(path)

            # Check resource limits before coordinate sampling.
            if segment_count > MAX_SEGMENTS_PER_PATH:
                skipped_paths += 1
                continue

            if (
                total_segments + segment_count
                > MAX_TOTAL_SEGMENTS
            ):
                skipped_paths += 1
                continue

            if not _path_has_finite_coordinates(path):
                invalid_paths += 1
                continue

            path_length = float(path.length(error=1e-3))

            if not np.isfinite(path_length):
                invalid_paths += 1
                continue

            total_path_length += path_length
            total_segments += segment_count
            total_path_data_bytes += path_size
            inspected_paths += 1

        except Exception:
            invalid_paths += 1
            logger.exception("SVG path inspection failed.")

    return {
        "valid_svg_xml": True,
        "svg_namespace_present": root.tag.startswith(
            "{" + svg_namespace + "}"
        ),
        "path_count": path_count,
        "inspected_paths": inspected_paths,
        "invalid_paths": invalid_paths,
        "skipped_paths": skipped_paths,
        "total_segments_inspected": total_segments,
        "total_path_length": round(total_path_length, 3),
        "path_inspection": "svgpathtools",
    }


def run_vectorizer_analysis(
    image_bytes: bytes,
    svg_text: str,
) -> dict[str, Any]:
    """Run CPU-based analysis after VTracer generates SVG."""
    return {
        "segmentation": segment_image(image_bytes),
        "geometry": analyze_contour_geometry(image_bytes),
        "svg": validate_and_inspect_svg(svg_text),
        "deep_learning": False,
        "gpu_required": False,
    }


def _shared_edge_key(segment: Line) -> tuple:
    """Create a direction-independent key for a straight edge."""
    start = (
        round(segment.start.real, SHARED_EDGE_PRECISION),
        round(segment.start.imag, SHARED_EDGE_PRECISION),
    )
    end = (
        round(segment.end.real, SHARED_EDGE_PRECISION),
        round(segment.end.imag, SHARED_EDGE_PRECISION),
    )

    return tuple(sorted((start, end)))


def _find_shared_edge_paths(
    root: etree._Element,
) -> set[int]:
    """
    Find closed line-only paths that share an identical
    straight-line edge with another path.

    Applies path, data-size, and segment limits before
    performing shared-edge analysis.
    """
    edge_owners: dict[tuple, set[int]] = {}
    protected_paths: set[int] = set()

    path_index = 0
    inspected_paths = 0
    total_path_data_bytes = 0
    total_segments = 0

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        if etree.QName(element).localname.lower() != "path":
            continue

        current_index = path_index
        path_index += 1

        if inspected_paths >= MAX_PATHS_TO_INSPECT:
            continue

        inspected_paths += 1

        path_data = element.get("d", "")

        if not path_data:
            continue

        path_size = len(path_data.encode("utf-8"))

        if path_size > MAX_PATH_DATA_BYTES:
            continue

        if (
            total_path_data_bytes + path_size
            > MAX_TOTAL_PATH_DATA_BYTES
        ):
            continue

        total_path_data_bytes += path_size

        try:
            path = parse_path(path_data)

            segment_count = len(path)

            # Reject oversized paths before coordinate sampling.
            if segment_count > MAX_SEGMENTS_PER_PATH:
                continue

            if (
                total_segments + segment_count
                > MAX_TOTAL_SEGMENTS
            ):
                continue

            if not _path_has_finite_coordinates(path):
                continue

            total_segments += segment_count

            if not path.isclosed() or not path:
                continue

            if not all(
                isinstance(segment, Line)
                for segment in path
            ):
                continue

            for segment in path:
                edge = _shared_edge_key(segment)

                if edge[0] == edge[1]:
                    continue

                edge_owners.setdefault(
                    edge,
                    set(),
                ).add(current_index)

        except Exception:
            logger.exception(
                "Shared-edge inspection skipped a path."
            )

    for owners in edge_owners.values():
        if len(owners) > 1:
            protected_paths.update(owners)

    return protected_paths


def _is_valid_closed_line_path(path: Path) -> bool:
    """
    Check whether a closed, line-only path forms valid geometry.

    Reject:
    - Open paths
    - Curved paths
    - Duplicate consecutive points
    - Zero-length edges
    - Non-finite coordinates
    - Self-intersecting or otherwise invalid polygons
    """
    if not path or not path.isclosed():
        return False

    if not all(
        isinstance(segment, Line)
        for segment in path
    ):
        return False

    points = [
        (segment.start.real, segment.start.imag)
        for segment in path
    ]

    if len(points) < 3:
        return False

    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            return False

    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]

        if current == following:
            return False

    for segment in path:
        if abs(segment.end - segment.start) <= 1e-9:
            return False

    try:
        polygon = Polygon(points)

        return (
            not polygon.is_empty
            and polygon.area > 0
            and polygon.is_valid
            and polygon.exterior.is_simple
        )

    except Exception:
        logger.exception("Polygon validity check failed.")
        return False


def refine_bezier_paths(
    svg_text: str,
    max_turn_degrees: float = 30.0,
    min_turn_degrees: float = 3.0,
    trim_ratio: float = 0.15,
) -> tuple[str, dict[str, Any]]:
    """
    Conservatively smooth shallow corners in closed,
    line-only SVG paths.

    Invalid polygons and paths sharing an identical full
    straight-line edge are protected from smoothing.

    Existing curves, open paths, and sharp corners
    are preserved.
    """
    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG input is empty.")

    if not (
        0 < min_turn_degrees
        < max_turn_degrees
        < 180
    ):
        raise ValueError("Invalid turn-angle settings.")

    if not 0 < trim_ratio <= 0.25:
        raise ValueError("Invalid trim ratio.")

    encoded = svg_text.encode("utf-8")

    if len(encoded) > MAX_SVG_BYTES:
        raise ValueError("SVG exceeds the allowed size.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )

    try:
        root = etree.fromstring(encoded, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError("SVG is not valid XML.") from exc

    if not isinstance(root.tag, str):
        raise ValueError("SVG root element is invalid.")

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("Document root is not SVG.")

    if root.getroottree().docinfo.doctype:
        raise ValueError(
            "SVG documents with DOCTYPE are not allowed."
        )

    protected_paths = _find_shared_edge_paths(root)

    refined_paths = 0
    refined_corners = 0
    skipped_paths = 0
    protected_path_count = 0
    inspected_paths = 0
    invalid_geometry_paths = 0
    total_path_bytes = 0
    total_segments = 0

    path_index = 0

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        if etree.QName(element).localname.lower() != "path":
            continue

        current_index = path_index
        path_index += 1

        if inspected_paths >= MAX_PATHS_TO_INSPECT:
            skipped_paths += 1
            continue

        inspected_paths += 1

        if current_index in protected_paths:
            protected_path_count += 1
            skipped_paths += 1
            continue

        path_data = element.get("d", "")

        if not path_data:
            skipped_paths += 1
            continue

        path_size = len(path_data.encode("utf-8"))

        if path_size > MAX_PATH_DATA_BYTES:
            skipped_paths += 1
            continue

        if (
            total_path_bytes + path_size
            > MAX_TOTAL_PATH_DATA_BYTES
        ):
            skipped_paths += 1
            continue

        try:
            path = parse_path(path_data)

            segment_count = len(path)

            # Apply segment limits before coordinate sampling.
            if segment_count < 3:
                skipped_paths += 1
                continue

            if segment_count > MAX_SEGMENTS_PER_PATH:
                skipped_paths += 1
                continue

            if (
                total_segments + segment_count
                > MAX_TOTAL_SEGMENTS
            ):
                skipped_paths += 1
                continue

            if not _path_has_finite_coordinates(path):
                invalid_geometry_paths += 1
                skipped_paths += 1
                continue

            total_path_bytes += path_size
            total_segments += segment_count

            if not path.isclosed():
                skipped_paths += 1
                continue

            if not all(
                isinstance(segment, Line)
                for segment in path
            ):
                skipped_paths += 1
                continue

            if not _is_valid_closed_line_path(path):
                invalid_geometry_paths += 1
                skipped_paths += 1
                continue

            points = [
                segment.start
                for segment in path
            ]

            if len(points) < 3:
                skipped_paths += 1
                continue

            count = len(points)
            corners = []

            for index in range(count):
                previous = points[(index - 1) % count]
                current = points[index]
                following = points[(index + 1) % count]

                incoming = current - previous
                outgoing = following - current

                incoming_length = abs(incoming)
                outgoing_length = abs(outgoing)

                if (
                    incoming_length < 1e-6
                    or outgoing_length < 1e-6
                ):
                    corners.append(None)
                    continue

                incoming_unit = incoming / incoming_length
                outgoing_unit = outgoing / outgoing_length

                dot = (
                    incoming_unit.real * outgoing_unit.real
                    + incoming_unit.imag * outgoing_unit.imag
                )

                dot = max(-1.0, min(1.0, dot))

                turn = float(
                    np.degrees(np.arccos(dot))
                )

                if (
                    turn < min_turn_degrees
                    or turn > max_turn_degrees
                ):
                    corners.append(None)
                    continue

                trim = min(
                    incoming_length * trim_ratio,
                    outgoing_length * trim_ratio,
                )

                if trim < 0.5:
                    corners.append(None)
                    continue

                start = current - incoming_unit * trim
                end = current + outgoing_unit * trim

                handle = trim * 0.55

                control1 = start + incoming_unit * handle
                control2 = end - outgoing_unit * handle

                corners.append(
                    (start, end, control1, control2)
                )

            corner_count = sum(
                corner is not None
                for corner in corners
            )

            if corner_count == 0:
                skipped_paths += 1
                continue

            new_segments = []

            first_corner = corners[0]

            if first_corner is not None:
                current_position = first_corner[1]
            else:
                current_position = points[0]

            for step in range(count):
                index = (step + 1) % count
                corner = corners[index]

                if corner is None:
                    target = points[index]

                    if abs(target - current_position) > 1e-8:
                        new_segments.append(
                            Line(current_position, target)
                        )

                    current_position = target
                    continue

                start, end, control1, control2 = corner

                if abs(start - current_position) > 1e-8:
                    new_segments.append(
                        Line(current_position, start)
                    )

                new_segments.append(
                    CubicBezier(
                        start,
                        control1,
                        control2,
                        end,
                    )
                )

                current_position = end

            if not new_segments:
                skipped_paths += 1
                continue

            refined_path = Path(*new_segments)

            if not refined_path.isclosed():
                skipped_paths += 1
                continue

            element.set("d", refined_path.d())

            refined_paths += 1
            refined_corners += corner_count

        except Exception:
            skipped_paths += 1
            logger.exception(
                "Bézier refinement skipped a path."
            )

    result = etree.tostring(
        root,
        encoding="unicode",
        xml_declaration=False,
    )

    validate_and_inspect_svg(result)

    return result, {
        "refinement_engine": "svgpathtools",
        "refined_paths": refined_paths,
        "refined_corners": refined_corners,
        "skipped_paths": skipped_paths,
        "invalid_geometry_paths": invalid_geometry_paths,
        "shared_edge_paths": len(protected_paths),
        "protected_shared_edge_paths": protected_path_count,
        "method": "conservative_shared_edge_protected_smoothing",
                             }
