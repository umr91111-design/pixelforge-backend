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

from topology_optimizer import analyze_svg_topology

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


def _validate_svg_active_content(root: etree._Element) -> None:
    """Reject active or embedded content in SVG artwork."""
    blocked_elements = {
        "script",
        "foreignobject",
    }

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        element_name = etree.QName(element).localname.lower()

        if element_name in blocked_elements:
            raise ValueError(
                f"SVG contains a prohibited element: {element_name}."
            )

        for attribute_name in element.attrib:
            attribute_local_name = (
                etree.QName(attribute_name).localname.lower()
            )

            if (
                attribute_local_name.startswith("on")
                and len(attribute_local_name) > 2
            ):
                raise ValueError(
                    "SVG contains a prohibited event-handler attribute."
                )


def _path_has_finite_coordinates(path: Path) -> bool:
    """Check that sampled points along SVG segments are finite."""
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


def _parse_svg_root(svg_text: str) -> etree._Element:
    """Parse SVG XML and apply document security checks."""
    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG input is empty.")

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

        if root.getroottree().docinfo.doctype:
            raise ValueError(
                "SVG documents with DOCTYPE are not allowed."
            )

    except etree.XMLSyntaxError as exc:
        raise ValueError("SVG is not valid XML.") from exc

    if not isinstance(root.tag, str):
        raise ValueError("SVG root element is invalid.")

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("Document root is not SVG.")

    _validate_svg_active_content(root)

    return root


def validate_and_inspect_svg(
    svg_text: str,
) -> dict[str, Any]:
    """Securely parse SVG and inspect a bounded number of paths."""
    root = _parse_svg_root(svg_text)

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
    """Run image, SVG and topology analysis after VTracer."""
    return {
        "segmentation": segment_image(image_bytes),
        "geometry": analyze_contour_geometry(image_bytes),
        "svg": validate_and_inspect_svg(svg_text),
        "topology": analyze_svg_topology(svg_text),
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
    """Find paths sharing an identical straight-line edge."""
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
    """Check whether a closed line-only path forms valid geometry."""
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
    Adaptively smooth shallow corners using cubic Bézier curves.

    Features:
    - Angle-dependent Bézier control handles.
    - Adaptive corner trimming.
    - Protection against overlapping trims on short edges.
    - Preservation of sharp corners and shared edges.
    - Preservation of existing SVG attributes.
    """

    if not (
        0 < min_turn_degrees
        < max_turn_degrees
        < 180
    ):
        raise ValueError("Invalid turn-angle settings.")

    if not 0 < trim_ratio <= 0.25:
        raise ValueError("Invalid trim ratio.")

    root = _parse_svg_root(svg_text)

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

            count = len(points)

            if count < 3:
                skipped_paths += 1
                continue

            # Step 1: Analyse corners and calculate adaptive trims.

            corners = [None] * count
            corner_trims = [0.0] * count

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
                    continue

                incoming_unit = incoming / incoming_length
                outgoing_unit = outgoing / outgoing_length

                dot = (
                    incoming_unit.real * outgoing_unit.real
                    + incoming_unit.imag * outgoing_unit.imag
                )

                dot = max(-1.0, min(1.0, dot))

                turn_degrees = float(
                    np.degrees(np.arccos(dot))
                )

                if not (
                    min_turn_degrees
                    <= turn_degrees
                    <= max_turn_degrees
                ):
                    continue

                trim = min(
                    incoming_length,
                    outgoing_length,
                ) * trim_ratio

                if trim < 0.5:
                    continue

                corners[index] = {
                    "incoming": incoming_unit,
                    "outgoing": outgoing_unit,
                    "angle": float(
                        np.radians(turn_degrees)
                    ),
                }

                corner_trims[index] = trim

            # Step 2: Prevent neighbouring curves from overlapping.

            for index in range(count):

                next_index = (index + 1) % count

                edge_length = abs(
                    points[next_index] - points[index]
                )

                if edge_length < 1e-9:
                    continue

                combined_trim = (
                    corner_trims[index]
                    + corner_trims[next_index]
                )

                maximum_trim = edge_length * 0.8

                if combined_trim > maximum_trim:

                    scale = (
                        maximum_trim / combined_trim
                    )

                    corner_trims[index] *= scale
                    corner_trims[next_index] *= scale

            # Step 3: Build adaptive cubic Bézier curves.

            for index in range(count):

                if corners[index] is None:
                    continue

                trim = corner_trims[index]

                if trim < 0.5:
                    corners[index] = None
                    continue

                current = points[index]

                incoming_unit = corners[index]["incoming"]
                outgoing_unit = corners[index]["outgoing"]
                angle = corners[index]["angle"]

                half_angle = angle / 2.0
                quarter_angle = angle / 4.0

                denominator = np.tan(half_angle)

                if abs(denominator) < 1e-9:
                    corners[index] = None
                    continue

                handle = (
                    (4.0 / 3.0)
                    * trim
                    * np.tan(quarter_angle)
                    / denominator
                )

                if not np.isfinite(handle) or handle <= 0:
                    corners[index] = None
                    continue

                start = current - incoming_unit * trim
                end = current + outgoing_unit * trim

                control1 = (
                    start + incoming_unit * handle
                )

                control2 = (
                    end - outgoing_unit * handle
                )

                corners[index] = {
                    "start": start,
                    "end": end,
                    "control1": control1,
                    "control2": control2,
                }

            corner_count = sum(
                corner is not None
                for corner in corners
            )

            if corner_count == 0:
                skipped_paths += 1
                continue

            # Step 4: Reconstruct the closed SVG path.

            new_segments = []

            first_corner = corners[0]

            if first_corner is not None:
                current_position = first_corner["end"]
            else:
                current_position = points[0]

            for step in range(count):

                index = (step + 1) % count
                corner = corners[index]

                if corner is None:

                    target = points[index]

                    if abs(
                        target - current_position
                    ) > 1e-8:

                        new_segments.append(
                            Line(
                                current_position,
                                target,
                            )
                        )

                    current_position = target
                    continue

                start = corner["start"]
                end = corner["end"]

                control1 = corner["control1"]
                control2 = corner["control2"]

                if abs(
                    start - current_position
                ) > 1e-8:

                    new_segments.append(
                        Line(
                            current_position,
                            start,
                        )
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

            # Only modify the path geometry.
            element.set("d", refined_path.d())

            refined_paths += 1
            refined_corners += corner_count

        except Exception:

            skipped_paths += 1

            logger.exception(
                "Adaptive Bézier refinement skipped a path."
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
        "method": "adaptive_angle_based_shared_edge_protected_smoothing",
        }
