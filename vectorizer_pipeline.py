
"""Unified CPU-only analysis pipeline for PixelForge AI vectorization."""

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
from svgpathtools import parse_path

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


def _decode_image(image_bytes: bytes) -> Image.Image:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Unable to decode the uploaded image.") from exc


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


def analyze_contour_geometry(image_bytes: bytes) -> dict[str, Any]:
    """Analyse contours and polygon validity using OpenCV and Shapely."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to decode image for contour analysis.")

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
                original_area = (
                    float(polygon.area) * scale_x * scale_y
                )
                total_area += original_area
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


def validate_and_inspect_svg(svg_text: str) -> dict[str, Any]:
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
            raise ValueError("SVG documents with DOCTYPE are not allowed.")

    except etree.XMLSyntaxError as exc:
        raise ValueError("Generated SVG is not valid XML.") from exc

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

            if segment_count > MAX_SEGMENTS_PER_PATH:
                skipped_paths += 1
                continue

            if total_segments + segment_count > MAX_TOTAL_SEGMENTS:
                skipped_paths += 1
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
    """Run the unified CPU analysis after VTracer generates SVG."""
    return {
        "segmentation": segment_image(image_bytes),
        "geometry": analyze_contour_geometry(image_bytes),
        "svg": validate_and_inspect_svg(svg_text),
        "deep_learning": False,
        "gpu_required": False,
    }
