
"""CPU-only processing helpers for the PixelForge Image-to-SVG pipeline."""

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
MAX_PATHS_TO_INSPECT = 10000


def segment_image(image_bytes: bytes) -> dict[str, Any]:
    """Create classical image-segmentation diagnostics using scikit-image."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        array = np.asarray(image)
        gray = color.rgb2gray(array)

        threshold = filters.threshold_otsu(gray)
        mask = gray < threshold

        cleaned = morphology.remove_small_objects(
            mask,
            min_size=12,
            connectivity=2,
        )

        return {
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "threshold": float(threshold),
            "foreground_pixels": int(np.count_nonzero(cleaned)),
            "segmentation_method": "otsu_threshold",
        }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Unable to segment the uploaded image.") from exc


def analyze_contour_geometry(image_bytes: bytes) -> dict[str, Any]:
    """Analyze detected contours and polygon validity with OpenCV and Shapely."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to decode image for contour analysis.")

    height, width = image.shape
    scale = min(1.0, 1600.0 / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
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
    total_area = 0.0

    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue

        points = contour.reshape(-1, 2)

        if len(points) < 3:
            continue

        polygon = Polygon(points)

        if polygon.is_empty or polygon.area <= 0:
            continue

        if polygon.is_valid:
            valid_polygons += 1
            total_area += float(polygon.area)
        else:
            invalid_polygons += 1

    return {
        "contours_found": len(contours),
        "valid_polygons": valid_polygons,
        "invalid_polygons": invalid_polygons,
        "polygon_area_pixels": round(total_area, 2),
        "geometry_engine": "shapely",
    }


def validate_and_inspect_svg(svg_text: str) -> dict[str, Any]:
    """Parse SVG safely and inspect path data without changing its geometry."""
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
    except etree.XMLSyntaxError as exc:
        raise ValueError("Generated SVG is not valid XML.") from exc

    svg_namespace = "http://www.w3.org/2000/svg"
    if etree.QName(root).localname != "svg":
        raise ValueError("The generated document is not an SVG.")

    path_count = 0
    inspected_paths = 0
    invalid_paths = 0
    total_path_length = 0.0

    for element in root.iter():
        if etree.QName(element).localname.lower() != "path":
            continue

        path_count += 1

        if inspected_paths >= MAX_PATHS_TO_INSPECT:
            break

        path_data = element.get("d", "")
        if not path_data:
            invalid_paths += 1
            continue

        try:
            path = parse_path(path_data)
            total_path_length += float(path.length(error=1e-3))
            inspected_paths += 1
        except Exception:
            invalid_paths += 1

    return {
        "valid_svg_xml": True,
        "svg_namespace_present": root.tag.startswith(
            "{" + svg_namespace + "}"
        ),
        "path_count": path_count,
        "inspected_paths": inspected_paths,
        "invalid_paths": invalid_paths,
        "total_path_length": round(total_path_length, 3),
        "path_inspection": "svgpathtools",
    }


def run_vectorizer_analysis(
    image_bytes: bytes,
    svg_text: str,
) -> dict[str, Any]:
    """Run optional CPU analysis after VTracer has produced an SVG."""
    return {
        "segmentation": segment_image(image_bytes),
        "geometry": analyze_contour_geometry(image_bytes),
        "svg": validate_and_inspect_svg(svg_text),
        "deep_learning": False,
        "gpu_required": False,
    }
