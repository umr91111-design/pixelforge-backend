
import math

import cv2
import numpy as np
from lxml import etree


SVG_NS = "http://www.w3.org/2000/svg"
PF_NS = "https://pixelforge.ai/ns"

MAX_IMAGE_SIDE = 1600
MAX_DETECTED_SHAPES = 500


def _looks_like_star(approx):
    """Recognize common star-like contours using alternating radii."""
    points = approx.reshape(-1, 2).astype(np.float32)

    if not 8 <= len(points) <= 12:
        return False

    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)

    outer = radii[::2]
    inner = radii[1::2]

    if len(outer) < 4 or len(inner) < 4:
        return False

    outer_mean = float(np.mean(outer))
    inner_mean = float(np.mean(inner))

    if min(outer_mean, inner_mean) <= 1e-6:
        return False

    ratio = max(outer_mean, inner_mean) / min(outer_mean, inner_mean)
    return ratio >= 1.35


def detect_shapes(image_bytes, min_area=150):
    """
    Detect geometric primitives using OpenCV contour analysis.

    This is classical computer vision, not a neural network.
    Returned areas and bounding boxes use original-image coordinates.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("Image data must be bytes.")

    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a non-negative number.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read image.")

    original_height, original_width = image.shape[:2]

    scale = min(
        1.0,
        MAX_IMAGE_SIDE / float(max(original_height, original_width)),
    )

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

    work_height, work_width = image.shape[:2]

    scale_x = original_width / float(work_width)
    scale_y = original_height / float(work_height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detected = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        vertices = len(approx)
        shape = "unknown"

        if vertices == 3:
            shape = "triangle"

        elif vertices == 4:
            rotated = cv2.minAreaRect(approx)
            side_a, side_b = rotated[1]

            if min(side_a, side_b) > 1e-6:
                ratio = max(side_a, side_b) / min(side_a, side_b)
                shape = "square" if ratio <= 1.10 else "rectangle"

        elif _looks_like_star(approx):
            shape = "star"

        else:
            circularity = (
                4.0 * math.pi * area
                / max(perimeter * perimeter, 1e-6)
            )

            fitted = None

            if len(contour) >= 5:
                try:
                    fitted = cv2.fitEllipse(contour)
                except cv2.error:
                    fitted = None

            if fitted is not None:
                axis_a, axis_b = fitted[1]

                if min(axis_a, axis_b) > 1e-6:
                    axis_ratio = max(axis_a, axis_b) / min(axis_a, axis_b)

                    if circularity >= 0.78 and axis_ratio <= 1.12:
                        shape = "circle"
                    elif 1.12 < axis_ratio <= 3.0:
                        shape = "ellipse"
                    else:
                        shape = "polygon"
                else:
                    shape = "polygon"

            elif circularity >= 0.80:
                shape = "circle"

            else:
                shape = "polygon"

        x, y, box_width, box_height = cv2.boundingRect(contour)

        detected.append({
            "shape": shape,
            "area": round(float(area * scale_x * scale_y), 2),
            "bounding_box": {
                "x": int(round(x * scale_x)),
                "y": int(round(y * scale_y)),
                "width": int(round(box_width * scale_x)),
                "height": int(round(box_height * scale_y)),
            },
            "recognition_method": "opencv_contours",
            "confidence": None,
        })

        if len(detected) >= MAX_DETECTED_SHAPES:
            break

    detected.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    return {
        "image_width": int(original_width),
        "image_height": int(original_height),
        "processing_width": int(work_width),
        "processing_height": int(work_height),
        "shapes_found": len(detected),
        "shapes": detected,
    }


def refine_svg_with_primitives(svg_text, shapes):
    """
    Add machine-readable shape-recognition metadata to an SVG.

    Original VTracer paths are preserved. This function does not yet
    repair shared boundaries, gaps, overlaps, or Bézier geometry.
    """
    if not isinstance(svg_text, str):
        raise ValueError("SVG content must be text.")

    if not isinstance(shapes, list):
        raise ValueError("Shapes must be provided as a list.")

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
        raise ValueError("Invalid SVG supplied for refinement.") from exc

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("SVG root element is invalid.")

    if root.getroottree().docinfo.doctype:
        raise ValueError("SVG documents containing a DOCTYPE are not allowed.")

    if not shapes:
        return etree.tostring(
            root,
            encoding="unicode",
        )

    metadata = root.find(
        "{http://www.w3.org/2000/svg}metadata"
    )

    if metadata is None:
        metadata = etree.Element(
            f"{{{SVG_NS}}}metadata",
            id="pixelforge-recognition",
        )
        root.insert(0, metadata)

    recognition = etree.SubElement(
        metadata,
        f"{{{PF_NS}}}recognition",
    )

    for item in shapes[:MAX_DETECTED_SHAPES]:
        if not isinstance(item, dict):
            continue

        box = item.get("bounding_box", {})

        if not isinstance(box, dict):
            continue

        try:
            x = int(box.get("x", 0))
            y = int(box.get("y", 0))
            width = int(box.get("width", 0))
            height = int(box.get("height", 0))
        except (TypeError, ValueError, OverflowError):
            continue

        primitive = etree.SubElement(
            recognition,
            f"{{{PF_NS}}}primitive",
        )

        primitive.set(
            "shape",
            str(item.get("shape", "unknown"))[:40],
        )
        primitive.set("x", str(x))
        primitive.set("y", str(y))
        primitive.set("width", str(width))
        primitive.set("height", str(height))

    return etree.tostring(
        root,
        encoding="unicode",
    )
