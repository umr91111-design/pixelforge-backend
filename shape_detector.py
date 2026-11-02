import math

import cv2
import numpy as np


def _looks_like_star(approx):
    """Heuristic for common 5-point star contours."""
    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) < 8 or len(points) > 12:
        return False

    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    if len(radii) < 8:
        return False

    outer = np.mean(radii[::2])
    inner = np.mean(radii[1::2])
    ratio = max(outer, inner) / max(min(outer, inner), 1e-6)
    return ratio >= 1.18


def detect_shapes(image_bytes, min_area=150):
    """
    Detect geometric primitives using OpenCV contour analysis.

    This is classical computer vision, not a neural network. Bounding boxes
    and areas are returned in original-image coordinates after any resize.
    """
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to read image.")

    original_height, original_width = image.shape[:2]
    max_side = 1600
    scale = min(1.0, max_side / float(max(original_height, original_width)))

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

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertices = len(approx)
        shape = "unknown"

        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            rotated = cv2.minAreaRect(approx)
            side_a, side_b = rotated[1]
            ratio = max(side_a, side_b) / max(min(side_a, side_b), 1e-6)
            shape = "square" if ratio <= 1.10 else "rectangle"
        elif _looks_like_star(approx):
            shape = "star"
        else:
            circularity = 4 * math.pi * area / max(perimeter * perimeter, 1e-6)
            fitted = None
            if len(contour) >= 5:
                try:
                    fitted = cv2.fitEllipse(contour)
                except cv2.error:
                    pass

            if fitted is not None:
                axis_a, axis_b = fitted[1]
                axis_ratio = max(axis_a, axis_b) / max(min(axis_a, axis_b), 1e-6)
                if circularity >= 0.78 and axis_ratio <= 1.12:
                    shape = "circle"
                elif 1.12 < axis_ratio <= 3.0:
                    shape = "ellipse"
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

    detected.sort(key=lambda item: item["area"], reverse=True)

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
    Safe first-stage SVG enrichment.

    Adds machine-readable recognition metadata while preserving all VTracer
    paths. It does not yet rebuild paths, snap shared edges or optimize a
    topology-aware vector graph; bounding boxes alone cannot safely do that.
    """
    if not isinstance(svg_text, str) or "<svg" not in svg_text[:4000].lower():
        raise ValueError("Invalid SVG supplied for refinement.")

    metadata_items = []
    for item in shapes[:500]:
        box = item.get("bounding_box", {})
        shape = str(item.get("shape", "unknown")).replace('"', "")
        metadata_items.append(
            '<pf:primitive shape="{shape}" x="{x}" y="{y}" width="{w}" height="{h}" />'.format(
                shape=shape,
                x=int(box.get("x", 0)),
                y=int(box.get("y", 0)),
                w=int(box.get("width", 0)),
                h=int(box.get("height", 0)),
            )
        )

    if not metadata_items:
        return svg_text

    marker = (
        '<metadata id="pixelforge-recognition">'
        '<pf:recognition xmlns:pf="https://pixelforge.ai/ns">'
        + "".join(metadata_items)
        + "</pf:recognition></metadata>"
    )

    root_end = svg_text.find(">")
    if root_end < 0:
        raise ValueError("Malformed SVG root element.")

    return svg_text[:root_end + 1] + marker + svg_text[root_end + 1:]
