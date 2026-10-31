import cv2
import numpy as np


def detect_shapes(image_bytes, min_area=150):
    """
    Detect basic geometric shapes in an image.
    Returns shape labels and bounding boxes.
    """

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read image.")

    height, width = image.shape[:2]

    # Reduce processing cost for large images.
    max_side = 1600
    scale = min(1.0, max_side / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        vertices = len(approx)
        shape = "unknown"

        if vertices == 3:
            shape = "triangle"

        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            if 0.90 <= ratio <= 1.10:
                shape = "square"
            else:
                shape = "rectangle"

        elif vertices > 4:
            circularity = (
                4 * np.pi * area / (perimeter * perimeter)
            )

            if circularity > 0.80:
                shape = "circle"
            else:
                shape = "polygon"

        x, y, w, h = cv2.boundingRect(contour)

        detected.append({
            "shape": shape,
            "area": round(float(area), 2),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
        })

    return {
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "shapes_found": len(detected),
        "shapes": detected
    }
