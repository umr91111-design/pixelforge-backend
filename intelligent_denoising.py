
"""Conservative, edge-aware noise removal for PixelForge AI."""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4_000_000

STRENGTH_SETTINGS = {
    "light": {
        "difference_threshold": 30,
        "neighbor_threshold": 12,
    },
    "balanced": {
        "difference_threshold": 22,
        "neighbor_threshold": 16,
    },
    "strong": {
        "difference_threshold": 16,
        "neighbor_threshold": 20,
    },
}


def _neighbor_support(lab_image, threshold):
    """Count nearby pixels with a similar LAB color."""

    height, width = lab_image.shape[:2]
    support = np.zeros((height, width), dtype=np.uint8)

    threshold_squared = threshold * threshold

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            if dx == 0 and dy == 0:
                continue

            y1 = max(0, -dy)
            y2 = min(height, height - dy)
            x1 = max(0, -dx)
            x2 = min(width, width - dx)

            if y1 >= y2 or x1 >= x2:
                continue

            # int32 prevents overflow during squared distances.
            center = lab_image[
                y1:y2, x1:x2
            ].astype(np.int32)

            neighbor = lab_image[
                y1 + dy:y2 + dy,
                x1 + dx:x2 + dx,
            ].astype(np.int32)

            difference = center - neighbor

            distance_squared = np.sum(
                difference * difference,
                axis=2,
            )

            support[y1:y2, x1:x2] += (
                distance_squared <= threshold_squared
            ).astype(np.uint8)

    return support


def remove_image_noise(
    image_bytes: bytes,
    strength: str = "balanced",
):
    """
    Remove conservative isolated color outliers.

    Connected details and pixels near transparency boundaries
    are protected. Alpha values and dimensions are preserved.

    Returns:
        PNG image bytes and a processing report.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("A valid image is required.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 5 MB upload limit.")

    if strength not in STRENGTH_SETTINGS:
        raise ValueError(
            "Strength must be light, balanced, or strong."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:

            image = ImageOps.exif_transpose(source)
            image.load()

            width, height = image.size

            if width < 1 or height < 1:
                raise ValueError(
                    "Image dimensions are invalid."
                )

            if width * height > MAX_PIXELS:
                raise ValueError(
                    "Image exceeds the 4 million pixel limit."
                )

            rgba = np.asarray(
                image.convert("RGBA"),
                dtype=np.uint8,
            ).copy()

    except ValueError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:

        raise ValueError(
            "The uploaded image could not be decoded."
        ) from exc

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    visible = alpha > 0

    if not np.any(visible):

        output = Image.fromarray(rgba, "RGBA")

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        return buffer.getvalue(), {
            "strength": strength,
            "pixels_examined": 0,
            "pixels_replaced": 0,
            "transparent_image": True,
            "width": width,
            "height": height,
        }

    settings = STRENGTH_SETTINGS[strength]

    # Generate a local median replacement candidate.
    median_rgb = cv2.medianBlur(rgb, 3)

    original_lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB,
    )

    median_lab = cv2.cvtColor(
        median_rgb,
        cv2.COLOR_RGB2LAB,
    )

    # Identify pixels that have no similar neighboring pixels.
    support = _neighbor_support(
        original_lab,
        settings["neighbor_threshold"],
    )

    # Use int32 to prevent squared-distance overflow.
    difference = (
        original_lab.astype(np.int32)
        - median_lab.astype(np.int32)
    )

    distance_squared = np.sum(
        difference * difference,
        axis=2,
    )

    threshold_squared = (
        settings["difference_threshold"] ** 2
    )

    # Protect pixels next to transparent areas.
    opaque_mask = (
        alpha > 0
    ).astype(np.uint8)

    safe_interior = cv2.erode(
        opaque_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    # Only replace isolated pixels that strongly differ
    # from their local median.
    candidates = (
        visible
        & safe_interior
        & (support == 0)
        & (distance_squared > threshold_squared)
    )

    result = rgba.copy()

    result_rgb = result[:, :, :3]

    result_rgb[candidates] = median_rgb[candidates]

    output = Image.fromarray(result, "RGBA")

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    report = {
        "strength": strength,
        "pixels_examined": int(
            np.count_nonzero(visible)
        ),
        "pixels_replaced": int(
            np.count_nonzero(candidates)
        ),
        "transparent_image": False,
        "alpha_preserved": True,
        "width": width,
        "height": height,
        "method": "conservative_isolated_outlier_removal",
    }

    return buffer.getvalue(), report
