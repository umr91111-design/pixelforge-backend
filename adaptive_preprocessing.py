
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_SAMPLE_PIXELS = 30_000
MAX_COLORS = 16


def adaptive_color_preprocess(
    image_bytes: bytes,
    colors: int = 8,
    smoothing: int = 3,
):
    """
    Reduce an image to an adaptive color palette.

    Returns:
        processed_png: PNG image bytes
        report: Information about the preprocessing

    This function does not perform vector tracing.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("A valid image is required.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 5 MB upload limit.")

    if not isinstance(colors, int) or isinstance(colors, bool):
        raise ValueError("Colors must be an integer.")

    if not 2 <= colors <= MAX_COLORS:
        raise ValueError("Colors must be between 2 and 16.")

    if smoothing not in (0, 3, 5, 7):
        raise ValueError("Smoothing must be 0, 3, 5, or 7.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

            if image.width * image.height > MAX_PIXELS:
                raise ValueError(
                    "Image exceeds the 4 million pixel limit."
                )

            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = np.asarray(
                    image.convert("RGBA"),
                    dtype=np.uint8,
                ).copy()
            else:
                rgb = np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                ).copy()

                alpha = np.full(
                    (image.height, image.width, 1),
                    255,
                    dtype=np.uint8,
                )

                rgba = np.concatenate((rgb, alpha), axis=2)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc

    height, width = rgba.shape[:2]

    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    visible_mask = alpha > 0
    visible_pixels = rgb[visible_mask]

    if len(visible_pixels) == 0:
        output = Image.fromarray(rgba, "RGBA")
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        return buffer.getvalue(), {
            "colors_requested": colors,
            "colors_used": 0,
            "smoothing": smoothing,
            "width": width,
            "height": height,
            "transparent_image": True,
        }

    working_rgb = rgb.copy()

    if smoothing:
        working_rgb = cv2.bilateralFilter(
            working_rgb,
            d=5,
            sigmaColor=float(smoothing * 12),
            sigmaSpace=float(smoothing * 2),
        )

    lab = cv2.cvtColor(
        working_rgb,
        cv2.COLOR_RGB2LAB,
    )

    visible_lab = lab[visible_mask]

    sample_count = min(
        len(visible_lab),
        MAX_SAMPLE_PIXELS,
    )

    if len(visible_lab) > sample_count:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(
            len(visible_lab),
            size=sample_count,
            replace=False,
        )
        samples = visible_lab[sample_indices]
    else:
        samples = visible_lab

    samples = np.ascontiguousarray(
        samples.reshape(-1, 3),
        dtype=np.float32,
    )

    actual_colors = min(colors, len(samples))

    cv2.setRNGSeed(42)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5,
    )

    _, _, centers = cv2.kmeans(
        samples,
        actual_colors,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.astype(np.float32)

    flat_lab = lab.reshape(-1, 3)
    visible_indices = np.flatnonzero(visible_mask.reshape(-1))

    result = rgb.reshape(-1, 3).copy()

    chunk_size = 50_000

    for start in range(0, len(visible_indices), chunk_size):
        indices = visible_indices[
            start:start + chunk_size
        ]

        pixels = flat_lab[indices].astype(np.float32)

        distances = (
            (pixels[:, None, :] - centers[None, :, :]) ** 2
        ).sum(axis=2)

        nearest = np.argmin(distances, axis=1)

        result[indices] = centers[nearest].clip(
            0, 255
        ).astype(np.uint8)

    result = result.reshape(height, width, 3)

    output_rgba = np.dstack((result, alpha))

    output = Image.fromarray(output_rgba, "RGBA")

    buffer = io.BytesIO()
    output.save(buffer, format="PNG", optimize=True)

    report = {
        "colors_requested": colors,
        "colors_used": actual_colors,
        "smoothing": smoothing,
        "width": width,
        "height": height,
        "transparent_image": False,
    }

    return buffer.getvalue(), report
