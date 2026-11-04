
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
    Adaptive color preprocessing for PixelForge hybrid VTracer.

    Features:
    - Adaptive color clustering in LAB color space.
    - Edge-preserving bilateral smoothing.
    - Transparent background preservation.
    - Memory-conscious chunked color mapping.
    - Deterministic color clustering.

    Returns:
        processed_png: PNG image bytes
        report: Processing information

    This function prepares images for vectorization.
    It does not perform vector tracing itself.
    """

    # Validate input
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

    # Decode image safely
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

            width, height = image.size

            if width * height > MAX_PIXELS:
                raise ValueError(
                    "Image exceeds the 4 million pixel limit."
                )

            has_alpha = (
                image.mode in ("RGBA", "LA")
                or (
                    image.mode == "P"
                    and "transparency" in image.info
                )
            )

            if has_alpha:
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
                    (height, width, 1),
                    255,
                    dtype=np.uint8,
                )

                rgba = np.concatenate(
                    (rgb, alpha),
                    axis=2,
                )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "The uploaded image could not be decoded."
        ) from exc

    # Separate RGB and alpha channels
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    visible_mask = alpha > 0
    visible_pixels = rgb[visible_mask]

    # Handle fully transparent images
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

    # Apply edge-preserving smoothing
    working_rgb = rgb.copy()

    if smoothing > 0:
        working_rgb = cv2.bilateralFilter(
            working_rgb,
            d=5,
            sigmaColor=float(smoothing * 12),
            sigmaSpace=float(smoothing * 2),
        )

    # Convert RGB to LAB for perceptual color clustering
    lab = cv2.cvtColor(
        working_rgb,
        cv2.COLOR_RGB2LAB,
    )

    visible_lab = lab[visible_mask]

    # Create a bounded deterministic sample
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

    actual_colors = min(
        colors,
        len(samples),
    )

    # Deterministic k-means color clustering
    cv2.setRNGSeed(42)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
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

    # Convert cluster centers from LAB back to RGB
    centers_lab_uint8 = np.clip(
        np.rint(centers),
        0,
        255,
    ).astype(np.uint8).reshape(1, -1, 3)

    centers_rgb = cv2.cvtColor(
        centers_lab_uint8,
        cv2.COLOR_LAB2RGB,
    ).reshape(-1, 3)

    # Map every visible pixel to its nearest color center
    flat_lab = lab.reshape(-1, 3)

    visible_indices = np.flatnonzero(
        visible_mask.reshape(-1)
    )

    result = rgb.reshape(-1, 3).copy()

    chunk_size = 50_000

    for start in range(
        0,
        len(visible_indices),
        chunk_size,
    ):
        indices = visible_indices[
            start:start + chunk_size
        ]

        pixels = flat_lab[indices].astype(
            np.float32
        )

        distances = (
            (
                pixels[:, None, :]
                - centers[None, :, :]
            ) ** 2
        ).sum(axis=2)

        nearest = np.argmin(
            distances,
            axis=1,
        )

        result[indices] = centers_rgb[nearest]

    # Restore original image dimensions and alpha
    result = result.reshape(
        height,
        width,
        3,
    )

    output_rgba = np.dstack(
        (result, alpha)
    )

    output = Image.fromarray(
        output_rgba,
        "RGBA",
    )

    # Encode PNG
    buffer = io.BytesIO()

    output.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    report = {
        "colors_requested": colors,
        "colors_used": actual_colors,
        "smoothing": smoothing,
        "width": width,
        "height": height,
        "transparent_image": False,
    }

    return buffer.getvalue(), report
