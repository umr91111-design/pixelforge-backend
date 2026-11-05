
import io
import threading
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_SAMPLE_PIXELS = 30_000
MAX_COLORS = 16
CHUNK_SIZE = 50_000

# OpenCV's RNG seed is global. Protect k-means initialization
# so simultaneous requests cannot change each other's seed.
_KMEANS_LOCK = threading.Lock()


def _decode_image(image_bytes: bytes):
    """Decode an image after checking its dimensions."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(io.BytesIO(image_bytes)) as source:
                width, height = source.size

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions.")

                if width * height > MAX_PIXELS:
                    raise ValueError(
                        "Image exceeds the 4 million pixel limit."
                    )

                image = ImageOps.exif_transpose(source)
                image.load()

                width, height = image.size

                if width * height > MAX_PIXELS:
                    raise ValueError(
                        "Image exceeds the 4 million pixel limit."
                    )

                has_alpha = (
                    "A" in image.getbands()
                    or (
                        image.mode == "P"
                        and "transparency" in image.info
                    )
                )

                rgba = np.asarray(
                    image.convert("RGBA"),
                    dtype=np.uint8,
                ).copy()

                return rgba, has_alpha

    except ValueError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
    ) as exc:
        raise ValueError(
            "The uploaded image could not be decoded safely."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "The uploaded image could not be decoded."
        ) from exc


def _build_sample(
    visible_lab: np.ndarray,
    edge_mask: np.ndarray,
    sample_limit: int,
) -> np.ndarray:
    """
    Build a deterministic sample containing both ordinary
    image colors and pixels near important edges.
    """

    total = len(visible_lab)

    if total <= sample_limit:
        return visible_lab

    edge_positions = np.flatnonzero(edge_mask)

    edge_budget = min(
        len(edge_positions),
        sample_limit // 3,
    )

    regular_budget = sample_limit - edge_budget

    # Evenly distributed deterministic sampling.
    regular_indices = np.linspace(
        0,
        total - 1,
        num=regular_budget,
        dtype=np.int64,
    )

    selected = [regular_indices]

    if edge_budget > 0:
        edge_indices = np.linspace(
            0,
            len(edge_positions) - 1,
            num=edge_budget,
            dtype=np.int64,
        )

        selected.append(
            edge_positions[edge_indices]
        )

    indices = np.unique(
        np.concatenate(selected)
    )

    # Fill any unused positions deterministically.
    if len(indices) < sample_limit:
        available = np.setdiff1d(
            np.arange(total),
            indices,
            assume_unique=True,
        )

        extra_count = min(
            sample_limit - len(indices),
            len(available),
        )

        if extra_count:
            extra_indices = np.linspace(
                0,
                len(available) - 1,
                num=extra_count,
                dtype=np.int64,
            )

            indices = np.concatenate(
                (indices, available[extra_indices])
            )

    return visible_lab[indices]


def _run_kmeans(
    samples: np.ndarray,
    color_count: int,
):
    """Run deterministic OpenCV k-means safely."""

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.4,
    )

    with _KMEANS_LOCK:
        cv2.setRNGSeed(42)

        _, _, centers = cv2.kmeans(
            samples,
            color_count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

    return centers.astype(np.float32)


def adaptive_color_preprocess(
    image_bytes: bytes,
    colors: int = 8,
    smoothing: int = 3,
):
    """
    Adaptive color preprocessing for PixelForge hybrid VTracer.

    Features:
    - Perceptual LAB color clustering.
    - Deterministic spatial and edge-aware sampling.
    - Edge-preserving bilateral smoothing.
    - Transparent background preservation.
    - Chunked nearest-color mapping.
    - Safe image dimension and upload validation.

    Returns:
        processed_png: PNG image bytes
        report: Processing information

    This function prepares images for vectorization.
    It does not perform vector tracing itself.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("A valid image is required.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 5 MB upload limit.")

    if (
        not isinstance(colors, int)
        or isinstance(colors, bool)
        or not 2 <= colors <= MAX_COLORS
    ):
        raise ValueError("Colors must be an integer between 2 and 16.")

    if (
        not isinstance(smoothing, int)
        or isinstance(smoothing, bool)
        or smoothing not in (0, 3, 5, 7)
    ):
        raise ValueError("Smoothing must be 0, 3, 5, or 7.")

    rgba, has_alpha = _decode_image(image_bytes)

    height, width = rgba.shape[:2]

    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    visible_mask = alpha > 0
    visible_pixels = rgb[visible_mask]

    # Fully transparent image: preserve it unchanged.
    if len(visible_pixels) == 0:
        output = Image.fromarray(rgba, "RGBA")

        buffer = io.BytesIO()
        output.save(buffer, format="PNG", optimize=True)

        return buffer.getvalue(), {
            "colors_requested": colors,
            "colors_used": 0,
            "smoothing": smoothing,
            "width": width,
            "height": height,
            "transparent_image": True,
            "alpha_preserved": True,
            "sampling": "not_required",
        }

    # Prevent hidden RGB values in transparent pixels from
    # contaminating visible pixels during smoothing.
    working_rgb = rgb.copy()
    working_rgb[~visible_mask] = 0

    if smoothing > 0:
        working_rgb = cv2.bilateralFilter(
            working_rgb,
            d=5,
            sigmaColor=float(smoothing * 12),
            sigmaSpace=float(smoothing * 2),
        )

        # Restore original RGB values for fully transparent pixels.
        working_rgb[~visible_mask] = 0

    lab = cv2.cvtColor(
        working_rgb,
        cv2.COLOR_RGB2LAB,
    )

    visible_lab = lab[visible_mask]

    # Detect edges in the visible image to help retain
    # samples around small shapes and region boundaries.
    gray = cv2.cvtColor(
        working_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150,
    )

    edge_mask = edges[visible_mask] > 0

    samples = _build_sample(
        visible_lab,
        edge_mask,
        MAX_SAMPLE_PIXELS,
    )

    samples = np.ascontiguousarray(
        samples.reshape(-1, 3),
        dtype=np.float32,
    )

    actual_colors = min(
        colors,
        len(samples),
    )

    centers = _run_kmeans(
        samples,
        actual_colors,
    )

    centers_lab = np.clip(
        np.rint(centers),
        0,
        255,
    ).astype(np.uint8).reshape(1, -1, 3)

    centers_rgb = cv2.cvtColor(
        centers_lab,
        cv2.COLOR_LAB2RGB,
    ).reshape(-1, 3)

    # Map visible pixels to their nearest LAB color center.
    flat_lab = lab.reshape(-1, 3)

    visible_indices = np.flatnonzero(
        visible_mask.reshape(-1)
    )

    result = rgb.reshape(-1, 3).copy()

    for start in range(
        0,
        len(visible_indices),
        CHUNK_SIZE,
    ):
        indices = visible_indices[
            start:start + CHUNK_SIZE
        ]

        pixels = flat_lab[indices].astype(
            np.float32
        )

        differences = (
            pixels[:, None, :]
            - centers[None, :, :]
        )

        distances = np.sum(
            differences * differences,
            axis=2,
        )

        nearest = np.argmin(
            distances,
            axis=1,
        )

        result[indices] = centers_rgb[nearest]

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

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    # Count colors actually present in the output.
    colors_used = len(
        np.unique(
            result[visible_mask],
            axis=0,
        )
    )

    report = {
        "colors_requested": colors,
        "colors_used": colors_used,
        "smoothing": smoothing,
        "width": width,
        "height": height,
        "transparent_image": False,
        "alpha_preserved": True,
        "sampling": "spatial_and_edge_aware",
        "sample_pixels": len(samples),
        "edge_samples_available": int(
            np.count_nonzero(edge_mask)
        ),
    }

    return buffer.getvalue(), report
