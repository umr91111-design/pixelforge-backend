"""Validated VTracer configuration builder for PixelForge AI."""

import vtracer

from tracing_presets import DEFAULT_PRESET, get_tracing_preset


# These are starting profiles, not guaranteed quality rankings.
# We will benchmark them against real images before final tuning.
VTRACER_SETTINGS = {
    "logo": {
        "mode": "spline",
        "filter_speckle": 2,
        "color_precision": 6,
        "layer_difference": 18,
        "corner_threshold": 45,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 2,
        "max_colors": 6,
        "optimize": 1,
    },
    "artwork": {
        "mode": "spline",
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 2,
        "max_colors": 8,
        "optimize": 1,
    },
    "line_art": {
        "mode": "spline",
        "filter_speckle": 2,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 45,
        "length_threshold": 3.5,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 2,
        "max_colors": None,
        "optimize": 1,
    },
    "high_detail": {
        "mode": "spline",
        "filter_speckle": 1,
        "color_precision": 7,
        "layer_difference": 12,
        "corner_threshold": 60,
        "length_threshold": 3.5,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3,
        "max_colors": 16,
        "optimize": 1,
    },
}


def build_vtracer_config(
    preset_name=DEFAULT_PRESET,
    tracing_mode=None,
):
    """Build a VTracer Config using a validated PixelForge preset.

    tracing_mode:
        None     -> use the preset's recommended mode
        "color"  -> color tracing
        "binary" -> monochrome tracing
    """

    preset = get_tracing_preset(preset_name)
    normalized_name = preset_name.strip().lower()

    if tracing_mode is None:
        tracing_mode = preset["recommended_mode"]

    if tracing_mode not in {"color", "binary"}:
        raise ValueError("Tracing mode must be color or binary.")

    settings = VTRACER_SETTINGS[normalized_name].copy()

    if tracing_mode == "binary":
        config = vtracer.Config.bw()
        settings["max_colors"] = None
    else:
        config = vtracer.Config()

    for key, value in settings.items():
        setattr(config, key, value)

    return config
