"""Advanced tracing presets for PixelForge AI."""

from copy import deepcopy


DEFAULT_PRESET = "artwork"

TRACING_PRESETS = {
    "logo": {
        "name": "Logo",
        "description": "Cleaner edges and simplified color regions.",
        "recommended_mode": "color",
        "preprocessing_colors": 6,
        "preprocessing_smoothing": 2,
    },

    "artwork": {
        "name": "Artwork",
        "description": "Balanced color preservation and detail.",
        "recommended_mode": "color",
        "preprocessing_colors": 8,
        "preprocessing_smoothing": 3,
    },

    "line_art": {
        "name": "Line Art",
        "description": "Designed for clean outlines and monochrome artwork.",
        "recommended_mode": "binary",
        "preprocessing_colors": 2,
        "preprocessing_smoothing": 1,
    },

    "high_detail": {
        "name": "High Detail",
        "description": "Preserves more color variation and fine details.",
        "recommended_mode": "color",
        "preprocessing_colors": 16,
        "preprocessing_smoothing": 1,
    },
}


def get_tracing_preset(preset_name=DEFAULT_PRESET):
    """Return a safe copy of the selected tracing preset."""

    if not isinstance(preset_name, str):
        raise ValueError("Preset name must be text.")

    normalized_name = preset_name.strip().lower()

    if normalized_name not in TRACING_PRESETS:
        raise ValueError(
            "Unknown tracing preset. Choose logo, artwork, "
            "line_art, or high_detail."
        )

    return deepcopy(TRACING_PRESETS[normalized_name])


def list_tracing_presets():
    """Return available presets without exposing internal settings."""

    return [
        {
            "id": preset_id,
            "name": settings["name"],
            "description": settings["description"],
            "recommended_mode": settings["recommended_mode"],
        }
        for preset_id, settings in TRACING_PRESETS.items()
      ]
