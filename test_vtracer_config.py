"""Tests for PixelForge AI VTracer configuration."""

import unittest

from vtracer_config import (
    VTRACER_SETTINGS,
    build_vtracer_config,
)


class TestVTracerConfig(unittest.TestCase):

    def test_all_presets_have_settings(self):
        expected = {
            "logo",
            "artwork",
            "line_art",
            "high_detail",
        }

        self.assertEqual(set(VTRACER_SETTINGS.keys()), expected)

    def test_artwork_config(self):
        config = build_vtracer_config("artwork", "color")

        self.assertEqual(config.mode, "spline")
        self.assertEqual(config.color_precision, 6)
        self.assertEqual(config.max_colors, 8)

    def test_logo_config(self):
        config = build_vtracer_config("logo", "color")

        self.assertEqual(config.filter_speckle, 2)
        self.assertEqual(config.max_colors, 6)

    def test_high_detail_config(self):
        config = build_vtracer_config("high_detail", "color")

        self.assertEqual(config.color_precision, 7)
        self.assertEqual(config.path_precision, 3)
        self.assertEqual(config.max_colors, 16)

    def test_line_art_defaults_to_binary(self):
        config = build_vtracer_config("line_art")

        self.assertEqual(config.clustering, "bw")

    def test_explicit_binary_mode(self):
        config = build_vtracer_config("artwork", "binary")

        self.assertEqual(config.clustering, "bw")
        self.assertIsNone(config.max_colors)

    def test_explicit_color_mode(self):
        config = build_vtracer_config("line_art", "color")

        self.assertEqual(config.max_colors, None)

    def test_invalid_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            build_vtracer_config("unknown", "color")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_vtracer_config("artwork", "invalid")

    def test_settings_are_not_mutated(self):
        original = VTRACER_SETTINGS["artwork"]["max_colors"]

        build_vtracer_config("artwork", "binary")

        self.assertEqual(
            VTRACER_SETTINGS["artwork"]["max_colors"],
            original,
        )


if __name__ == "__main__":
    unittest.main()
