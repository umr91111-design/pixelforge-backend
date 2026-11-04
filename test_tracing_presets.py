import unittest

from tracing_presets import (
    DEFAULT_PRESET,
    TRACING_PRESETS,
    get_tracing_preset,
    list_tracing_presets,
)


class TestTracingPresets(unittest.TestCase):

    def test_all_four_presets_exist(self):
        expected = {
            "logo",
            "artwork",
            "line_art",
            "high_detail",
        }

        self.assertEqual(set(TRACING_PRESETS.keys()), expected)

    def test_default_preset_is_artwork(self):
        preset = get_tracing_preset()

        self.assertEqual(DEFAULT_PRESET, "artwork")
        self.assertEqual(preset["name"], "Artwork")

    def test_logo_preset(self):
        preset = get_tracing_preset("logo")

        self.assertEqual(preset["preprocessing_colors"], 6)
        self.assertEqual(preset["preprocessing_smoothing"], 2)

    def test_artwork_preset(self):
        preset = get_tracing_preset("artwork")

        self.assertEqual(preset["preprocessing_colors"], 8)
        self.assertEqual(preset["preprocessing_smoothing"], 3)

    def test_line_art_preset(self):
        preset = get_tracing_preset("line_art")

        self.assertEqual(preset["recommended_mode"], "binary")
        self.assertEqual(preset["preprocessing_colors"], 2)

    def test_high_detail_preset(self):
        preset = get_tracing_preset("high_detail")

        self.assertEqual(preset["preprocessing_colors"], 16)
        self.assertEqual(preset["preprocessing_smoothing"], 1)

    def test_preset_names_are_case_insensitive(self):
        preset = get_tracing_preset("LOGO")

        self.assertEqual(preset["name"], "Logo")

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            get_tracing_preset("unknown")

    def test_non_string_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            get_tracing_preset(None)

    def test_returned_settings_are_independent(self):
        first = get_tracing_preset("artwork")
        first["preprocessing_colors"] = 99

        second = get_tracing_preset("artwork")

        self.assertEqual(second["preprocessing_colors"], 8)

    def test_preset_list_contains_four_items(self):
        presets = list_tracing_presets()

        self.assertEqual(len(presets), 4)

    def test_preset_list_does_not_expose_internal_settings(self):
        presets = list_tracing_presets()

        for preset in presets:
            self.assertNotIn("preprocessing_colors", preset)
            self.assertNotIn("preprocessing_smoothing", preset)


if __name__ == "__main__":
    unittest.main()
