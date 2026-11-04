import io
import unittest

import numpy as np
from PIL import Image

from adaptive_preprocessing import adaptive_color_preprocess


def make_test_image(mode="RGB"):
    image = Image.new(mode, (64, 64))

    if mode == "RGB":
        pixels = image.load()

        for y in range(64):
            for x in range(64):
                if x < 32:
                    pixels[x, y] = (255, 0, 0)
                else:
                    pixels[x, y] = (0, 0, 255)

    elif mode == "RGBA":
        pixels = image.load()

        for y in range(64):
            for x in range(64):
                if x < 32:
                    pixels[x, y] = (255, 0, 0, 255)
                else:
                    pixels[x, y] = (0, 0, 255, 0)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestAdaptivePreprocessing(unittest.TestCase):

    def test_output_is_valid_png(self):
        data, report = adaptive_color_preprocess(
            make_test_image(),
            colors=4,
        )

        result = Image.open(io.BytesIO(data))

        self.assertEqual(result.format, "PNG")
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(report["width"], 64)
        self.assertEqual(report["height"], 64)

    def test_color_count_is_limited(self):
        data, report = adaptive_color_preprocess(
            make_test_image(),
            colors=2,
        )

        result = np.asarray(
            Image.open(io.BytesIO(data)).convert("RGB")
        )

        unique_colors = np.unique(
            result.reshape(-1, 3),
            axis=0,
        )

        self.assertLessEqual(len(unique_colors), 2)
        self.assertLessEqual(report["colors_used"], 2)

    def test_transparency_is_preserved(self):
        data, _ = adaptive_color_preprocess(
            make_test_image("RGBA"),
            colors=4,
        )

        result = np.asarray(
            Image.open(io.BytesIO(data)).convert("RGBA")
        )

        self.assertTrue(np.all(result[:, 32:, 3] == 0))
        self.assertTrue(np.all(result[:, :32, 3] == 255))

    def test_all_transparent_image(self):
        image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        data, report = adaptive_color_preprocess(
            buffer.getvalue(),
            colors=4,
        )

        result = np.asarray(
            Image.open(io.BytesIO(data)).convert("RGBA")
        )

        self.assertTrue(np.all(result[:, :, 3] == 0))
        self.assertTrue(report["transparent_image"])

    def test_invalid_color_count(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(
                make_test_image(),
                colors=1,
            )

    def test_invalid_smoothing(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(
                make_test_image(),
                smoothing=4,
            )

    def test_invalid_image_bytes(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(
                b"not an image",
            )

    def test_empty_image_bytes(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(b"")


if __name__ == "__main__":
    unittest.main()
