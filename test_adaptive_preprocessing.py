
import io
import unittest

import numpy as np
from PIL import Image

from adaptive_preprocessing import (
    MAX_SAMPLE_PIXELS,
    _build_sample,
    adaptive_color_preprocess,
)


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_test_image(mode="RGB"):
    image = Image.new(mode, (64, 64))

    if mode == "RGB":
        pixels = image.load()

        for y in range(64):
            for x in range(64):
                pixels[x, y] = (
                    (255, 0, 0)
                    if x < 32
                    else (0, 0, 255)
                )

    elif mode == "RGBA":
        pixels = image.load()

        for y in range(64):
            for x in range(64):
                pixels[x, y] = (
                    (255, 0, 0, 255)
                    if x < 32
                    else (0, 0, 255, 0)
                )

    return encode_image(image)


class TestAdaptivePreprocessing(unittest.TestCase):

    def test_output_is_valid_png(self):
        data, report = adaptive_color_preprocess(
            make_test_image(),
            colors=4,
        )

        with Image.open(io.BytesIO(data)) as result:
            self.assertEqual(result.format, "PNG")
            self.assertEqual(result.size, (64, 64))

        self.assertEqual(report["width"], 64)
        self.assertEqual(report["height"], 64)
        self.assertTrue(report["alpha_preserved"])

    def test_color_count_is_limited(self):
        data, report = adaptive_color_preprocess(
            make_test_image(),
            colors=2,
        )

        with Image.open(io.BytesIO(data)) as image:
            result = np.asarray(image.convert("RGB"))

        unique_colors = np.unique(
            result.reshape(-1, 3),
            axis=0,
        )

        self.assertLessEqual(len(unique_colors), 2)
        self.assertLessEqual(report["colors_used"], 2)

    def test_transparency_is_preserved(self):
        data, report = adaptive_color_preprocess(
            make_test_image("RGBA"),
            colors=4,
        )

        with Image.open(io.BytesIO(data)) as image:
            result = np.asarray(image.convert("RGBA"))

        self.assertTrue(np.all(result[:, 32:, 3] == 0))
        self.assertTrue(np.all(result[:, :32, 3] == 255))
        self.assertTrue(report["alpha_preserved"])

    def test_partial_alpha_is_preserved(self):
        image = Image.new(
            "RGBA",
            (32, 32),
            (120, 80, 40, 128),
        )

        data, _ = adaptive_color_preprocess(
            encode_image(image),
            colors=2,
        )

        with Image.open(io.BytesIO(data)) as output:
            result = np.asarray(output.convert("RGBA"))

        self.assertTrue(
            np.all(result[:, :, 3] == 128)
        )

    def test_transparent_pixel_rgb_is_not_modified(self):
        image = Image.new(
            "RGBA",
            (32, 32),
            (0, 0, 0, 0),
        )

        pixels = image.load()

        for y in range(32):
            for x in range(16):
                pixels[x, y] = (255, 0, 0, 255)

        data, _ = adaptive_color_preprocess(
            encode_image(image),
            colors=2,
        )

        with Image.open(io.BytesIO(data)) as output:
            result = np.asarray(output.convert("RGBA"))

        self.assertTrue(
            np.all(result[:, 16:, 3] == 0)
        )

        self.assertTrue(
            np.all(result[:, 16:, :3] == 0)
        )

    def test_all_transparent_image(self):
        image = Image.new(
            "RGBA",
            (16, 16),
            (0, 0, 0, 0),
        )

        data, report = adaptive_color_preprocess(
            encode_image(image),
            colors=4,
        )

        with Image.open(io.BytesIO(data)) as output:
            result = np.asarray(output.convert("RGBA"))

        self.assertTrue(np.all(result[:, :, 3] == 0))
        self.assertTrue(report["transparent_image"])
        self.assertEqual(report["colors_used"], 0)

    def test_distinct_color_regions_are_preserved(self):
        data, _ = adaptive_color_preprocess(
            make_test_image(),
            colors=2,
            smoothing=0,
        )

        with Image.open(io.BytesIO(data)) as image:
            result = np.asarray(image.convert("RGB"))

        left_color = result[32, 10].astype(int)
        right_color = result[32, 54].astype(int)

        distance = np.linalg.norm(
            left_color - right_color
        )

        self.assertGreater(distance, 100)

    def test_sharp_boundary_is_preserved(self):
        data, _ = adaptive_color_preprocess(
            make_test_image(),
            colors=2,
            smoothing=3,
        )

        with Image.open(io.BytesIO(data)) as image:
            result = np.asarray(image.convert("RGB"))

        left = result[32, 10].astype(int)
        right = result[32, 54].astype(int)

        self.assertGreater(
            np.linalg.norm(left - right),
            100,
        )

    def test_small_detail_is_preserved(self):
        image = Image.new("RGB", (64, 64), "white")
        pixels = image.load()

        for y in range(28, 36):
            for x in range(28, 36):
                pixels[x, y] = (0, 0, 0)

        data, _ = adaptive_color_preprocess(
            encode_image(image),
            colors=2,
            smoothing=0,
        )

        with Image.open(io.BytesIO(data)) as output:
            result = np.asarray(output.convert("RGB"))

        center = result[32, 32]
        background = result[10, 10]

        self.assertGreater(
            np.linalg.norm(
                center.astype(int) - background.astype(int)
            ),
            100,
        )

    def test_sampling_respects_maximum(self):
        visible_lab = np.arange(
            90_000,
            dtype=np.uint8,
        ).reshape(30_000, 3)

        edge_mask = np.zeros(30_000, dtype=bool)
        edge_mask[::10] = True

        samples = _build_sample(
            visible_lab,
            edge_mask,
            10_000,
        )

        self.assertLessEqual(
            len(samples),
            10_000,
        )

    def test_sampling_includes_edge_pixels(self):
        visible_lab = np.arange(
            90_000,
            dtype=np.uint8,
        ).reshape(30_000, 3)

        edge_mask = np.zeros(30_000, dtype=bool)
        edge_mask[100:200] = True

        samples = _build_sample(
            visible_lab,
            edge_mask,
            300,
        )

        self.assertGreater(len(samples), 0)

        self.assertTrue(
            any(
                np.array_equal(sample, visible_lab[100])
                for sample in samples
            )
        )

    def test_sampling_is_deterministic(self):
        visible_lab = np.arange(
            90_000,
            dtype=np.uint8,
        ).reshape(30_000, 3)

        edge_mask = np.zeros(30_000, dtype=bool)
        edge_mask[::5] = True

        first = _build_sample(
            visible_lab,
            edge_mask,
            10_000,
        )

        second = _build_sample(
            visible_lab,
            edge_mask,
            10_000,
        )

        np.testing.assert_array_equal(first, second)

    def test_processing_is_deterministic(self):
        source = make_test_image()

        first, _ = adaptive_color_preprocess(
            source,
            colors=2,
        )

        second, _ = adaptive_color_preprocess(
            source,
            colors=2,
        )

        self.assertEqual(first, second)

    def test_large_image_is_rejected(self):
        image = Image.new("RGB", (2001, 2000), "white")

        with self.assertRaises(ValueError):
            adaptive_color_preprocess(
                encode_image(image),
            )

    def test_invalid_color_count(self):
        for value in (1, 17, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    adaptive_color_preprocess(
                        make_test_image(),
                        colors=value,
                    )

    def test_invalid_smoothing(self):
        for value in (4, -1, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    adaptive_color_preprocess(
                        make_test_image(),
                        smoothing=value,
                    )

    def test_invalid_image_bytes(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(b"not an image")

    def test_empty_image_bytes(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess(b"")

    def test_non_bytes_input(self):
        with self.assertRaises(ValueError):
            adaptive_color_preprocess("not bytes")

    def test_upload_size_limit(self):
        oversized = b"x" * (5 * 1024 * 1024 + 1)

        with self.assertRaises(ValueError):
            adaptive_color_preprocess(oversized)


if __name__ == "__main__":
    unittest.main()
