import io
import unittest

import numpy as np
from PIL import Image

from intelligent_denoising import remove_image_noise


def encode_image(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_image(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGBA"))


class TestIntelligentDenoising(unittest.TestCase):

    def test_output_is_valid_png(self):
        image = np.full((32, 32, 3), 255, dtype=np.uint8)

        output, report = remove_image_noise(
            encode_image(image)
        )

        result = decode_image(output)

        self.assertEqual(result.shape, (32, 32, 4))
        self.assertEqual(report["width"], 32)
        self.assertEqual(report["height"], 32)

    def test_isolated_speckle_is_removed(self):
        image = np.full(( nine :=  nine,  nine, 3), 255, dtype=np.uint8)
        image[4, 4] = (0, 0, 0)

        output, report = remove_image_noise(
            encode_image(image),
            strength="balanced",
        )

        result = decode_image(output)

        self.assertGreater(report["pixels_replaced"], 0)
        self.assertGreater(
            int(np.mean(result[4, 4, :3])),
            200,
        )

    def test_connected_thin_line_is_preserved(self):
        image = np.full((15, 15, 3), 255, dtype=np.uint8)
        image[7, 3:12] = (0, 0, 0)

        output, report = remove_image_noise(
            encode_image(image),
            strength="balanced",
        )

        result = decode_image(output)

        for x in range(3, 12):
            self.assertLess(
                int(np.mean(result[7, x, :3])),
                50,
            )

        self.assertEqual(report["pixels_replaced"], 0)

    def test_transparency_is_preserved(self):
        image = np.zeros((16, 16, 4), dtype=np.uint8)

        image[:, :, :3] = 255
        image[:, :, 3] = 0

        image[4:12, 4:12, 3] = 255
        image[4:12, 4:12, :3] = 0

        output, report = remove_image_noise(
            encode_image(image)
        )

        result = decode_image(output)

        np.testing.assert_array_equal(
            result[:, :, 3],
            image[:, :, 3],
        )

        self.assertTrue(report["alpha_preserved"])

    def test_transparent_image_is_supported(self):
        image = np.zeros((12, 12, 4), dtype=np.uint8)

        output, report = remove_image_noise(
            encode_image(image)
        )

        result = decode_image(output)

        self.assertTrue(report["transparent_image"])
        self.assertEqual(report["pixels_replaced"], 0)
        self.assertTrue(np.all(result[:, :, 3] == 0))

    def test_image_dimensions_are_preserved(self):
        image = np.full((20, 35, 3), 200, dtype=np.uint8)

        output, _ = remove_image_noise(
            encode_image(image)
        )

        result = decode_image(output)

        self.assertEqual(result.shape[:2], (20, 35))

    def test_processing_is_deterministic(self):
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        image[10, 10] = (0, 0, 0)

        source = encode_image(image)

        first, _ = remove_image_noise(source)
        second, _ = remove_image_noise(source)

        self.assertEqual(first, second)

    def test_all_strengths_are_supported(self):
        image = np.full((16, 16, 3), 255, dtype=np.uint8)
        source = encode_image(image)

        for strength in ("light", "balanced", "strong"):
            output, report = remove_image_noise(
                source,
                strength=strength,
            )

            self.assertTrue(output)
            self.assertEqual(report["strength"], strength)

    def test_invalid_strength_is_rejected(self):
        image = np.full((8, 8, 3), 255, dtype=np.uint8)

        with self.assertRaises(ValueError):
            remove_image_noise(
                encode_image(image),
                strength="extreme",
            )

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            remove_image_noise(b"")

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            remove_image_noise(b"not an image")


if __name__ == "__main__":
    unittest.main()
