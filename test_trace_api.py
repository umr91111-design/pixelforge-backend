
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

import main


TEST_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
width="100" height="100" viewBox="0 0 100 100">
<path d="M 0 0 L 100 0 L 100 100 L 0 100 Z"
fill="black"/>
</svg>"""

REFINED_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
width="100" height="100" viewBox="0 0 100 100">
<path d="M 0 0 C 30 0 70 0 100 0 L 100 100 L 0 100 Z"
fill="black"/>
</svg>"""


def make_test_image():
    buffer = io.BytesIO()

    image = Image.new("RGB", (10, 10), "white")
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def make_upload():
    return UploadFile(
        filename="test.png",
        file=io.BytesIO(make_test_image()),
        headers=Headers({"content-type": "image/png"}),
    )


class TestTraceAPIBezierIntegration(unittest.IsolatedAsyncioTestCase):

    async def test_trace_api_runs_bezier_refinement(self):
        worker_result = SimpleNamespace(
            returncode=0,
            stdout=TEST_SVG.encode("utf-8"),
            stderr=b"",
        )

        fake_stats = {
            "refined_paths": 1,
            "refined_corners": 2,
            "skipped_paths": 0,
        }

        with (
            patch("main.run_worker", return_value=worker_result),
            patch("main.run_pipeline_analysis", return_value={}),
            patch(
                "main.refine_bezier_paths",
                return_value=(REFINED_SVG, fake_stats),
            ) as bezier_mock,
        ):
            response = await main.trace_image(
                image=make_upload(),
                mode="color",
                recognize_primitives=False,
                refine_geometry=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/svg+xml")

        self.assertEqual(
            response.headers["X-PixelForge-Bezier-Refinement"],
            "completed",
        )

        self.assertEqual(
            response.headers["X-PixelForge-Bezier-Paths"],
            "1",
        )

        self.assertEqual(
            response.headers["X-PixelForge-Bezier-Corners"],
            "2",
        )

        self.assertIn(b"<svg", response.body)
        self.assertIn(b"100 100", response.body)

        main.validate_svg(response.body.decode("utf-8"))
        bezier_mock.assert_called_once()

    async def test_trace_api_keeps_original_svg_if_bezier_fails(self):
        worker_result = SimpleNamespace(
            returncode=0,
            stdout=TEST_SVG.encode("utf-8"),
            stderr=b"",
        )

        with (
            patch("main.run_worker", return_value=worker_result),
            patch("main.run_pipeline_analysis", return_value={}),
            patch(
                "main.refine_bezier_paths",
                side_effect=ValueError("Test refinement failure"),
            ),
        ):
            response = await main.trace_image(
                image=make_upload(),
                mode="color",
                recognize_primitives=False,
                refine_geometry=True,
            )

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.headers["X-PixelForge-Bezier-Refinement"],
            "failed",
        )

        self.assertEqual(
            response.body,
            main.validate_svg(TEST_SVG),
        )

        main.validate_svg(response.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
