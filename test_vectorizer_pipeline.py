import unittest

from lxml import etree
from svgpathtools import parse_path

from vectorizer_pipeline import (
    refine_bezier_paths,
    validate_and_inspect_svg,
)


class TestBezierRefinement(unittest.TestCase):

    def test_shallow_corner_is_refined(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="200" height="120"
             viewBox="0 0 200 120">
            <path d="M 0 0 L 100 0 L 200 20
                     L 200 120 L 0 120 Z"
                  fill="black"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertGreater(report["refined_paths"], 0)
        self.assertGreater(report["refined_corners"], 0)
        validate_and_inspect_svg(result)
        self.assertIn("C", result)

    def test_sharp_rectangle_is_preserved(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="100" height="100">
            <path d="M 0 0 L 100 0 L 100 100
                     L 0 100 Z"
                  fill="black"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertEqual(report["refined_corners"], 0)
        self.assertNotIn("C", result)

    def test_existing_curve_is_preserved(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="100" height="100">
            <path d="M 0 50 C 20 0 80 0 100 50"
                  fill="none"
                  stroke="black"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertIn("C", result)

    def test_invalid_svg_is_rejected(self):
        with self.assertRaises(ValueError):
            refine_bezier_paths("<not-an-svg>")

    def test_very_short_edge_is_preserved(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="120" height="120">
            <path d="M 0 0 L 100 0 L 101 1
                     L 100 100 L 0 100 Z"
                  fill="black"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertEqual(report["refined_corners"], 0)
        self.assertNotIn("C", result)

    def test_self_intersecting_path_is_not_refined(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="100" height="100">
            <path d="M 0 0 L 100 100
                     L 0 100 L 100 0 Z"
                  fill="black"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertEqual(report["refined_corners"], 0)
        self.assertNotIn("C", result)

    def test_shared_edge_paths_are_preserved_but_unrelated_path_is_refined(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="500" height="200"
             viewBox="0 0 500 200">

            <path id="left"
                  d="M 0 0 L 100 0 L 100 100 L 0 100 Z"
                  fill="black"/>

            <path id="right"
                  d="M 100 0 L 200 0 L 200 100 L 100 100 Z"
                  fill="red"/>

            <path id="independent"
                  d="M 300 0 L 400 0 L 500 20
                     L 500 120 L 300 120 Z"
                  fill="blue"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        validate_and_inspect_svg(result)

        root = etree.fromstring(result.encode("utf-8"))

        paths = {
            element.get("id"): element.get("d")
            for element in root.iter()
            if isinstance(element.tag, str)
            and etree.QName(element).localname == "path"
        }

        original_paths = {
            "left": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "right": "M 100 0 L 200 0 L 200 100 L 100 100 Z",
        }

        for path_id, original_data in original_paths.items():
            original = parse_path(original_data)
            actual = parse_path(paths[path_id])

            self.assertEqual(
                len(actual),
                len(original),
                f"{path_id} segment count changed",
            )

            for actual_segment, original_segment in zip(actual, original):
                self.assertEqual(
                    type(actual_segment),
                    type(original_segment),
                    f"{path_id} segment type changed",
                )

                self.assertLessEqual(
                    abs(actual_segment.start - original_segment.start),
                    1e-6,
                    f"{path_id} segment start changed",
                )

                self.assertLessEqual(
                    abs(actual_segment.end - original_segment.end),
                    1e-6,
                    f"{path_id} segment end changed",
                )

        self.assertIn("C", paths["independent"])


if __name__ == "__main__":
    unittest.main()
