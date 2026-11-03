
import unittest

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


if __name__ == "__main__":
    unittest.main()
