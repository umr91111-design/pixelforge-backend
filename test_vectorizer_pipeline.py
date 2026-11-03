import unittest
from unittest.mock import patch

from lxml import etree
from svgpathtools import Line, Path

from vectorizer_pipeline import (
    MAX_SEGMENTS_PER_PATH,
    refine_bezier_paths,
    validate_and_inspect_svg,
)


def make_complex_path():
    """Create a path exceeding the configured segment limit."""
    segments = []

    for index in range(MAX_SEGMENTS_PER_PATH + 1):
        segments.append(
            Line(
                complex(index, 0),
                complex(index + 1, 0),
            )
        )

    return Path(*segments)


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
        self.assertEqual(report["invalid_geometry_paths"], 1)
        self.assertNotIn("C", result)

    def test_shared_edge_paths_are_preserved_but_unrelated_path_is_refined(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="700" height="200"
             viewBox="0 0 700 200">

            <path id="left"
                  d="M 0 0 L 100 0 L 200 20
                     L 200 120 L 0 120 Z"
                  fill="black"/>

            <path id="right"
                  d="M 200 20 L 300 0 L 400 20
                     L 400 120 L 200 120 Z"
                  fill="red"/>

            <path id="independent"
                  d="M 450 0 L 550 0 L 643.97 34.2
                     L 643.97 134.2 L 450 134.2 Z"
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

        self.assertEqual(report["shared_edge_paths"], 2)
        self.assertEqual(report["protected_shared_edge_paths"], 2)

        self.assertNotIn("C", paths["left"])
        self.assertNotIn("C", paths["right"])
        self.assertIn("C", paths["independent"])

    def test_shared_edge_is_detected_in_opposite_direction(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="300" height="150">

            <path id="first"
                  d="M 0 0 L 100 0 L 200 20
                     L 200 120 L 0 120 Z"/>

            <path id="second"
                  d="M 200 20 L 300 20 L 300 120
                     L 200 120 Z"/>
        </svg>
        """

        result, report = refine_bezier_paths(svg)

        self.assertEqual(report["shared_edge_paths"], 2)
        self.assertEqual(report["protected_shared_edge_paths"], 2)

        root = etree.fromstring(result.encode("utf-8"))

        paths = {
            element.get("id"): element.get("d")
            for element in root.iter()
            if isinstance(element.tag, str)
            and etree.QName(element).localname == "path"
        }

        self.assertNotIn("C", paths["first"])
        self.assertNotIn("C", paths["second"])


class TestSvgPathValidation(unittest.TestCase):

    def test_script_elements_are_rejected(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <script>alert('unsafe')</script>
            <path d="M 0 0 L 10 10 Z"/>
        </svg>
        """

        with self.assertRaises(ValueError):
            validate_and_inspect_svg(svg)

    def test_event_handler_attributes_are_rejected(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg"
             onload="alert('unsafe')">
            <path d="M 0 0 L 10 10 Z"/>
        </svg>
        """

        with self.assertRaises(ValueError):
            validate_and_inspect_svg(svg)

    def test_foreign_object_elements_are_rejected(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <foreignObject width="100" height="100">
                <div>Unsafe embedded content</div>
            </foreignObject>
        </svg>
        """

        with self.assertRaises(ValueError):
            validate_and_inspect_svg(svg)

    def test_malformed_path_data_is_counted_as_invalid(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 0 L 10"/>
        </svg>
        """

        report = validate_and_inspect_svg(svg)

        self.assertEqual(report["path_count"], 1)
        self.assertEqual(report["invalid_paths"], 1)
        self.assertEqual(report["inspected_paths"], 0)

    def test_non_finite_coordinates_are_rejected_during_inspection(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 0 L 10 10 Z"/>
        </svg>
        """

        invalid_path = Path(
            Line(
                0 + 0j,
                complex(float("nan"), 10),
            )
        )

        with patch(
            "vectorizer_pipeline.parse_path",
            return_value=invalid_path,
        ):
            report = validate_and_inspect_svg(svg)

        self.assertEqual(report["path_count"], 1)
        self.assertEqual(report["invalid_paths"], 1)
        self.assertEqual(report["inspected_paths"], 0)

    def test_non_finite_coordinates_are_not_refined(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 0 L 10 10 L 20 20 Z"/>
        </svg>
        """

        invalid_path = Path(
            Line(0 + 0j, 10 + 10j),
            Line(10 + 10j, complex(float("nan"), 20)),
            Line(complex(float("nan"), 20), 0 + 0j),
        )

        with patch(
            "vectorizer_pipeline.parse_path",
            return_value=invalid_path,
        ):
            result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertEqual(report["invalid_geometry_paths"], 1)
        self.assertEqual(report["skipped_paths"], 1)
        self.assertNotIn("C", result)

    def test_excessive_segments_are_skipped_during_inspection(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 0 L 10 10 Z"/>
        </svg>
        """

        complex_path = make_complex_path()

        with patch(
            "vectorizer_pipeline.parse_path",
            return_value=complex_path,
        ), patch(
            "vectorizer_pipeline._path_has_finite_coordinates"
        ) as coordinate_check:

            report = validate_and_inspect_svg(svg)

        self.assertEqual(report["path_count"], 1)
        self.assertEqual(report["inspected_paths"], 0)
        self.assertEqual(report["skipped_paths"], 1)
        self.assertEqual(report["invalid_paths"], 0)
        coordinate_check.assert_not_called()

    def test_excessive_segments_are_skipped_during_refinement(self):
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 0 L 10 10 Z"/>
        </svg>
        """

        complex_path = make_complex_path()

        with patch(
            "vectorizer_pipeline.parse_path",
            return_value=complex_path,
        ), patch(
            "vectorizer_pipeline._path_has_finite_coordinates"
        ) as coordinate_check:

            result, report = refine_bezier_paths(svg)

        self.assertEqual(report["refined_paths"], 0)
        self.assertEqual(report["refined_corners"], 0)
        self.assertEqual(report["skipped_paths"], 1)
        self.assertNotIn("C", result)
        coordinate_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
