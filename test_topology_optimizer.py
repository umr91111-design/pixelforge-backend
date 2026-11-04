
import unittest

from topology_optimizer import analyze_svg_topology


def make_svg(*paths):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + "".join(paths)
        + "</svg>"
    )


def square(path_id, x1, y1, x2, y2):
    return (
        f'<path id="{path_id}" '
        f'd="M{x1} {y1} L{x2} {y1} '
        f'L{x2} {y2} L{x1} {y2} Z"/>'
    )


class TestTopologyOptimizer(unittest.TestCase):

    def test_detects_overlapping_polygons(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 5, 5, 15, 15),
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["overlaps_found"], 1)
        self.assertGreater(
            result["overlaps"][0]["overlap_area"], 0
        )

    def test_detects_shared_boundaries(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 10, 0, 20, 10),
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(
            result["shared_boundaries_found"], 1
        )
        self.assertAlmostEqual(
            result["shared_boundaries"][0]["shared_length"],
            10.0,
        )

    def test_detects_small_gaps(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 11, 0, 21, 10),
        )

        result = analyze_svg_topology(
            svg,
            gap_tolerance=1.5,
        )

        self.assertEqual(result["nearby_gaps_found"], 1)
        self.assertAlmostEqual(
            result["nearby_gaps"][0]["distance"],
            1.0,
        )

    def test_ignores_gaps_outside_tolerance(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 15, 0, 25, 10),
        )

        result = analyze_svg_topology(
            svg,
            gap_tolerance=1.5,
        )

        self.assertEqual(result["nearby_gaps_found"], 0)

    def test_skips_open_paths(self):
        svg = make_svg(
            '<path d="M0 0 L10 0 L10 10"/>'
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["valid_polygons"], 0)
        self.assertEqual(result["skipped_paths"], 1)

    def test_skips_curved_paths(self):
        svg = make_svg(
            '<path d="M0 0 C10 0 10 10 20 10 Z"/>'
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["valid_polygons"], 0)
        self.assertEqual(result["skipped_paths"], 1)

    def test_skips_transformed_paths(self):
        svg = make_svg(
            '<g transform="translate(10,10)">'
            + square("a", 0, 0, 10, 10)
            + "</g>"
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["valid_polygons"], 0)
        self.assertEqual(result["skipped_paths"], 1)

    def test_rejects_script_elements(self):
        svg = make_svg(
            '<script>alert(1)</script>',
            square("a", 0, 0, 10, 10),
        )

        with self.assertRaises(ValueError):
            analyze_svg_topology(svg)

    def test_rejects_invalid_svg(self):
        with self.assertRaises(ValueError):
            analyze_svg_topology("<not-svg>")

    def test_rejects_negative_gap_tolerance(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10)
        )

        with self.assertRaises(ValueError):
            analyze_svg_topology(svg, gap_tolerance=-1)

    def test_does_not_modify_svg(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 5, 5, 15, 15),
        )

        result = analyze_svg_topology(svg)

        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["geometry_modified"])


if __name__ == "__main__":
    unittest.main()
