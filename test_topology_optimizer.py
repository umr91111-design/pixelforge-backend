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
            svg, gap_tolerance=1.5
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
            svg, gap_tolerance=1.5
        )

        self.assertEqual(result["nearby_gaps_found"], 0)
        self.assertEqual(result["repair_candidates_found"], 0)

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
        svg = make_svg(square("a", 0, 0, 10, 10))

        with self.assertRaises(ValueError):
            analyze_svg_topology(svg, gap_tolerance=-1)

    def test_rejects_excessive_gap_tolerance(self):
        svg = make_svg(square("a", 0, 0, 10, 10))

        with self.assertRaises(ValueError):
            analyze_svg_topology(svg, gap_tolerance=101)

    def test_creates_overlap_review_candidate(self):
        svg = make_svg(
            square("logo-a", 0, 0, 10, 10),
            square("logo-b", 5, 5, 15, 15),
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["repair_candidates_found"], 1)

        candidate = result["repair_candidates"][0]

        self.assertEqual(candidate["type"], "overlap_review")
        self.assertEqual(
            candidate["path_a"]["element_id"], "logo-a"
        )
        self.assertEqual(
            candidate["path_b"]["element_id"], "logo-b"
        )
        self.assertFalse(
            candidate["automatic_repair_allowed"]
        )

    def test_creates_gap_review_candidate(self):
        svg = make_svg(
            square("left", 0, 0, 10, 10),
            square("right", 11, 0, 21, 10),
        )

        result = analyze_svg_topology(
            svg, gap_tolerance=1.5
        )

        self.assertEqual(result["repair_candidates_found"], 1)

        candidate = result["repair_candidates"][0]

        self.assertEqual(candidate["type"], "gap_review")
        self.assertEqual(
            candidate["path_a"]["element_id"], "left"
        )
        self.assertEqual(
            candidate["path_b"]["element_id"], "right"
        )
        self.assertFalse(
            candidate["automatic_repair_allowed"]
        )

    def test_shared_boundary_creates_no_repair_candidate(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 10, 0, 20, 10),
        )

        result = analyze_svg_topology(svg)

        self.assertEqual(result["shared_boundaries_found"], 1)
        self.assertEqual(result["repair_candidates_found"], 0)

    def test_repair_planning_is_enabled_but_auto_repair_is_disabled(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 5, 5, 15, 15),
        )

        result = analyze_svg_topology(svg)

        self.assertTrue(result["repair_planning_enabled"])
        self.assertFalse(result["automatic_repair_enabled"])

    def test_does_not_modify_svg(self):
        svg = make_svg(
            square("a", 0, 0, 10, 10),
            square("b", 5, 5, 15, 15),
        )

        result = analyze_svg_topology(svg)

        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["geometry_modified"])

    def test_zero_gap_tolerance_is_allowed(self):
        svg = make_svg(square("a", 0, 0, 10, 10))

        result = analyze_svg_topology(svg, gap_tolerance=0)

        self.assertEqual(result["gap_tolerance"], 0.0)


if __name__ == "__main__":
    unittest.main()
