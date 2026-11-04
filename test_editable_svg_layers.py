import unittest

from lxml import etree

from editable_svg_layers import create_editable_svg_layers


SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"


class TestEditableSVGLayers(unittest.TestCase):

    def test_creates_layer_for_svg_paths(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path id="one" fill="red" d="M0 0 L10 0 L10 10 Z"/>
            <path id="two" fill="red" d="M20 0 L30 0 L30 10 Z"/>
        </svg>
        """

        result, report = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))
        layers = root.findall(f"{{{SVG_NS}}}g")

        self.assertEqual(len(layers), 1)
        self.assertEqual(report["layers_created"], 1)
        self.assertEqual(report["paths_processed"], 2)

    def test_different_colors_create_separate_layers(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path fill="red" d="M0 0 L10 0 L10 10 Z"/>
            <path fill="blue" d="M20 0 L30 0 L30 10 Z"/>
        </svg>
        """

        result, report = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))
        layers = root.findall(f"{{{SVG_NS}}}g")

        self.assertEqual(len(layers), 2)
        self.assertEqual(report["layers_created"], 2)

    def test_layer_has_inkscape_attributes(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path fill="red" d="M0 0 L10 0 L10 10 Z"/>
        </svg>
        """

        result, _ = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))
        layer = root.find(f"{{{SVG_NS}}}g")

        self.assertEqual(
            layer.get(f"{{{INKSCAPE_NS}}}groupmode"),
            "layer",
        )

        self.assertIsNotNone(
            layer.get(f"{{{INKSCAPE_NS}}}label")
        )

    def test_path_geometry_is_preserved(self):
        path_data = "M0 0 C10 0 10 10 20 10 Z"

        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path fill="red" d="{path_data}"/>
        </svg>
        """

        result, _ = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))
        path = root.find(f".//{{{SVG_NS}}}path")

        self.assertEqual(path.get("d"), path_data)

    def test_drawing_order_is_preserved(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path id="first" fill="red" d="M0 0 L1 0 L1 1 Z"/>
            <path id="second" fill="blue" d="M2 0 L3 0 L3 1 Z"/>
            <circle id="middle" cx="5" cy="5" r="2"/>
            <path id="last" fill="green" d="M6 0 L7 0 L7 1 Z"/>
        </svg>
        """

        result, _ = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))

        ids = []

        for element in root:
            if element.tag.endswith("g"):
                ids.extend(
                    child.get("id")
                    for child in element
                    if child.tag.endswith("path")
                )
            else:
                ids.append(element.get("id"))

        self.assertEqual(
            ids,
            ["first", "second", "middle", "last"],
        )

    def test_style_fill_is_detected(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <path style="fill: #ff0000" d="M0 0 L1 0 L1 1 Z"/>
        </svg>
        """

        result, report = create_editable_svg_layers(svg)

        self.assertEqual(report["layers_created"], 1)
        self.assertIn("#ff0000", result)

    def test_empty_svg_is_rejected(self):
        with self.assertRaises(ValueError):
            create_editable_svg_layers("")

    def test_invalid_xml_is_rejected(self):
        with self.assertRaises(ValueError):
            create_editable_svg_layers("<svg><path></svg>")

    def test_script_is_rejected(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <script>alert(1)</script>
        </svg>
        """

        with self.assertRaises(ValueError):
            create_editable_svg_layers(svg)

    def test_generated_layer_ids_do_not_duplicate_existing_ids(self):
        svg = f"""
        <svg xmlns="{SVG_NS}">
            <g id="pixelforge-layer-0001"/>
            <path id="shape-1" fill="red" d="M0 0 L10 10"/>
        </svg>
        """

        result, report = create_editable_svg_layers(svg)

        root = etree.fromstring(result.encode("utf-8"))

        ids = [
            element.get("id")
            for element in root.iter()
            if element.get("id")
        ]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("pixelforge-layer-0001-1", ids)
        self.assertEqual(report["layers_created"], 1)


if __name__ == "__main__":
    unittest.main()
