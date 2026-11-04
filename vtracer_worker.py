"""PixelForge AI VTracer worker with configurable tracing presets."""

import sys
import traceback

from lxml import etree

from vtracer_config import build_vtracer_config


ALLOWED_FORMATS = {"png", "jpg", "webp"}
ALLOWED_MODES = {"color", "binary"}
ALLOWED_PRESETS = {
    "logo",
    "artwork",
    "line_art",
    "high_detail",
}

DEFAULT_PRESET = "artwork"

MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_SVG_BYTES = 20 * 1024 * 1024


def validate_svg(svg_text):
    """Validate that VTracer returned a well-formed SVG document."""

    if not isinstance(svg_text, str):
        raise ValueError("VTracer output is not text.")

    svg_bytes = svg_text.encode("utf-8")

    if not svg_bytes:
        raise ValueError("VTracer returned empty SVG output.")

    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError("Generated SVG exceeds the size limit.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
        remove_comments=False,
    )

    root = etree.fromstring(svg_bytes, parser=parser)

    root_name = etree.QName(root).localname.lower()

    if root_name != "svg":
        raise ValueError(
            "VTracer output does not have an SVG root element."
        )

    if root.getroottree().docinfo.doctype:
        raise ValueError(
            "SVG document type declarations are not allowed."
        )

    return svg_text


def main():
    if len(sys.argv) not in {3, 4}:
        print(
            "Expected image format, tracing mode, and optional preset.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    image_format = sys.argv[1].lower()
    mode = sys.argv[2].lower()

    preset_name = (
        sys.argv[3].strip().lower()
        if len(sys.argv) == 4
        else DEFAULT_PRESET
    )

    if image_format not in ALLOWED_FORMATS:
        print(
            "Unsupported image format.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if mode not in ALLOWED_MODES:
        print(
            "Unsupported tracing mode.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if preset_name not in ALLOWED_PRESETS:
        print(
            "Unsupported tracing preset.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    image_data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)

    if not image_data:
        print(
            "No image data received.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if len(image_data) > MAX_INPUT_BYTES:
        print(
            "Input image exceeds the size limit.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        print(
            (
                f"Starting VTracer: mode={mode}, "
                f"format={image_format}, preset={preset_name}"
            ),
            file=sys.stderr,
            flush=True,
        )

        config = build_vtracer_config(
            preset_name=preset_name,
            tracing_mode=mode,
        )

        svg_output = config.convert_bytes(
            image_data,
            format=image_format,
        )

        if isinstance(svg_output, bytes):
            svg_text = svg_output.decode("utf-8")
        elif isinstance(svg_output, str):
            svg_text = svg_output
        else:
            raise ValueError(
                "VTracer returned an unsupported output type."
            )

        svg_text = validate_svg(svg_text)

        # stdout must contain only the SVG document.
        sys.stdout.write(svg_text)
        sys.stdout.flush()

        print(
            (
                "VTracer conversion completed: "
                f"preset={preset_name}"
            ),
            file=sys.stderr,
            flush=True,
        )

        return 0

    except Exception as error:
        print(
            f"VTracer conversion failed: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
