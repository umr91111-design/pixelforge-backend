
import os
import sys
import tempfile
import traceback

import vtracer


def main():
    if len(sys.argv) != 2:
        print("Missing image format.", file=sys.stderr)
        return 2

    image_format = sys.argv[1].lower()
    extensions = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}

    if image_format not in extensions:
        print("Unsupported image format.", file=sys.stderr)
        return 2

    image_data = sys.stdin.buffer.read()
    if not image_data:
        print("No image data received.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(
            temp_dir, "input" + extensions[image_format]
        )
        output_path = os.path.join(temp_dir, "output.svg")

        with open(input_path, "wb") as image_file:
            image_file.write(image_data)

        print("Starting file-based VTracer conversion.", file=sys.stderr)

        vtracer.convert_image_to_svg_py(input_path, output_path)

        with open(output_path, "rb") as svg_file:
            svg_data = svg_file.read()

        if not svg_data.strip().startswith(b"<"):
            print("VTracer did not produce SVG output.", file=sys.stderr)
            return 1

        sys.stdout.buffer.write(svg_data)
        sys.stdout.buffer.flush()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)

