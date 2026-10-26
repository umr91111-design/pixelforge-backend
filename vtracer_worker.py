
import io
import sys
import traceback
import tempfile
import os

import vtracer
from PIL import Image


def main():
    if len(sys.argv) != 3:
        print("Usage: worker.py IMAGE_FORMAT MODE", file=sys.stderr)
        return 2

    image_format = sys.argv[1].lower()
    mode = sys.argv[2].lower()

    if image_format not in {"png", "jpg", "webp"}:
        print("Unsupported image format.", file=sys.stderr)
        return 2

    if mode not in {"color", "binary"}:
        print("Unsupported tracing mode.", file=sys.stderr)
        return 2

    raw_data = sys.stdin.buffer.read()
    if not raw_data:
        print("No image data received.", file=sys.stderr)
        return 2

    try:
        with Image.open(io.BytesIO(raw_data)) as source:
            source.load()
            image = source.convert("RGB")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.png")
            output_path = os.path.join(temp_dir, "output.svg")

            image.save(input_path, format="PNG")
            image.close()

            print(
                f"Tracing normalized PNG using mode={mode}",
                file=sys.stderr,
                flush=True,
            )

            vtracer.convert_image_to_svg_py(
                input_path,
                output_path,
                colormode=mode,
            )

            with open(output_path, "rb") as svg_file:
                svg_data = svg_file.read()

            if b"<svg" not in svg_data[:2000].lower():
                print("VTracer produced invalid SVG.", file=sys.stderr)
                return 1

            sys.stdout.buffer.write(svg_data)
            sys.stdout.buffer.flush()

        return 0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
