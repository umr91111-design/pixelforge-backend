
import sys
import traceback

import vtracer


def main():
    if len(sys.argv) != 3:
        print("Missing image format or mode.", file=sys.stderr)
        return 2

    image_format = sys.argv[1].lower()
    mode = sys.argv[2].lower()

    if image_format not in {"png", "jpg", "webp"}:
        print("Unsupported image format.", file=sys.stderr)
        return 2

    if mode not in {"color", "binary"}:
        print("Unsupported tracing mode.", file=sys.stderr)
        return 2

    image_data = sys.stdin.buffer.read()

    if not image_data:
        print("No image data received.", file=sys.stderr)
        return 2

    print(
        f"Starting raw-byte VTracer: format={image_format}, mode={mode}",
        file=sys.stderr,
        flush=True,
    )

    try:
        svg = vtracer.convert_raw_image_to_svg(
            image_data,
            img_format=image_format,
            colormode=mode,
        )

        if isinstance(svg, str):
            svg = svg.encode("utf-8")

        if b"<svg" not in svg[:2000].lower():
            print("VTracer returned invalid SVG output.", file=sys.stderr)
            return 1

        sys.stdout.buffer.write(svg)
        sys.stdout.buffer.flush()

        print("VTracer conversion completed.", file=sys.stderr, flush=True)
        return 0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
