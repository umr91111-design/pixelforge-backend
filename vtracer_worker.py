
import sys

import vtracer


def main():
    if len(sys.argv) != 2:
        print("Missing image format argument.", file=sys.stderr)
        return 2

    image_format = sys.argv[1]
    if image_format not in {"png", "jpg", "webp"}:
        print("Unsupported image format.", file=sys.stderr)
        return 2

    image_data = sys.stdin.buffer.read()
    if not image_data:
        print("No image data received.", file=sys.stderr)
        return 2

    svg = vtracer.convert_raw_image_to_svg(
        image_data,
        img_format=image_format,
    )

    sys.stdout.buffer.write(svg.encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"VTracer worker error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
