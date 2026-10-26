
import sys
import traceback

import vtracer


def main():
    if len(sys.argv) != 3:
        print("Missing image format or tracing mode.", file=sys.stderr)
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

    try:
        print(
            f"Starting VTracer 1.0 API: mode={mode}, format={image_format}",
            file=sys.stderr,
            flush=True,
        )

        config = vtracer.Config()

        if mode == "binary":
            config = vtracer.Config.bw()

        svg = config.convert_bytes(image_data, format=image_format)

        if not isinstance(svg, str):
            svg = str(svg)

        if "<svg" not in svg[:2000].lower():
            print("VTracer returned invalid SVG output.", file=sys.stderr)
            return 1

        sys.stdout.write(svg)
        sys.stdout.flush()

        print("VTracer conversion completed.", file=sys.stderr, flush=True)
        return 0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
