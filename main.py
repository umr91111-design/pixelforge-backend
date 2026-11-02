import asyncio
import io
import logging
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from shape_detector import detect_shapes, refine_svg_with_primitives

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixelforge")

BASE_DIR = Path(__file__).resolve().parent
WORKER_PATH = BASE_DIR / "vtracer_worker.py"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_SVG_BYTES = 20 * 1024 * 1024
WORKER_TIMEOUT = 60
MAX_CONCURRENT_CONVERSIONS = 2

conversion_slots = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)

ALLOWED_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}

MODE_ALIASES = {
    "color": "color",
    "binary": "binary",
    "black-white": "binary",
    "black_white": "binary",
    "bw": "binary",
}

ALLOWED_ORIGINS = [
    "https://capable-chebakia-c33932.netlify.app",
    "https://playful-squirrel-c30122.netlify.app",
    "https://umr91111-design.github.io",
]

app = FastAPI(
    title="PixelForge API",
    description=(
        "PixelForge image vectorization API. Current processing uses VTracer "
        "and OpenCV contour-based primitive recognition. No trained deep-learning "
        "model or topology-aware vector graph optimizer is bundled."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
# FastAPI currently defaults to OpenAPI 3.1. Set the advertised schema version
# for clients requiring OpenAPI 3.0.x compatibility.
app.openapi_version = "3.0.3"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return Response("Invalid Content-Length", status_code=400)
        if size > MAX_UPLOAD_BYTES + 256 * 1024:
            return Response("Request too large", status_code=413)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    # FastAPI documentation pages need their Swagger UI / ReDoc CDN assets.
    # Keep the restrictive policy on all API routes.
    if request.url.path in {"/docs", "/redoc"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "img-src 'self' data: https://cdn.jsdelivr.net; "
            "font-src 'self' data: https://fonts.gstatic.com https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
    return response


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "pixelforge-api",
        "version": app.version,
        "openapi": "/openapi.json",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "pixelforge-api",
        "features": {
            "vtracer": True,
            "opencv_primitive_detection": True,
            "svg_recognition_metadata": True,
            "deep_learning_model": False,
            "topology_aware_graph_optimizer": False,
            "openapi": "/openapi.json",
        },
    }


@app.get("/api/features")
async def features():
    return {
        "vectorization": {
            "engine": "VTracer",
            "modes": ["color", "binary"],
        },
        "primitive_recognition": {
            "engine": "OpenCV contour analysis",
            "shapes": [
                "circle", "ellipse", "rectangle", "square",
                "triangle", "star", "polygon",
            ],
            "deep_learning": False,
        },
        "geometry_refinement": {
            "status": "recognition metadata only",
            "topology_aware_shared_boundary_optimization": False,
        },
        "developer_api": {
            "openapi": "/openapi.json",
            "schema_version": app.openapi_version,
        },
    }


def validate_image(data: bytes):
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(
                    status_code=400,
                    detail="Please upload a PNG, JPG, or WebP image.",
                )
            width, height = img.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large. Please use a smaller image.",
                )
            img.verify()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from exc
    return width, height


async def read_image_upload(image: UploadFile):
    image_format = ALLOWED_TYPES.get(image.content_type or "")
    if not image_format:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPG, or WebP image.",
        )

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller.",
        )

    width, height = validate_image(data)
    return data, image_format, width, height


async def acquire_slot():
    try:
        await asyncio.wait_for(conversion_slots.acquire(), timeout=2)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="The service is busy. Please try again shortly.",
        ) from exc


@app.post("/api/detect-shapes")
async def detect_image_shapes(image: UploadFile = File(...)):
    try:
        data, _, width, height = await read_image_upload(image)
        await acquire_slot()
        try:
            result = await asyncio.to_thread(detect_shapes, data)
        finally:
            conversion_slots.release()

        result["original_image_width"] = width
        result["original_image_height"] = height
        return result

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to process this image.",
        ) from exc
    except Exception as exc:
        logger.exception("Shape detection failed.")
        raise HTTPException(
            status_code=500,
            detail="Shape detection failed.",
        ) from exc
    finally:
        await image.close()


def run_worker(image_format: str, mode: str, data: bytes):
    return subprocess.run(
        [
            sys.executable,
            str(WORKER_PATH),
            image_format,
            mode,
        ],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=WORKER_TIMEOUT,
        check=False,
        cwd=str(BASE_DIR),
        shell=False,
    )


@app.post("/api/trace")
async def trace_image(
    image: UploadFile = File(...),
    mode: str = Form("color"),
    recognize_primitives: bool = Form(True),
    refine_geometry: bool = Form(True),
):
    try:
        selected_mode = MODE_ALIASES.get(mode.strip().lower())
        if not selected_mode:
            raise HTTPException(
                status_code=400,
                detail="Choose Color Artwork or Black & White.",
            )

        data, image_format, width, height = await read_image_upload(image)

        logger.info(
            "Vectorization request: mode=%s format=%s dimensions=%sx%s",
            selected_mode, image_format, width, height,
        )

        await acquire_slot()
        started = time.monotonic()
        detected_shapes = []
        try:
            try:
                result = await asyncio.to_thread(
                    run_worker, image_format, selected_mode, data
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning("VTracer worker timed out.")
                raise HTTPException(
                    status_code=504,
                    detail="Conversion took too long. Please try a smaller image.",
                ) from exc
            except OSError as exc:
                logger.exception("Could not start VTracer worker.")
                raise HTTPException(
                    status_code=500,
                    detail="Image conversion is temporarily unavailable.",
                ) from exc

            if result.returncode != 0:
                logger.error(
                    "VTracer failed with exit code %s: %s",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace")[-2000:],
                )
                raise HTTPException(
                    status_code=500,
                    detail="Image conversion failed.",
                )

            svg_bytes = result.stdout
            if not svg_bytes or len(svg_bytes) > MAX_SVG_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Generated SVG is empty or too large. Try a simpler image.",
                )

            try:
                svg_text = svg_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Vectorizer returned invalid SVG encoding.",
                ) from exc

            if "<svg" not in svg_text[:4000].lower():
                logger.error("VTracer returned an invalid SVG response.")
                raise HTTPException(
                    status_code=500,
                    detail="Conversion did not produce a valid SVG file.",
                )

            if recognize_primitives:
                try:
                    detection = await asyncio.to_thread(detect_shapes, data)
                    detected_shapes = detection.get("shapes", [])
                except Exception:
                    # Primitive detection is an enhancement, not a reason to
                    # lose a valid VTracer result.
                    logger.exception("Primitive recognition failed; using VTracer SVG.")

            if refine_geometry and detected_shapes:
                try:
                    svg_text = await asyncio.to_thread(
                        refine_svg_with_primitives,
                        svg_text,
                        detected_shapes,
                    )
                except Exception:
                    logger.exception("SVG refinement failed; retaining VTracer output.")

            elapsed = round(time.monotonic() - started, 3)
            logger.info(
                "VTracer completed: output=%s bytes shapes=%s elapsed=%ss",
                len(svg_text.encode("utf-8")),
                len(detected_shapes),
                elapsed,
            )

            final_svg = svg_text.encode("utf-8")
            if len(final_svg) > MAX_SVG_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Generated SVG is too large after processing.",
                )

            return Response(
                content=final_svg,
                media_type="image/svg+xml",
                headers={
                    "Content-Disposition": 'attachment; filename="pixelforge.svg"',
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "X-PixelForge-Detected-Shapes": str(len(detected_shapes)),
                    "X-PixelForge-Deep-Learning": "not-configured",
                },
            )
        finally:
            conversion_slots.release()

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected VTracer error.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected conversion error occurred.",
        ) from exc
    finally:
        await image.close()
