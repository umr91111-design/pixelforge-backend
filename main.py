
import asyncio
import io
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from shape_detector import detect_shapes 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixelforge")

app = FastAPI(
    title="PixelForge API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    # Reject oversized declared requests early.
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return Response("Invalid Content-Length", status_code=400)

        # Multipart data includes extra bytes beyond the image itself.
        if size > MAX_UPLOAD_BYTES + 256 * 1024:
            return Response("Request too large", status_code=413)

    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

    return response


@app.get("/")
async def root():
    return {"status": "ok", "service": "pixelforge-api"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "pixelforge-api"}
    @app.post("/api/detect-shapes")
async def detect_image_shapes(image: UploadFile = File(...)):
    try:
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
                detail="The uploaded image is empty.",
            )

        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 5 MB or smaller.",
            )

        validate_image(data)

        try:
            await asyncio.wait_for(conversion_slots.acquire(), timeout=2)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=429,
                detail="The service is busy. Please try again shortly.",
            )

        try:
            result = await asyncio.to_thread(detect_shapes, data)
        finally:
            conversion_slots.release()

        return result

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Unable to process this image.",
        )
    except Exception:
        logger.exception("Shape detection failed.")
        raise HTTPException(
            status_code=500,
            detail="Shape detection failed.",
        )
    finally:
        await image.close()


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
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    return width, height


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
):
    try:
        image_format = ALLOWED_TYPES.get(image.content_type or "")

        if not image_format:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PNG, JPG, or WebP image.",
            )

        selected_mode = MODE_ALIASES.get(mode.strip().lower())

        if not selected_mode:
            raise HTTPException(
                status_code=400,
                detail="Choose Color Artwork or Black & White.",
            )

        # Read at most one byte beyond the limit.
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

        logger.info(
            "VTracer request: mode=%s format=%s dimensions=%sx%s",
            selected_mode,
            image_format,
            width,
            height,
        )

        # Bound simultaneous conversions.
        try:
            await asyncio.wait_for(
                conversion_slots.acquire(),
                timeout=2,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=429,
                detail="The service is busy. Please try again shortly.",
            )

        try:
            result = await asyncio.to_thread(
                run_worker,
                image_format,
                selected_mode,
                data,
            )
        except subprocess.TimeoutExpired:
            logger.warning("VTracer worker timed out.")
            raise HTTPException(
                status_code=504,
                detail="Conversion took too long. Please try a smaller image.",
            )
        except OSError:
            logger.exception("Could not start VTracer worker.")
            raise HTTPException(
                status_code=500,
                detail="Image conversion is temporarily unavailable.",
            )
        finally:
            conversion_slots.release()

        if result.returncode != 0:
            # Do not log uploaded content or raw subprocess output.
            logger.error(
                "VTracer failed with exit code %s",
                result.returncode,
            )
            raise HTTPException(
                status_code=500,
                detail="Image conversion failed.",
            )

        svg = result.stdout

        if len(svg) > MAX_SVG_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Generated SVG is too large. Try a simpler image.",
            )

        if b"<svg" not in svg[:2000].lower():
            logger.error("VTracer returned an invalid SVG response.")
            raise HTTPException(
                status_code=500,
                detail="Conversion did not produce a valid SVG file.",
            )

        logger.info("VTracer completed: output=%s bytes", len(svg))

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge.svg"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected VTracer error.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected conversion error occurred.",
        )
    finally:
        await image.close()
