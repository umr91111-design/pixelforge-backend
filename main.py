
import io
import logging

import vtracer
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixelforge")

app = FastAPI(title="PixelForge AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://playful-squirrel-c30122.netlify.app",
        "https://capable-chebakia-c33932.netlify.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_PIXELS = 4_000_000

ALLOWED_TYPES = {
    "image/png": ("png", "PNG"),
    "image/jpeg": ("jpg", "JPEG"),
    "image/webp": ("webp", "WEBP"),
}


@app.get("/")
def home():
    return {
        "name": "PixelForge AI API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pixelforge-api"}


@app.post("/api/trace")
async def trace_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PNG, JPG, or WebP image.",
            )

        data = await file.read(MAX_FILE_SIZE + 1)

        if not data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 5 MB.",
            )

        image_format, expected_format = ALLOWED_TYPES[file.content_type]

        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != expected_format:
                    raise HTTPException(
                        status_code=400,
                        detail="The image contents do not match the file type.",
                    )

                width, height = image.size

                if width * height > MAX_PIXELS:
                    raise HTTPException(
                        status_code=400,
                        detail="Image dimensions are too large. Use an image with at most 4 million pixels.",
                    )

                image.load()

        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            logger.warning("Image validation failed: %s", e)
            raise HTTPException(
                status_code=400,
                detail="This image is invalid or damaged. Try another PNG, JPG, or WebP image.",
            )

        logger.info(
            "Starting VTracer: format=%s size=%s bytes dimensions=%sx%s",
            image_format,
            len(data),
            width,
            height,
        )

        svg = vtracer.convert_raw_image_to_svg(
            data,
            img_format=image_format,
        )

        if isinstance(svg, bytes):
            svg = svg.decode("utf-8")

        if not svg.lstrip().startswith("<?xml") and "<svg" not in svg[:500]:
            raise ValueError("VTracer did not return valid SVG content")

        logger.info("VTracer conversion completed successfully")

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge.svg"',
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("VTracer conversion failed")
        raise HTTPException(
            status_code=500,
            detail="Image conversion failed. Please try a smaller or simpler image.",
        )
    finally:
        await file.close()


                
