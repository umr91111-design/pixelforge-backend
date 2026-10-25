
import io
import logging
import subprocess
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixelforge")

app = FastAPI(title="PixelForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://capable-chebakia-c33932.netlify.app",
        "https://playful-squirrel-c30122.netlify.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "pixelforge-api"}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pixelforge-api"}


@app.post("/api/trace")
async def trace_image(file: UploadFile = File(...)):
    allowed_types = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }

    image_format = allowed_types.get(file.content_type)
    if not image_format:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPG, or WebP image.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller.",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size

        if width * height > 4_000_000:
            raise HTTPException(
                status_code=413,
                detail="Image dimensions are too large. Please use a smaller image.",
            )

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    logger.info(
        "Starting isolated VTracer conversion: format=%s size=%s dimensions=%sx%s",
        image_format, len(data), width, height,
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "vtracer_worker.py",
                image_format,
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            logger.error(
    "VTracer worker failed: returncode=%s stdout=%r stderr=%r",
    result.returncode,
    result.stdout[:500],
    result.stderr[:3000],
)
            raise HTTPException(
                status_code=500,
                detail="Image conversion failed. Please try a simpler image.",
            )

        svg = result.stdout

        if not svg.strip().startswith(b"<"):
            logger.error("VTracer worker returned invalid SVG output.")
            raise HTTPException(
                status_code=500,
                detail="Conversion did not produce a valid SVG file.",
            )

        logger.info("VTracer conversion completed: output=%s bytes", len(svg))

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge.svg"'
            },
        )

    except subprocess.TimeoutExpired:
        logger.error("VTracer worker timed out after 30 seconds.")
        raise HTTPException(
            status_code=504,
            detail="Conversion took too long. Please try a smaller image.",
        )


            
