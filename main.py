
import io
import logging
import subprocess
import sys
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from rembg import remove, new_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixelforge")

app = FastAPI(title="PixelForge API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_PATH = os.path.join(BASE_DIR, "vtracer_worker.py")

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

# Load the background-removal model on first use.
_rembg_session = None


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        logger.info("Loading background-removal AI model...")
        _rembg_session = new_session("u2net")
        logger.info("Background-removal model loaded.")
    return _rembg_session


@app.get("/")
def root():
    return {"status": "ok", "service": "pixelforge-api"}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pixelforge-api"}


@app.post("/api/trace")
async def trace_image(
    image: UploadFile = File(...),
    mode: str = Form("color"),
):
    allowed_types = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }

    image_format = allowed_types.get(image.content_type)
    if not image_format:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPG, or WebP image.",
        )

    mode_aliases = {
        "color": "color",
        "binary": "binary",
        "black-white": "binary",
        "black_white": "binary",
        "bw": "binary",
    }

    selected_mode = mode_aliases.get(mode.strip().lower())
    if not selected_mode:
        raise HTTPException(
            status_code=400,
            detail="Choose Color Artwork or Black & White.",
        )

    data = await image.read()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size

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
        "Starting VTracer: mode=%s format=%s dimensions=%sx%s",
        selected_mode, image_format, width, height,
    )

    try:
        result = subprocess.run(
            [sys.executable, WORKER_PATH, image_format, selected_mode],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
            cwd=BASE_DIR,
        )

        if result.returncode != 0:
            logger.error(
                "VTracer failed: code=%s stderr=%r",
                result.returncode, result.stderr[:3000],
            )
            raise HTTPException(
                status_code=500,
                detail="Image conversion failed. Check Render logs for details.",
            )

        svg = result.stdout
        if b"<svg" not in svg[:2000].lower():
            raise HTTPException(
                status_code=500,
                detail="Conversion did not produce a valid SVG file.",
            )

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge.svg"',
                "Cache-Control": "no-store",
            },
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Conversion took too long. Try a smaller image.",
        )
    finally:
        await image.close()


@app.post("/api/remove-background")
async def remove_background(image: UploadFile = File(...)):
    allowed_types = {"image/png", "image/jpeg", "image/webp"}

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPG, or WebP image.",
        )

    try:
        data = await image.read()

        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 5 MB or smaller.",
            )

        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()

            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size

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
                detail="The uploaded file is not a valid image.",
            )

        logger.info(
            "Removing background: dimensions=%sx%s", width, height
        )

        session = get_rembg_session()
        output = remove(data, session=session)

        # Verify that the output is a valid PNG with transparency.
        with Image.open(io.BytesIO(output)) as result:
            result.load()
            rgba = result.convert("RGBA")

            transparent_png = io.BytesIO()
            rgba.save(transparent_png, format="PNG")

        logger.info("Background removed successfully.")

        return Response(
            content=transparent_png.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge-no-background.png"',
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Background removal failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Background removal failed. The server may be out of memory "
                "or still downloading the AI model. Check Render logs."
            ),
        )
    finally:
        await image.close()
