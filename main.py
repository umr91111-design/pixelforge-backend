
import vtracer
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

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
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


@app.get("/")
def home():
    return {
        "name": "PixelForge AI API",
        "status": "running",
        "message": "Backend is connected and ready for development.",
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

        image_format = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }[file.content_type]

        svg = vtracer.convert_raw_image_to_svg(
            data,
            img_format=image_format,
        )

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": 'attachment; filename="pixelforge.svg"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"VTracer conversion error: {e!r}")
        raise HTTPException(
            status_code=422,
            detail=f"Conversion failed: {e!s}",
        )
    finally:
        await file.close()
