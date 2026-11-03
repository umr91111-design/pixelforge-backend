
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
from lxml import etree

from shape_detector import detect_shapes, refine_svg_with_primitives
from vectorizer_pipeline import run_vectorizer_analysis


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
        "PixelForge image vectorization API. Uses VTracer for vectorization, "
        "OpenCV for primitive recognition, and CPU-based image and SVG "
        "analysis. Topology-aware graph optimization is not yet implemented."
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

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


def validate_svg(svg_text: str) -> bytes:
    """
    Validate SVG as well-formed XML.

    Disables external entity resolution, network access and DTD loading.
    Returns the original SVG encoded as UTF-8.
    """

    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG output is empty.")

    svg_bytes = svg_text.encode("utf-8")

    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError("SVG output exceeds the size limit.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
        remove_comments=False,
    )

    root = etree.fromstring(svg_bytes, parser=parser)

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("The document root is not an SVG element.")

    if root.getroottree().docinfo.doctype:
        raise ValueError("SVG document type declarations are not allowed.")

    return svg_bytes


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
        "version": app.version,
        "features": {
            "vtracer": True,
            "opencv_primitive_detection": True,
            "scikit_image_segmentation": True,
            "shapely_geometry_analysis": True,
            "svg_path_inspection": True,
            "svg_recognition_metadata": True,
            "deep_learning_model": False,
            "gpu_required": False,
            "topology_aware_graph_optimizer": False,
            "shared_boundary_repair": False,
            "bezier_path_refinement": False,
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
                "circle",
                "ellipse",
                "rectangle",
                "square",
                "triangle",
                "star",
                "polygon",
            ],
            "deep_learning": False,
        },
        "image_analysis": {
            "segmentation": "scikit-image Otsu thresholding",
            "contour_geometry": "OpenCV and Shapely",
            "gpu_required": False,
        },
        "svg_analysis": {
            "xml_parser": "lxml",
            "path_inspection": "svgpathtools",
            "changes_svg_geometry": False,
        },
        "geometry_refinement": {
            "status": "recognition metadata and diagnostics",
            "topology_aware_shared_boundary_optimization": False,
            "gap_and_overlap_repair": False,
            "bezier_path_refinement": False,
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

            if (
                width < 1
                or height < 1
                or width * height > MAX_PIXELS
            ):
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Image dimensions are too large. "
                        "Please use a smaller image."
                    ),
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
        await asyncio.wait_for(
            conversion_slots.acquire(),
            timeout=2,
        )

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
            result = await asyncio.to_thread(
                detect_shapes,
                data,
            )

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


def run_pipeline_analysis(data: bytes, svg_text: str):
    """
    Run CPU-based diagnostics after VTracer generates an SVG.

    This function does not rewrite vector paths or repair topology.
    """
    return run_vectorizer_analysis(data, svg_text)


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
            selected_mode,
            image_format,
            width,
            height,
        )

        await acquire_slot()

        started = time.monotonic()
        detected_shapes = []
        analysis_completed = False

        try:
            # STEP 1: Run VTracer.
            try:
                result = await asyncio.to_thread(
                    run_worker,
                    image_format,
                    selected_mode,
                    data,
                )

            except subprocess.TimeoutExpired as exc:
                logger.warning("VTracer worker timed out.")

                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Conversion took too long. "
                        "Please try a smaller image."
                    ),
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
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-2000:],
                )

                raise HTTPException(
                    status_code=500,
                    detail="Image conversion failed.",
                )

            svg_bytes = result.stdout

            if not svg_bytes or len(svg_bytes) > MAX_SVG_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Generated SVG is empty or too large. "
                        "Try a simpler image."
                    ),
                )

            try:
                svg_text = svg_bytes.decode("utf-8")

            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Vectorizer returned invalid SVG encoding.",
                ) from exc

            # STEP 2: Validate the actual SVG XML.
            try:
                validate_svg(svg_text)

            except (ValueError, etree.XMLSyntaxError) as exc:
                logger.exception("VTracer returned invalid SVG XML.")

                raise HTTPException(
                    status_code=500,
                    detail="Conversion did not produce a valid SVG file.",
                ) from exc

            # STEP 3: Detect shapes with OpenCV.
            if recognize_primitives:
                try:
                    detection = await asyncio.to_thread(
                        detect_shapes,
                        data,
                    )

                    detected_shapes = detection.get("shapes", [])

                    logger.info(
                        "OpenCV detected %s shapes.",
                        len(detected_shapes),
                    )

                except Exception:
                    logger.exception(
                        "Primitive recognition failed; "
                        "keeping the original VTracer SVG."
                    )

            # STEP 4: Add shape-recognition metadata.
            if refine_geometry and detected_shapes:
                try:
                    svg_text = await asyncio.to_thread(
                        refine_svg_with_primitives,
                        svg_text,
                        detected_shapes,
                    )

                    # Check the SVG again after metadata processing.
                    validate_svg(svg_text)

                except Exception:
                    logger.exception(
                        "Recognition metadata failed; "
                        "keeping the original VTracer SVG."
                    )

                    # Re-run the original VTracer output through validation.
                    validate_svg(
                        svg_bytes.decode("utf-8")
                    )
                    svg_text = svg_bytes.decode("utf-8")

            # STEP 5: Run integrated CPU analysis.
            if refine_geometry:
                try:
                    analysis = await asyncio.to_thread(
                        run_pipeline_analysis,
                        data,
                        svg_text,
                    )

                    segmentation = analysis.get("segmentation", {})
                    geometry = analysis.get("geometry", {})
                    svg_analysis = analysis.get("svg", {})

                    logger.info(
                        (
                            "Pipeline analysis completed: "
                            "segmentation=%s geometry=%s svg=%s"
                        ),
                        segmentation,
                        geometry,
                        svg_analysis,
                    )

                    analysis_completed = True

                except Exception:
                    logger.exception(
                        "Pipeline analysis failed; "
                        "retaining the generated SVG."
                    )

            # STEP 6: Final output validation.
            final_svg = validate_svg(svg_text)

            elapsed = round(
                time.monotonic() - started,
                3,
            )

            logger.info(
                (
                    "Vectorization completed: "
                    "output=%s bytes shapes=%s analysis=%s elapsed=%ss"
                ),
                len(final_svg),
                len(detected_shapes),
                analysis_completed,
                elapsed,
            )

            if not refine_geometry:
                analysis_status = "disabled"
            elif analysis_completed:
                analysis_status = "completed"
            else:
                analysis_status = "failed"

            return Response(
                content=final_svg,
                media_type="image/svg+xml",
                headers={
                    "Content-Disposition": (
                        'attachment; filename="pixelforge.svg"'
                    ),
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "X-PixelForge-Detected-Shapes": str(
                        len(detected_shapes)
                    ),
                    "X-PixelForge-Deep-Learning": "not-configured",
                    "X-PixelForge-CPU-Analysis": analysis_status,
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
