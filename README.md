# PixelForge backend update (v1.1.0)

## Included
- `main.py`: preserves `/`, `/api/health`, `/api/detect-shapes`, and `/api/trace`.
- `shape_detector.py`: OpenCV contour detection for circles, ellipses, squares, rectangles, triangles, stars and polygons.
- `vtracer_worker.py`: VTracer color/binary conversion worker.
- `requirements.txt`: backend dependencies.

## What's implemented
- VTracer raster-to-SVG in color and binary modes.
- OpenCV-based primitive detection, now also called during `/api/trace`.
- Bounding boxes and areas mapped back to original image coordinates.
- SVG recognition metadata added without altering VTracer paths.
- OpenAPI schema at `/openapi.json`; Swagger UI at `/docs`.

## What's not implemented yet
- No trained deep-learning model or model weights are included.
- No topology-aware vector graph engine, shared-boundary snapping, or curve reconstruction is included. The current SVG enrichment deliberately preserves VTracer geometry to avoid introducing guessed paths, gaps or overlaps.
- No language-specific SDK packages are generated yet. The OpenAPI schema is the starting point for SDK generation.

## Existing frontend compatibility
Existing multipart requests containing `image` and `mode` still work.
Optional `/api/trace` form fields:
- `recognize_primitives`: `true` or `false` (default `true`)
- `refine_geometry`: `true` or `false` (default `true`)

## Limits
- Upload size: 5 MB.
- Backend image size: 4 million pixels.
- Worker timeout: 60 seconds.
- Concurrent jobs: 2.

## Render start command
`uvicorn main:app --host 0.0.0.0 --port $PORT`

After deployment, test:
1. `/api/health`
2. `/api/features`
3. `/openapi.json`
4. `/api/trace` with PNG/JPEG/WebP in color and binary modes.
5. `/api/detect-shapes` with simple shapes.
