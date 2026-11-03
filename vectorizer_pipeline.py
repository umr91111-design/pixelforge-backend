def _is_valid_closed_line_path(path: Path) -> bool:
    """
    Check whether a closed, line-only path forms valid geometry.

    Reject:
    - Open paths
    - Curved paths
    - Duplicate consecutive points
    - Zero-length edges
    - Non-finite coordinates
    - Self-intersecting or otherwise invalid polygons
    """
    if not path or not path.isclosed():
        return False

    if not all(
        isinstance(segment, Line)
        for segment in path
    ):
        return False

    points = [
        (segment.start.real, segment.start.imag)
        for segment in path
    ]

    if len(points) < 3:
        return False

    # Reject NaN and infinite coordinates.
    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            return False

    # Reject duplicate consecutive points,
    # including between the final and first vertices.
    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]

        if current == following:
            return False

    # Reject zero-length or extremely short line segments.
    for segment in path:
        if abs(segment.end - segment.start) <= 1e-9:
            return False

    try:
        polygon = Polygon(points)

        return (
            not polygon.is_empty
            and polygon.area > 0
            and polygon.is_valid
            and polygon.exterior.is_simple
        )

    except Exception:
        logger.exception("Polygon validity check failed.")
        return False
