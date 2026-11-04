"""Editable SVG layer preparation for PixelForge AI."""

from lxml import etree


MAX_SVG_BYTES = 20 * 1024 * 1024
MAX_ELEMENTS = 10000

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"


def _parse_svg(svg_text):
    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError("SVG input is empty.")

    if len(svg_text.encode("utf-8")) > MAX_SVG_BYTES:
        raise ValueError("SVG exceeds the size limit.")

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )

    try:
        root = etree.fromstring(svg_text.encode("utf-8"), parser)
    except (etree.XMLSyntaxError, ValueError, UnicodeError) as exc:
        raise ValueError("Invalid SVG XML.") from exc

    if etree.QName(root).localname.lower() != "svg":
        raise ValueError("Document root is not SVG.")

    if root.getroottree().docinfo.doctype:
        raise ValueError("SVG document type declarations are not allowed.")

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue

        if etree.QName(element).localname.lower() in {
            "script",
            "foreignobject",
        }:
            raise ValueError("SVG contains prohibited active content.")

        for attribute in element.attrib:
            name = etree.QName(attribute).localname.lower()
            if name.startswith("on") and len(name) > 2:
                raise ValueError("SVG contains an event handler.")

    return root


def _get_fill(path):
    fill = path.get("fill")
    style = path.get("style", "")

    for declaration in style.split(";"):
        if ":" not in declaration:
            continue

        name, value = declaration.split(":", 1)

        if name.strip().lower() == "fill":
            fill = value.strip()
            break

    return (fill or "black").strip().lower()


def _collect_existing_ids(root):
    """Collect existing IDs so generated layer IDs cannot collide."""
    return {
        element.get("id")
        for element in root.iter()
        if isinstance(element.tag, str) and element.get("id")
    }


def _next_layer_id(existing_ids, layer_number):
    """Generate a unique, deterministic layer ID."""
    base = f"pixelforge-layer-{layer_number:04d}"
    candidate = base
    suffix = 1

    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1

    existing_ids.add(candidate)
    return candidate


def _make_layer(root, paths, fill, layer_number, existing_ids):
    namespace = etree.QName(root).namespace

    group_tag = f"{{{namespace}}}g" if namespace else "g"
    group = etree.Element(group_tag)

    group.set("id", _next_layer_id(existing_ids, layer_number))
    group.set(f"{{{INKSCAPE_NS}}}groupmode", "layer")
    group.set(
        f"{{{INKSCAPE_NS}}}label",
        f"Layer {layer_number:04d} — {fill}",
    )

    for path in paths:
        group.append(path)

    return group


def create_editable_svg_layers(svg_text):
    """Group consecutive paths with matching fills into editable layers."""

    root = _parse_svg(svg_text)
    children = list(root)

    if len(children) > MAX_ELEMENTS:
        raise ValueError("SVG contains too many elements.")

    existing_ids = _collect_existing_ids(root)
    output_children = []

    pending_paths = []
    pending_fill = None

    layer_count = 0
    path_count = 0

    def flush_layer():
        nonlocal layer_count, pending_paths, pending_fill

        if not pending_paths:
            return

        layer_count += 1

        output_children.append(
            _make_layer(
                root,
                pending_paths,
                pending_fill,
                layer_count,
                existing_ids,
            )
        )

        pending_paths = []
        pending_fill = None

    for child in children:
        is_path = (
            isinstance(child.tag, str)
            and etree.QName(child).localname.lower() == "path"
        )

        if not is_path:
            flush_layer()
            output_children.append(child)
            continue

        path_count += 1
        fill = _get_fill(child)

        if pending_paths and fill != pending_fill:
            flush_layer()

        if not pending_paths:
            pending_fill = fill

        pending_paths.append(child)

    flush_layer()

    for child in list(root):
        root.remove(child)

    for child in output_children:
        root.append(child)

    layered_svg = etree.tostring(
        root,
        encoding="unicode",
        xml_declaration=False,
    )

    return layered_svg, {
        "layer_engine": "pixelforge_editable_svg_layers",
        "layers_created": layer_count,
        "paths_processed": path_count,
        "geometry_modified": False,
        "drawing_order_preserved": True,
        "layer_format": "inkscape",
        }
