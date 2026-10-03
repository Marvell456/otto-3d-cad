# AI-assisted 3D dynamic geometry engine.
import json
import os
import re
import sys
from dotenv import load_dotenv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from openai import OpenAI
from shape_catalog import create_preset

load_dotenv()
GEMINI_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_API_KEYS", "").split(",")
    if key.strip()
]
for key_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
    key = os.getenv(key_name, "").strip()
    if key and key not in GEMINI_KEYS:
        GEMINI_KEYS.append(key)
if not GEMINI_KEYS:
    GEMINI_KEYS = ["missing-gemini-key"]
_gemini_key_index = 0

def _make_gemini_client(api_key):
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

client = _make_gemini_client(GEMINI_KEYS[0])

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
SPHERE_SEGMENTS = 24
CYLINDER_SEGMENTS = 24
DEBUG = True  # Prints raw JSON and resolved offsets for layout diagnosis.

SYSTEM_PROMPT = """
You are a 3D Geometry CAD Compiler. Deconstruct the user's requested object into dynamic primitive shapes connected through relative spatial coordinates.

AVAILABLE PRIMITIVE TYPES:
- "cube": dimensions require {"dx": float, "dy": float, "dz": float}  (a box; dx=width along X, dy=depth along Y, dz=height along Z)
- "sphere": dimensions require {"radius": float}
- "cylinder": dimensions require {"radius": float, "height": float}  (axis points along local Z by default)
- "cone": dimensions require {"radius": float, "height": float}  (axis points along local Z by default, apex at +Z)
- "pyramid": dimensions require {"base": float, "height": float}  (apex points along local Z by default)

AXIS CONVENTION:
- X = left/right (width), Y = front/back (depth), Z = up/down (height).
- Every primitive is authored in its OWN local frame with its "main axis" (height/length) along local Z, THEN optionally rotated, THEN placed in the world.

ROTATION (CRITICAL FOR REALISM):
Many real objects are NOT oriented straight up. A plane's fuselage lies on its side, a table leg stands up, a wing is a flat horizontal slab, a pencil lies flat, a wheel's flat face points sideways. To reorient any primitive, add an optional "rotation" object:
  "rotation": {"rx": degrees_around_X, "ry": degrees_around_Y, "rz": degrees_around_Z}
Examples:
  - A cylinder is normally vertical (axis along Z). Rotate "ry": 90 to make it point along X (horizontal, like a fuselage or a lying-down pencil body, or a wheel whose flat circular face should point sideways).
  - Rotate "rz": 90 to spin a box around so its long side faces a different direction.
  - Leave "rotation" out (or all zeros) for parts that should stay upright, like table legs, tree trunks, bottle bodies, vertical tail fins.
Only rotate a part when its natural resting orientation truly is NOT axis-aligned (e.g. a horizontal fuselage, a sideways-facing wheel, a sloped roof beam). The flat faces of an axis-aligned object - the walls of a crate, a tabletop, the sides of a house - do NOT need rotation; build them directly with the correct dx/dy/dz so they already face the right way. If you are not sure a part needs rotation, leave "rotation" out entirely.

VALID MATPLOTLIB COLORS:
"red", "blue", "green", "yellow", "black", "white", "gray", "silver", "tan", "brown", "pink", "orange", "cyan", "purple", "gold", "darkgray", "lightblue".

STRUCTURAL DECONSTRUCTION RULES:
1. MATCH COMPLEXITY TO THE REAL OBJECT: If the requested object truly is just one primitive shape in real life (e.g. "a box", "a cube", "a ball", "a block", "a cylinder", "a can"), output exactly ONE primitive that matches it. Do NOT invent panels, walls, or extra sub-parts just to hit a minimum part count - a plain box is one cube, nothing more. Only decompose an object into multiple parts (typically 3 to 8) when it has genuinely distinct real-world components (e.g. a chair has a seat, backrest, and legs; a car has a body and wheels; a plane has a fuselage, wings, and a tail). Think about the object's TRUE real-world structure, not "how many shapes can I add".
2. PROPORTION MATTERS: Give each part realistic relative size. A wing should be wide and flat (large dx or dy, small dz), not a thick cylinder. A fuselage should be much longer than it is wide. A tabletop should be thin and wide; legs should be thin and tall. Vary dimensions across parts to match real-world proportions - never make every part roughly the same size. A small functional detail (a pencil's graphite tip, a knife's point, a door handle) should still be a CLEARLY VISIBLE fraction of the object - roughly 3-10% of its overall length or size is a reasonable floor. Do not make such a part so tiny it effectively disappears.
3. ORIENT PARTS CORRECTLY: Decide the object's natural resting orientation first (e.g. a plane's fuselage is horizontal, a bottle is vertical, a table is low and wide) and use "rotation" so each part's main axis matches how it sits in the real world (see ROTATION section above).
4. ATTACHMENT SYSTEM:
   - Unique "id" string for every primitive.
   - Core base primitive must have "attach_to": null and "position": "center".
   - Secondary primitives MUST attach to a parent id via "attach_to": "<parent_id>".
   - "position": Choose from ["top", "bottom", "front", "back", "left", "right", "center"] - pick whichever makes structural sense in the real object, even for parts you then rotate (e.g. wings still attach "left"/"right" of the fuselage even though the fuselage cylinder itself is rotated to lie horizontally). Placement automatically accounts for each part's rotation, so rotated parts will sit flush against their parent.
5. RELATIVE SPATIAL OFFSETS:
   The chosen "position" (top/bottom/left/right/front/back) fully determines placement on that axis and always produces flush, zero-gap contact - the renderer ignores any offset you put on that SAME axis, so don't set it (e.g. offset_z is ignored on a "top"/"bottom" attachment; offset_x is ignored on "left"/"right"; offset_y is ignored on "front"/"back"). It cannot be used to add a gap or an overlap - do not try.
   Use offset_x/offset_y/offset_z only on the OTHER two axes not named by "position":
   - Spreading several children symmetrically (e.g. 4 table legs all use position: "bottom", spread out using offset_x and offset_y).
   - Sliding a part along its parent while still attached (e.g. a wing attached "left"/"right" of a fuselage can be slid toward the nose or tail with offset_y, since Y is not controlled by "left"/"right").
   - "offset_x": float (shift along X - only meaningful when position is "top", "bottom", "front", "back", or "center")
   - "offset_y": float (shift along Y - only meaningful when position is "top", "bottom", "left", "right", or "center")
   - "offset_z": float (shift along Z - only meaningful when position is "left", "right", "front", "back", or "center")

QUALITY GATE:
Before returning JSON, mentally inspect the finished object from the side and
top. It must be recognizable from its silhouette, not merely a valid pile of
primitives. Keep all parts connected to one coherent root. Do not use "top" to
stack parts vertically unless that is how the real object is assembled. For a
part attached to a rotated parent, remember that position is based on the
parent's WORLD bounding box, not its local axis. Use "center" plus offsets when
that is clearer. First choose the object's natural resting orientation, then
make the largest silhouette-defining parts, then add only the details needed
for recognition. Never satisfy the schema with a vertical stack of unrelated
boxes.

EXAMPLE DECOMPOSITION PATTERNS (use these as reasoning templates, not fixed answers - adapt proportions and rotations to whatever the user actually asks for):
- Pencil (lying flat): long cylindrical body rotated ry:90 so it's horizontal -> wood cone tip attached "right", also rotated ry:90 so its point faces outward -> tiny graphite cone at the very point -> eraser cylinder rotated ry:90 attached "left" -> thin metal ferrule cylinder (rotated ry:90) between eraser and body.
- Table: flat wide thin tabletop box (large dx/dy, small dz) -> 4 thin vertical leg cylinders (no rotation needed) placed at the 4 corners using +/- offset_x and offset_y combinations, position "bottom".
- Car: low wide rectangular lower body box -> smaller cabin box on "top", set back or centered -> 4 wheel cylinders rotated ry:90 (so the flat circular face points sideways like a real wheel) placed at the 4 corners via offset_x/offset_y.
- Airplane: elongated fuselage cylinder rotated ry:90 (horizontal, nose-to-tail along X) -> 2 flat wide wing boxes (wide dx for span, thin dz, modest dy for chord) attached "left" and "right" of the fuselage, offset to sit at the fuselage's mid-length -> a smaller flat horizontal tail wing box near the rear -> a vertical tail fin box (upright, no rotation) at the very back top.
- House: main rectangular body box -> pyramid roof on "top" -> small door box on "front" -> small flat window boxes on "front".

OUTPUT REQUIREMENT: Return ONLY valid JSON with top-level key "primitives". No code fences, no markdown formatting, no explanations.
"""

FAST_SYSTEM_PROMPT = """
You generate simple recognizable 3D diagrams from primitive shapes.
Return ONLY JSON: {"primitives":[...]}.
Each part needs id, type, dimensions, attach_to, position, and optional
rotation, color, and offsets. Types are cube, sphere, cylinder, cone, pyramid.
For cubes, dimensions MUST be an object with dx, dy, dz. Position MUST be one
of top, bottom, front, back, left, right, center; use offset_x, offset_y, and
offset_z for numeric placement.
Use one root with attach_to null and attach every other part to an earlier
part. Use 2-8 parts, realistic proportions, one coherent silhouette, and
center plus offsets when that is clearer than directional attachment.
Axes: X width/length, Y depth, Z up. Primitive height is local Z. Rotate
cylinders with ry:90 for horizontal length. Make the largest recognizable
parts first; never return a random vertical stack.
"""

def _gemini_completion(**kwargs):
    # Call Gemini and turn quota failures into an actionable local error.
    global client, _gemini_key_index
    first_index = _gemini_key_index
    last_error = None
    for offset in range(len(GEMINI_KEYS)):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            last_error = error
            status_code = getattr(error, "status_code", None)
            is_quota_error = status_code == 429 or "quota" in str(error).lower()
            if not is_quota_error or offset == len(GEMINI_KEYS) - 1:
                break
            _gemini_key_index = (first_index + offset + 1) % len(GEMINI_KEYS)
            client = _make_gemini_client(GEMINI_KEYS[_gemini_key_index])
    retry_match = re.search(
        r"retry(?: again)? in\s+([0-9]+(?:\.[0-9]+)?)s",
        str(last_error),
        flags=re.IGNORECASE,
    )
    wait_message = (
        f" Try again in about {max(1, round(float(retry_match.group(1))))} seconds."
        if retry_match
        else " Try again after the quota resets."
    )
    raise RuntimeError(
        "All configured Gemini API keys have reached their quota."
        + wait_message
    ) from last_error

def clean_input_text(text):
    patterns = [
        r"^(can you\s+)?(please\s+)?(create|make|generate|build|draw|show|give)\s+(me\s+)?(a\s+|an\s+|the\s+)?",
        r"^(how about generating me a 3d model of a\s+|how about generating me a 3d model of\s+)",
        r"^(3d\s+model\s+of\s+|3d\s+shape\s+of\s+|3d\s+)",
        r"^(a\s+|an\s+)",
    ]
    cleaned = text.lower().strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(
        r"\s+(3d\s+model|model|shape|diagram)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned if cleaned else text

def sanitize_color(raw_color):
    if not isinstance(raw_color, str):
        return "gray"
    color_clean = raw_color.lower().strip()
    custom_map = {
        "skin": "tan",
        "flesh": "peachpuff",
        "silver": "lightgray",
        "clear": "cyan",
        "transparent": "lightblue",
    }
    if color_clean in custom_map:
        return custom_map[color_clean]
    if mcolors.is_color_like(color_clean):
        return color_clean
    return "gray"

# Absolute layout engine (supports multi-child corner & side offsets)
def resolve_primitive_positions(primitives):
    computed = {}

    pending = list(primitives)
    ordered = []
    known_ids = set()
    while pending:
        progressed = False
        for item in pending[:]:
            item_id = str(item.get("id", f"item_{id(item)}"))
            parent_id = item.get("attach_to")
            if parent_id is None or str(parent_id) in known_ids:
                ordered.append(item)
                known_ids.add(item_id)
                pending.remove(item)
                progressed = True
        if not progressed:
            raise ValueError("Primitive layout contains an unresolved attachment cycle")

    for item in ordered:
        item_id = str(item.get("id", f"item_{id(item)}"))
        ptype = item.get("type")
        dims = item.get("dimensions", {})
        parent_id = item.get("attach_to")
        pos = item.get("position", "top")
        rot = item.get("rotation") or {}
        rotation = (
            float(rot.get("rx", 0.0)),
            float(rot.get("ry", 0.0)),
            float(rot.get("rz", 0.0)),
        )

        rel_ox = float(item.get("offset_x", 0.0))
        rel_oy = float(item.get("offset_y", 0.0))
        rel_oz = float(item.get("offset_z", 0.0))

        if ptype in ("cube", "box"):
            dx, dy, dz = (
                float(dims.get("dx", 1)),
                float(dims.get("dy", 1)),
                float(dims.get("dz", 1)),
            )
        elif ptype in ("cylinder", "cone"):
            r = float(dims.get("radius", 1))
            dx, dy, dz = r * 2, r * 2, float(dims.get("height", 2))
        elif ptype == "pyramid":
            b = float(dims.get("base", 1))
            dx, dy, dz = b, b, float(dims.get("height", 1))
        elif ptype in ("sphere", "ball"):
            r = float(dims.get("radius", 1))
            dx, dy, dz = r * 2, r * 2, r * 2
        else:
            dx, dy, dz = 1.0, 1.0, 1.0

        # Use world-space size after rotation for all distance calculations.
        dx, dy, dz = _rotated_extents(dx, dy, dz, rotation)

        if not parent_id or str(parent_id) not in computed:
            computed[item_id] = {
                "offset": [rel_ox, rel_oy, rel_oz],
                "bounds": [dx, dy, dz],
            }
        else:
            parent = computed[str(parent_id)]
            px, py, pz = parent["offset"]
            pdx, pdy, pdz = parent["bounds"]

            off_x, off_y, off_z = px + rel_ox, py + rel_oy, pz + rel_oz

            # "position" fully determines the axis it names; same-axis offset is ignored.
            if pos == "top":
                off_z = pz + (pdz / 2) + (dz / 2)
            elif pos == "bottom":
                off_z = pz - (pdz / 2) - (dz / 2)
            elif pos == "front":
                off_y = py + (pdy / 2) + (dy / 2)
            elif pos == "back":
                off_y = py - (pdy / 2) - (dy / 2)
            elif pos == "right":
                off_x = px + (pdx / 2) + (dx / 2)
            elif pos == "left":
                off_x = px - (pdx / 2) - (dx / 2)

            computed[item_id] = {
                "offset": [off_x, off_y, off_z],
                "bounds": [dx, dy, dz],
            }

        item["offset"] = computed[item_id]["offset"]

        if DEBUG:
            print(
                f"  [{item_id}] type={ptype} attach_to={parent_id} position={pos} "
                f"rotation={rotation} raw_offset_xyz=({rel_ox}, {rel_oy}, {rel_oz}) "
                f"-> world_offset={[round(v, 3) for v in item['offset']]} "
                f"size=({dx:.2f}, {dy:.2f}, {dz:.2f})"
            )

    return ordered

# Geometry mesh calculations
def rotation_matrix(rx=0.0, ry=0.0, rz=0.0):
    # Combined rotation matrix (Rz * Ry * Rx) built from degrees.
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def transform_points(points, offset, rotation=None):
    pts = np.asarray(points, dtype=float)
    if rotation is not None and any(rotation):
        R = rotation_matrix(*rotation)
        pts = pts @ R.T
    return pts + np.array(offset)

def _rotated_extents(dx, dy, dz, rotation):
    # World-space AABB size of a rotated dx*dy*dz box centered on its origin.
    if rotation is None or not any(rotation):
        return dx, dy, dz
    corners = np.array(
        [
            [sx * dx / 2, sy * dy / 2, sz * dz / 2]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    R = rotation_matrix(*rotation)
    world = corners @ R.T
    size = world.max(axis=0) - world.min(axis=0)
    return float(size[0]), float(size[1]), float(size[2])

def cube_faces(dx, dy, dz, offset=(0, 0, 0), rotation=None):
    x, y, z = [-dx / 2, dx / 2], [-dy / 2, dy / 2], [-dz / 2, dz / 2]
    local_corners = np.array([[i, j, k] for i in x for j in y for k in z])
    world_corners = transform_points(local_corners, offset, rotation)

    faces_idx = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    faces = [[world_corners[i] for i in face] for face in faces_idx]
    return faces, get_bbox(world_corners)

def pyramid_faces(base, height, offset=(0, 0, 0), rotation=None):
    half = base / 2
    local_pts = [
        [-half, -half, -height / 2],
        [half, -half, -height / 2],
        [half, half, -height / 2],
        [-half, half, -height / 2],
        [0, 0, height / 2],
    ]
    world_pts = transform_points(local_pts, offset, rotation)

    faces = [
        [world_pts[0], world_pts[1], world_pts[2], world_pts[3]],
        [world_pts[0], world_pts[1], world_pts[4]],
        [world_pts[1], world_pts[2], world_pts[4]],
        [world_pts[2], world_pts[3], world_pts[4]],
        [world_pts[3], world_pts[0], world_pts[4]],
    ]
    return faces, get_bbox(world_pts)

def cone_faces(radius, height, offset=(0, 0, 0), rotation=None):
    theta = np.linspace(0, 2 * np.pi, CYLINDER_SEGMENTS)
    base_pts = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, -height / 2)]
    )
    apex_pt = np.array([[0, 0, height / 2]])

    world_base = transform_points(base_pts, offset, rotation)
    world_apex = transform_points(apex_pt, offset, rotation)[0]

    faces = [
        [world_base[i], world_base[i + 1], world_apex]
        for i in range(len(theta) - 1)
    ]
    faces.append(list(world_base))
    return faces, get_bbox(np.vstack([world_base, world_apex]))

def cylinder_surface(radius, height, offset=(0, 0, 0), rotation=None):
    theta = np.linspace(0, 2 * np.pi, CYLINDER_SEGMENTS)
    theta_grid, z_grid = np.meshgrid(theta, np.array([-height / 2, height / 2]))

    pts = np.column_stack(
        [
            (radius * np.cos(theta_grid)).ravel(),
            (radius * np.sin(theta_grid)).ravel(),
            z_grid.ravel(),
        ]
    )
    world_pts = transform_points(pts, offset, rotation)

    x_world = world_pts[:, 0].reshape(theta_grid.shape)
    y_world = world_pts[:, 1].reshape(theta_grid.shape)
    z_world = world_pts[:, 2].reshape(z_grid.shape)

    caps = [
        list(world_pts[:CYLINDER_SEGMENTS]),
        list(world_pts[CYLINDER_SEGMENTS:]),
    ]
    return (x_world, y_world, z_world), caps, get_bbox(world_pts)

def sphere_surface(radius, offset=(0, 0, 0), rotation=None):
    u = np.linspace(0, 2 * np.pi, SPHERE_SEGMENTS)
    v = np.linspace(0, np.pi, SPHERE_SEGMENTS)

    x_loc = radius * np.outer(np.cos(u), np.sin(v))
    y_loc = radius * np.outer(np.sin(u), np.sin(v))
    z_loc = radius * np.outer(np.ones_like(u), np.cos(v))

    pts = np.column_stack([x_loc.ravel(), y_loc.ravel(), z_loc.ravel()])
    world_pts = transform_points(pts, offset, rotation)

    x_world = world_pts[:, 0].reshape(x_loc.shape)
    y_world = world_pts[:, 1].reshape(y_loc.shape)
    z_world = world_pts[:, 2].reshape(z_loc.shape)

    return (x_world, y_world, z_world), get_bbox(world_pts)

def get_bbox(pts):
    pts = np.asarray(pts)
    return (
        np.min(pts[:, 0]),
        np.max(pts[:, 0]),
        np.min(pts[:, 1]),
        np.max(pts[:, 1]),
        np.min(pts[:, 2]),
        np.max(pts[:, 2]),
    )

# Pipeline execution & rendering
def _validate_layout(layout_data):
    # Validate the model contract before geometry code consumes it.
    if not isinstance(layout_data, dict) or not isinstance(
        layout_data.get("primitives"), list
    ):
        return ["top-level 'primitives' list is missing"]

    primitives = layout_data["primitives"]
    errors = []
    ids = set()
    valid_types = {"cube", "box", "sphere", "ball", "cylinder", "cone", "pyramid"}
    valid_positions = {"top", "bottom", "front", "back", "left", "right", "center"}
    all_ids = {
        str(item.get("id", ""))
        for item in primitives
        if isinstance(item, dict) and item.get("id")
    }

    for index, item in enumerate(primitives):
        if not isinstance(item, dict):
            errors.append(f"part {index} is not an object")
            continue
        item_id = str(item.get("id", ""))
        if not item_id or item_id in ids:
            errors.append(f"part {index} has a missing or duplicate id")
        ids.add(item_id)
        if item.get("type") not in valid_types:
            errors.append(f"{item_id or index} has an invalid primitive type")
        if item.get("position", "top") not in valid_positions:
            errors.append(f"{item_id or index} has an invalid position")
        parent_id = item.get("attach_to")
        if parent_id is not None and str(parent_id) not in all_ids:
            errors.append(f"{item_id or index} references a missing parent")

        dimensions = item.get("dimensions")
        if not isinstance(dimensions, dict):
            errors.append(f"{item_id or index} is missing dimensions")
            continue
        required = {"radius"} if item.get("type") in {"sphere", "ball"} else set()
        if item.get("type") in {"cylinder", "cone"}:
            required = {"radius", "height"}
        elif item.get("type") == "pyramid":
            required = {"base", "height"}
        elif item.get("type") in {"cube", "box"}:
            required = {"dx", "dy", "dz"}
        for key in required:
            try:
                if not np.isfinite(float(dimensions[key])) or float(dimensions[key]) <= 0:
                    errors.append(f"{item_id or index} has invalid {key}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"{item_id or index} is missing valid {key}")

    roots = [item for item in primitives if isinstance(item, dict) and item.get("attach_to") is None]
    if len(roots) != 1:
        errors.append(f"layout must have exactly one root, found {len(roots)}")
    return errors

def _normalize_layout(layout_data):
    # Convert common model shorthand into the renderer's strict schema.
    if not isinstance(layout_data, dict) or not isinstance(
        layout_data.get("primitives"), list
    ):
        return layout_data

    normalized = json.loads(json.dumps(layout_data))
    for index, item in enumerate(normalized["primitives"]):
        if not isinstance(item, dict):
            normalized["primitives"][index] = {"id": f"invalid_{index}"}
            continue
        if "dimensions" not in item or not isinstance(
            item.get("dimensions"), (dict, list)
        ):
            item["dimensions"] = {}
        for key in ("radius", "height", "base"):
            if key in item and key not in item["dimensions"]:
                item["dimensions"][key] = item.pop(key)
        rotation = item.get("rotation")
        if isinstance(rotation, (list, tuple)):
            item["rotation"] = {
                "rx": rotation[0] if len(rotation) > 0 else 0,
                "ry": rotation[1] if len(rotation) > 1 else 0,
                "rz": rotation[2] if len(rotation) > 2 else 0,
            }
        elif isinstance(rotation, dict):
            for old_key, new_key in (("x", "rx"), ("y", "ry"), ("z", "rz")):
                if old_key in rotation and new_key not in rotation:
                    rotation[new_key] = rotation.pop(old_key)
        else:
            item["rotation"] = {"rx": 0, "ry": 0, "rz": 0}
        dimensions = item.get("dimensions")
        ptype = item.get("type")
        if isinstance(dimensions, list):
            if ptype in {"cube", "box"} and len(dimensions) >= 3:
                item["dimensions"] = {
                    "dx": dimensions[0],
                    "dy": dimensions[1],
                    "dz": dimensions[2],
                }
            elif ptype in {"cylinder", "cone"} and len(dimensions) >= 2:
                item["dimensions"] = {
                    "radius": dimensions[0],
                    "height": dimensions[1],
                }
            elif ptype == "pyramid" and len(dimensions) >= 2:
                item["dimensions"] = {
                    "base": dimensions[0],
                    "height": dimensions[1],
                }
            elif ptype in {"sphere", "ball"} and dimensions:
                item["dimensions"] = {"radius": dimensions[0]}
        elif not isinstance(dimensions, dict):
            item["dimensions"] = {}
        elif ptype in {"sphere", "ball"} and isinstance(dimensions, dict):
            if "radius" not in dimensions and all(
                key in dimensions for key in ("dx", "dy", "dz")
            ):
                item["dimensions"] = {
                    "radius": max(
                        float(dimensions["dx"]),
                        float(dimensions["dy"]),
                        float(dimensions["dz"]),
                    )
                    / 2
                }

        position = item.get("position")
        if isinstance(position, list):
            if len(position) >= 3:
                item["offset_x"], item["offset_y"], item["offset_z"] = position[:3]
            item["position"] = "center"

    return normalized

def _decode_layout_response(raw_text):
    # Decode JSON even when the provider adds markdown around the object.
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return _normalize_layout(json.loads(cleaned))
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return _normalize_layout(json.loads(cleaned[start:end + 1]))
        raise

def _repair_layout(clean_text, layout_data, errors):
    # Give the model one focused correction pass for malformed output.
    repair_prompt = f"""
The following draft for '{clean_text}' is not acceptable:
{json.dumps(layout_data, indent=2)}

Problems found: {', '.join(errors)}

Return a corrected JSON object with only a 'primitives' array. Preserve the
object's real-world identity and silhouette. Use one root, attach every other
part to an existing parent, use positive dimensions, and avoid arbitrary
vertical stacking. Check the result from both top and side views before
returning it. Return JSON only.
"""
    response = _gemini_completion(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FAST_SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.0,
        max_tokens=1800,
    )
    return _decode_layout_response(response.choices[0].message.content)

def parse_description_with_gemini(user_text):
    clean_text = clean_input_text(user_text)
    catalog_layout = create_preset(clean_text)
    if catalog_layout is not None:
        if DEBUG:
            print(f"\n--- LOCAL CATALOG MODEL: {clean_text} ---")
        return catalog_layout
    messages = [
        {"role": "system", "content": FAST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Generate 3D attachment primitives for object: '{clean_text}'",
        },
    ]
    response = _gemini_completion(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.1,
        max_tokens=8192,
    )
    raw_json = response.choices[0].message.content or ""
    try:
        layout_data = _decode_layout_response(raw_json)
    except json.JSONDecodeError:
        recovery_messages = [
            {"role": "system", "content": FAST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Return a compact, complete JSON 3D model for '{clean_text}'. "
                    "Use at most 5 primitives. Include every required field, use short ids, "
                    "and return JSON only. Do not explain anything."
                ),
            },
        ]
        last_error = None
        for _attempt in range(2):
            retry_response = _gemini_completion(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=recovery_messages,
                temperature=0.0,
                max_tokens=8192,
            )
            raw_json = retry_response.choices[0].message.content or ""
            try:
                layout_data = _decode_layout_response(raw_json)
                break
            except json.JSONDecodeError as error:
                last_error = error
        else:
            raise ValueError(
                f"Gemini returned incomplete JSON after retries: {last_error}"
            ) from last_error
    if DEBUG:
        print("\n--- RAW MODEL OUTPUT ---")
        print(raw_json)
        print("--- END RAW OUTPUT ---")
    errors = _validate_layout(layout_data)
    if errors:
        layout_data = _repair_layout(clean_text, layout_data, errors)
        repaired_errors = _validate_layout(layout_data)
        if repaired_errors:
            raise ValueError(
                "The 3D model generator returned an invalid layout after repair: "
                + "; ".join(repaired_errors)
            )
    return layout_data

def _get_rotation(item):
    rot = item.get("rotation") or {}
    if isinstance(rot, (list, tuple)):
        rot = list(rot) + [0, 0, 0]
        return tuple(float(value or 0) for value in rot[:3])
    if not isinstance(rot, dict):
        rot = {}
    return (
        _safe_float(rot.get("rx", 0.0)),
        _safe_float(rot.get("ry", 0.0)),
        _safe_float(rot.get("rz", 0.0)),
    )

def _safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default

def _surface_faces(surface):
    # Convert a gridded surface into faces for global depth sorting.
    x_values, y_values, z_values = surface
    rows, columns = x_values.shape
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            faces.append([
                (x_values[row, column], y_values[row, column], z_values[row, column]),
                (x_values[row + 1, column], y_values[row + 1, column], z_values[row + 1, column]),
                (x_values[row + 1, column + 1], y_values[row + 1, column + 1], z_values[row + 1, column + 1]),
                (x_values[row, column + 1], y_values[row, column + 1], z_values[row, column + 1]),
            ])
    return faces

def _solid_collection(faces, colors):
    # Create one opaque collection so every face is sorted globally.
    return Poly3DCollection(
        faces,
        facecolor=colors,
        edgecolor="#20242b",
        linewidths=0.18,
        alpha=1.0,
        zsort="average",
    )

def _triangle_normal(first, second, third):
    normal = np.cross(np.asarray(second) - first, np.asarray(third) - first)
    length = np.linalg.norm(normal)
    return normal / length if length else np.zeros(3)

def _write_obj(path, faces):
    with open(path, "w", encoding="utf-8") as output:
        output.write("# Exported 3D model\n")
        vertex_index = 1
        for face in faces:
            vertices = [np.asarray(point, dtype=float) for point in face]
            for vertex in vertices:
                output.write("v {:.9g} {:.9g} {:.9g}\n".format(*vertex))
            indices = " ".join(
                str(index) for index in range(vertex_index, vertex_index + len(vertices))
            )
            output.write(f"f {indices}\n")
            vertex_index += len(vertices)

def _write_stl(path, faces):
    with open(path, "w", encoding="ascii") as output:
        output.write("solid model\n")
        for face in faces:
            if len(face) < 3:
                continue
            first = np.asarray(face[0], dtype=float)
            for index in range(1, len(face) - 1):
                second = np.asarray(face[index], dtype=float)
                third = np.asarray(face[index + 1], dtype=float)
                normal = _triangle_normal(first, second, third)
                output.write("  facet normal {:.9g} {:.9g} {:.9g}\n".format(*normal))
                output.write("    outer loop\n")
                for vertex in (first, second, third):
                    output.write("      vertex {:.9g} {:.9g} {:.9g}\n".format(*vertex))
                output.write("    endloop\n  endfacet\n")
        output.write("endsolid model\n")

def _export_model(path, file_type, faces):
    if file_type == "OBJ (.obj)":
        _write_obj(path, faces)
    else:
        _write_stl(path, faces)

def render_primitives(ax, primitives, mesh_faces=None):
    bboxes = []
    all_faces = []
    all_colors = []
    for item in primitives:
        ptype = item.get("type")
        color = sanitize_color(item.get("color", "gray"))
        offset = item.get("offset", [0, 0, 0])
        dims = item.get("dimensions", {})
        rotation = _get_rotation(item)

        if ptype in ("cube", "box"):
            faces, bbox = cube_faces(
                dims.get("dx", 1.0),
                dims.get("dy", 1.0),
                dims.get("dz", 1.0),
                offset,
                rotation,
            )
            all_faces.extend(faces)
            all_colors.extend([color] * len(faces))
            bboxes.append(bbox)

        elif ptype in ("sphere", "ball"):
            surface, bbox = sphere_surface(dims.get("radius", 1.0), offset, rotation)
            faces = _surface_faces(surface)
            all_faces.extend(faces)
            all_colors.extend([color] * len(faces))
            bboxes.append(bbox)

        elif ptype == "cylinder":
            surface, caps, bbox = cylinder_surface(
                dims.get("radius", 1.0), dims.get("height", 2.0), offset, rotation
            )
            faces = _surface_faces(surface) + caps
            all_faces.extend(faces)
            all_colors.extend([color] * len(faces))
            bboxes.append(bbox)

        elif ptype == "cone":
            faces, bbox = cone_faces(
                dims.get("radius", 1.0), dims.get("height", 2.0), offset, rotation
            )
            all_faces.extend(faces)
            all_colors.extend([color] * len(faces))
            bboxes.append(bbox)

        elif ptype == "pyramid":
            faces, bbox = pyramid_faces(
                dims.get("base", 1.0), dims.get("height", 2.0), offset, rotation
            )
            all_faces.extend(faces)
            all_colors.extend([color] * len(faces))
            bboxes.append(bbox)

    if all_faces:
        ax.add_collection3d(_solid_collection(all_faces, all_colors))
        if mesh_faces is not None:
            mesh_faces.extend(all_faces)
    return bboxes

def _increase_rotation_sensitivity(ax, factor=3.0):
    # Scale only 3D rotate drag deltas, leaving pan and zoom behavior intact.
    callbacks = ax.figure.canvas.callbacks.callbacks.get("motion_notify_event", {})
    original_callback_id = None
    for callback_id, callback in list(callbacks.items()):
        callback_function = callback() if callable(callback) else None
        if callback_function == ax._on_move:
            original_callback_id = callback_id
            break
    if original_callback_id is None:
        return

    ax.figure.canvas.mpl_disconnect(original_callback_id)
    original_on_move = ax._on_move

    def faster_on_move(event):
        if (
            ax.button_pressed in ax._rotate_btn
            and event.inaxes is ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            physical_x, physical_y = event.xdata, event.ydata
            event.xdata = ax._sx + (physical_x - ax._sx) * factor
            event.ydata = ax._sy + (physical_y - ax._sy) * factor
            original_on_move(event)
            ax._sx, ax._sy = physical_x, physical_y
            return
        original_on_move(event)

    ax.figure.canvas.mpl_connect("motion_notify_event", faster_on_move)

def _remove_toolbar_buttons(toolbar, names):
    for name in names:
        button = getattr(toolbar, "_buttons", {}).get(name)
        if button is not None:
            button.destroy()

def _add_export_bar(manager, faces):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    export_bar = tk.Frame(manager.window, bg="#0B0E14")
    export_bar.pack(side="bottom", fill="x", padx=8, pady=6)
    ttk.Label(export_bar, text="3D model format:").pack(side="left", padx=(0, 6))
    file_type = tk.StringVar(value="STL (.stl)")
    type_menu = ttk.Combobox(
        export_bar,
        textvariable=file_type,
        values=("STL (.stl)", "OBJ (.obj)"),
        state="readonly",
        width=12,
    )
    type_menu.pack(side="left", padx=(0, 6))

    def save_model():
        selected_type = file_type.get()
        extension = ".obj" if selected_type == "OBJ (.obj)" else ".stl"
        path = filedialog.asksaveasfilename(
            title="Export 3D model",
            defaultextension=extension,
            filetypes=[
                ("Wavefront OBJ", "*.obj") if extension == ".obj" else ("STL model", "*.stl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            _export_model(path, selected_type, faces)
            messagebox.showinfo("Export complete", f"Saved 3D model to:\n{path}")
        except OSError as error:
            messagebox.showerror("Export failed", str(error))

    ttk.Button(export_bar, text="Save 3D model", command=save_model).pack(side="left")

def display_diagram(user_prompt, layout_data):
    from skeleton import HandSkeletonController

    fig = plt.figure(figsize=(8, 7), facecolor="#0B0E14")
    ax = fig.add_axes([0.035, 0.035, 0.93, 0.91], projection="3d")
    ax.set_facecolor("#111827")

    raw_primitives = layout_data.get("primitives", [])
    resolved_primitives = resolve_primitive_positions(raw_primitives)
    mesh_faces = []
    bboxes = render_primitives(ax, resolved_primitives, mesh_faces)

    if not bboxes:
        print(" ! No valid primitives rendered.")
        plt.close(fig)
        return

    xmin = min(b[0] for b in bboxes)
    xmax = max(b[1] for b in bboxes)
    ymin = min(b[2] for b in bboxes)
    ymax = max(b[3] for b in bboxes)
    zmin = min(b[4] for b in bboxes)
    zmax = max(b[5] for b in bboxes)

    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-6)
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    half = span / 2 * 1.3

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=24, azim=-58)

    ax.set_xlabel("X", color="#CBD5E1", labelpad=8)
    ax.set_ylabel("Y", color="#CBD5E1", labelpad=8)
    ax.set_zlabel("Z", color="#CBD5E1", labelpad=8)
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(True, color="#334155", linewidth=0.6, alpha=0.65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#111827")
        axis.pane.set_edgecolor("#334155")
    ax.set_title(
        f"3D MODEL  •  {user_prompt.upper()}\n"
        "(Open: zoom in  •  Grip: zoom out  •  Point: swipe to rotate)",
        color="#E2E8F0",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    def on_scroll(event):
        base_scale = 1.15 ** 2
        cur_xlim, cur_ylim, cur_zlim = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim(),
        )
        scale_factor = (
            (1 / base_scale)
            if event.button == "up"
            else (base_scale if event.button == "down" else 1.0)
        )

        xc, yc, zc = (
            (cur_xlim[0] + cur_xlim[1]) / 2,
            (cur_ylim[0] + cur_ylim[1]) / 2,
            (cur_zlim[0] + cur_zlim[1]) / 2,
        )
        xs, ys, zs = (
            (cur_xlim[1] - cur_xlim[0]) * scale_factor,
            (cur_ylim[1] - cur_ylim[0]) * scale_factor,
            (cur_zlim[1] - cur_zlim[0]) * scale_factor,
        )

        ax.set_xlim([xc - xs / 2, xc + xs / 2])
        ax.set_ylim([yc - ys / 2, yc + ys / 2])
        ax.set_zlim([zc - zs / 2, zc + zs / 2])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    _increase_rotation_sensitivity(ax)

    def keep_axes_fitted(_event=None):
        # Reassert the normalized canvas position after fullscreen resize events.
        ax.set_position([0.035, 0.035, 0.93, 0.91])

    fig.canvas.mpl_connect("resize_event", keep_axes_fitted)
    hand_controller = HandSkeletonController(fig, ax)
    hand_controller.start()
    manager = plt.get_current_fig_manager()
    _remove_toolbar_buttons(getattr(manager, "toolbar", None), {"Subplots", "Save"})
    _add_export_bar(manager, mesh_faces)

    def enter_fullscreen():
        try:
            manager.window.attributes("-fullscreen", True)
        except (AttributeError, RuntimeError, OSError):
            try:
                manager.full_screen_toggle()
            except (AttributeError, RuntimeError):
                try:
                    manager.window.state("zoomed")
                except (AttributeError, RuntimeError):
                    return
        try:
            manager.window.update_idletasks()
        except AttributeError:
            pass
        keep_axes_fitted()
        fig.canvas.draw_idle()

    try:
        manager.window.after(100, enter_fullscreen)
    except (AttributeError, RuntimeError):
        enter_fullscreen()
    try:
        plt.show()
    finally:
        hand_controller.stop()
        plt.close(fig)

def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
        print(f"\n[shapes.py] Live Spatial Construction: '{text}'")
        try:
            layout_data = parse_description_with_gemini(text)
            display_diagram(text, layout_data)
        except Exception as e:
            print(f"Error rendering shape: {e}")
        return

    print(__doc__)
    while True:
        try:
            text = input(
                "\nDescribe ANY object to generate a 3D diagram (or 'quit'): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            layout_data = parse_description_with_gemini(text)
            display_diagram(text, layout_data)
        except Exception as e:
            print(f"Error rendering shape: {e}")

if __name__ == "__main__":
    main()