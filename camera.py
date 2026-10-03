# Camera capture tool: turns a webcam frame into a 3D primitive model via Gemini.
import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageTk

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

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Optional stronger model for builds; defaults to GEMINI_MODEL.
BUILD_MODEL_NAME = os.getenv("GEMINI_BUILD_MODEL", MODEL_NAME)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Hard limits keep the viewer smooth and the build predictable.
MAX_PARTS = 48
MIN_DETAIL_PARTS = 5
MAX_REFINE_ROUNDS = 2
IMAGE_MAX_SIDE = 1280
BUILD_TIMEOUT_SECONDS = 240
ENABLE_SHADING = True

# Prompts
BUILD_SYSTEM_PROMPT = """
You are an expert 3D modeler. You look at a webcam photo and rebuild the main
object in it out of geometric primitives so that the result is instantly
recognisable as that specific object. The viewer draws each primitive as a
solid colour with thin outlines and simple directional shading, so silhouette,
proportions, colours and visible surface details (screens, buttons, handles,
windows, labels, legs) are what make the model look good.

WHAT TO MODEL
- The main object: the closest, most central, largest in-focus thing. Ignore the
  background, table, walls, shadows, reflections, cables, and any hand, fingers
  or arm holding the object.
- If two or three separate objects are clearly the subject, model each one,
  the main one in the most detail, placed where they touch each other.
- Copy what you actually see: real proportions, real counts (if a calculator
  has 4 columns and 5 rows of keys, build exactly that), real colours.
- Never fall back to a generic box or cylinder. A single plain shape is only
  correct when the object really is one plain shape (a ball, a wooden block).

COORDINATES
- One consistent unit. Scale the object so its longest side is about 10 units
  and put its middle near (0, 0, 0).
- Keep the pose from the photo: photo-up is +Z, photo-right is +X, and the side
  facing the camera is the FRONT, which faces -Y. An object held up to the
  camera (phone, calculator, book, poster) therefore stands upright with its
  face toward -Y. An object on a table rests on its base with its top toward +Z.

PRIMITIVES (each part is centred on its own "center": [x, y, z])
- "box":      "size" {"x", "y", "z"}  full lengths along X, Y, Z.
- "cylinder": "size" {"radius", "height"}  axis along Z.
- "cone":     "size" {"radius", "height"}  axis along Z, tip toward +Z.
- "pyramid":  "size" {"base", "height"}  square base, axis along Z, tip toward +Z.
- "sphere":   "size" {"radius"}.
Optional "rotation": [rx, ry, rz] in degrees about the part's own centre
(applied X, then Y, then Z). Only rotate parts that are not axis-aligned:
  - cylinder/cone rx=90  -> axis points along Y, toward the camera (round
    buttons, knobs, lenses, clock faces, wheel hubs seen from the front);
    for a cone rx=90 puts the tip toward -Y and rx=-90 toward +Y.
  - cylinder/cone ry=90  -> axis points along X (a horizontal bar, a wheel that
    is seen from the side, a fuselage).
  - rz spins a part around the vertical axis; small rx/ry tilt slanted parts
    (roof planes, an angled screen).

DETAIL AND QUALITY
- Use 15-40 parts for anything with structure (device, house, vehicle, animal,
  furniture). Use fewer only for genuinely simple objects. Never more than 48.
- List the biggest, silhouette-defining parts first, then the details.
- Build in layers. Example, calculator: body, screen bezel, LCD, then the key
  grid as separate small boxes in true rows and columns with even gaps, then a
  solar panel / brand strip / side bevel if visible.
- Details lying on a surface (keys, screens, panels, windows, logos) must
  protrude 0.15-0.3 above it: never flush, and never a stack of hair-thin
  layers, because the viewer sorts those badly. Make a bezel and the screen on
  it each at least 0.15 thick. A key 0.3 thick on a face at y=-0.5 has its
  center at y=-0.65.
- Every part must touch or overlap the part it belongs to. Nothing floats.
  A part of height h resting on a surface at z=s has center z = s + h/2.
- Use cylinders for round parts and boxes for everything else. Use at most 6
  spheres in total (they are expensive to draw).
- Colours: hex "#RRGGBB" sampled from what you see. Give each visibly different
  region its own colour. Never use pure black; use dark charcoal such as
  "#2B2E33". Never use pure white; use "#F1F1EC".
- Hidden sides: give them a plain, sensible colour. Do not invent features you
  cannot see or reasonably infer.

OUTPUT: return ONLY valid JSON (no markdown), with the keys in exactly this order:
{
  "object": "short name, e.g. handheld calculator",
  "plan": "1-2 sentences: the main parts and their x/y/z extents so pieces line up",
  "primitives": [
    {"id": "body", "type": "box", "center": [0, 0, 0],
     "size": {"x": 6, "y": 1, "z": 10}, "color": "#2B2E33"},
    {"id": "knob", "type": "cylinder", "center": [1.5, -0.6, 3],
     "size": {"radius": 0.4, "height": 0.2}, "rotation": [90, 0, 0],
     "color": "#E8833A"}
  ]
}
""".strip()

BUILD_USER_PROMPT = (
    "Rebuild the main object in this photo as a detailed 3D primitive model. "
    "Return only the JSON described in the instructions."
)

DESCRIBE_PROMPT = (
    "Name the main object in this photo and its main colours, plus any distinctive "
    "parts attached to it, in at most 15 words, for example 'red brick house with a "
    "chicken on top'. Do not use the words box, cube, cylinder, cone, sphere, ball, "
    "block, pyramid or can. Reply with only that phrase."
)

# Gemini helpers
def _make_client(api_key):
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

def _is_quota_error(error):
    return (
        getattr(error, "status_code", None) == 429
        or "quota" in str(error).lower()
        or "rate limit" in str(error).lower()
    )

def _is_retryable_model_error(error):
    message = str(error).lower()
    status_code = getattr(error, "status_code", None)
    return (
        _is_quota_error(error)
        or status_code in {500, 502, 503, 504}
        or "overloaded" in message
        or "temporarily unavailable" in message
        or "resource exhausted" in message
        or "service unavailable" in message
    )

def _chat(messages, model, max_tokens, temperature=0.2, json_mode=True):
    # Rotates through every configured API key on retryable errors.
    last_error = None
    for attempt in range(2):
        for key in GEMINI_KEYS:
            try:
                kwargs = {}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = _make_client(key).chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                return content
            except Exception as error:
                last_error = error
                if not _is_retryable_model_error(error):
                    raise RuntimeError(f"Gemini request failed: {error}") from error
        if attempt == 0:
            time.sleep(2.0)
    raise RuntimeError(
        "Gemini request failed after trying all configured keys: "
        f"{last_error}"
    ) from last_error

def _parse_json(text):
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start:end + 1])
        raise

def _encode_frame(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, IMAGE_MAX_SIDE / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("The captured frame could not be encoded.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

def _image_message(text, data_url):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

# Turning the model's JSON into clean parts
_TYPE_ALIASES = {
    "box": "cube", "cube": "cube", "cuboid": "cube", "block": "cube",
    "rect": "cube", "slab": "cube", "plane": "cube",
    "cylinder": "cylinder", "cone": "cone", "pyramid": "pyramid",
    "sphere": "sphere", "ball": "sphere",
}

def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _vec3(value, keys):
    if isinstance(value, dict):
        values = [_as_float(value.get(key)) for key in keys]
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        values = [_as_float(item) for item in value[:3]]
    else:
        return None
    return None if any(item is None for item in values) else values

def _rotation_of(raw):
    value = raw.get("rotation")
    if isinstance(value, dict):
        vec = _vec3(value, ("rx", "ry", "rz")) or _vec3(value, ("x", "y", "z"))
    else:
        vec = _vec3(value, ())
    return tuple(vec) if vec else (0.0, 0.0, 0.0)

def _dims_of(ptype, raw):
    size = raw.get("size", raw.get("dimensions"))
    source = {}
    if isinstance(size, dict):
        source.update(size)
    elif isinstance(size, (list, tuple)):
        values = [_as_float(item) for item in size]
        if ptype == "cube" and len(values) >= 3:
            source = dict(zip(("x", "y", "z"), values[:3]))
        elif ptype in ("cylinder", "cone") and len(values) >= 2:
            source = {"radius": values[0], "height": values[1]}
        elif ptype == "pyramid" and len(values) >= 2:
            source = {"base": values[0], "height": values[1]}
        elif ptype == "sphere" and values:
            source = {"radius": values[0]}
    for key in ("x", "y", "z", "dx", "dy", "dz", "radius", "height", "base",
                "diameter", "width", "depth"):
        if key not in source and key in raw:
            source[key] = raw[key]

    def get(*names):
        for name in names:
            value = _as_float(source.get(name))
            if value is not None:
                return value
        return None

    def radius():
        value = get("radius")
        if value is None:
            diameter = get("diameter")
            value = diameter / 2 if diameter is not None else None
        return value

    if ptype == "cube":
        dims = {"dx": get("x", "dx", "width"), "dy": get("y", "dy", "depth"),
                "dz": get("z", "dz", "height")}
    elif ptype in ("cylinder", "cone"):
        dims = {"radius": radius(), "height": get("height", "z")}
    elif ptype == "pyramid":
        dims = {"base": get("base", "x", "width"), "height": get("height", "z")}
    else:
        dims = {"radius": radius()}
    if any(value is None or value <= 0 for value in dims.values()):
        return None
    return {key: max(0.01, round(value, 4)) for key, value in dims.items()}

def _coerce_parts(data):
    # Validate model JSON and return clean internal part dicts.
    raw_parts = data.get("primitives") if isinstance(data, dict) else None
    if not isinstance(raw_parts, list):
        return []
    parts, seen = [], set()
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            continue
        ptype = _TYPE_ALIASES.get(str(raw.get("type", "")).lower().strip())
        center = _vec3(raw.get("center", raw.get("position")), ("x", "y", "z"))
        if ptype is None or center is None:
            continue
        dims = _dims_of(ptype, raw)
        if dims is None:
            continue
        part_id = str(raw.get("id") or f"part_{index}")
        if part_id in seen:
            part_id = f"{part_id}_{index}"
        seen.add(part_id)
        parts.append({
            "id": part_id,
            "type": ptype,
            "dims": dims,
            "center": np.array(center, dtype=float),
            "rot": _rotation_of(raw),
            "color": str(raw.get("color") or "gray").strip(),
        })
        if len(parts) >= MAX_PARTS:
            break
    return parts

# Geometry audit: bounding boxes, floating parts, tiny specks
def _rotation_matrix(rx, ry, rz):
    # Same convention as shapes.rotation_matrix: Rz @ Ry @ Rx, degrees.
    rx, ry, rz = np.radians([rx, ry, rz])
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])
    return rot_z @ rot_y @ rot_x

def _half_extents(part):
    matrix = _rotation_matrix(*part["rot"])
    dims = part["dims"]
    if part["type"] == "cube":
        half = np.array([dims["dx"], dims["dy"], dims["dz"]]) / 2
        return np.abs(matrix) @ half
    if part["type"] == "sphere":
        return np.full(3, dims["radius"])
    if part["type"] == "pyramid":
        half = np.array([dims["base"] / 2, dims["base"] / 2, dims["height"] / 2])
        return np.abs(matrix) @ half
    axis = matrix[:, 2]  # cylinder / cone
    return (np.abs(axis) * dims["height"] / 2
            + dims["radius"] * np.sqrt(np.clip(1 - axis ** 2, 0, 1)))

def _boxes(parts):
    halves = [_half_extents(part) for part in parts]
    lows = np.array([p["center"] - h for p, h in zip(parts, halves)])
    highs = np.array([p["center"] + h for p, h in zip(parts, halves)])
    return lows, highs

def _span(parts):
    lows, highs = _boxes(parts)
    return float(max((highs.max(axis=0) - lows.min(axis=0)).max(), 1e-6))

def _gap_vector(low_a, high_a, low_b, high_b):
    # Signed per-axis move that brings box A up against box B (0 if overlapping).
    gap = np.zeros(3)
    for axis in range(3):
        if low_b[axis] > high_a[axis]:
            gap[axis] = low_b[axis] - high_a[axis]
        elif low_a[axis] > high_b[axis]:
            gap[axis] = -(low_a[axis] - high_b[axis])
    return gap

def _components(parts, tolerance):
    lows, highs = _boxes(parts)
    parent = list(range(len(parts)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if np.all(lows[i] - tolerance <= highs[j]) and np.all(lows[j] - tolerance <= highs[i]):
                parent[find(i)] = find(j)
    groups = {}
    for index in range(len(parts)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())

def _closest_gap(parts, group, main):
    lows, highs = _boxes(parts)
    best = None
    for i in group:
        for j in main:
            gap = _gap_vector(lows[i], highs[i], lows[j], highs[j])
            distance = float(np.linalg.norm(gap))
            if best is None or distance < best[0]:
                best = (distance, gap)
    return best

def _volume(part):
    return float(np.prod(2 * _half_extents(part)))

def _floating_groups(parts):
    # Groups of parts that do not touch the main body, with their gap size.
    if len(parts) < 2:
        return []
    span = _span(parts)
    groups = _components(parts, 0.02 * span)
    if len(groups) < 2:
        return []
    main = max(groups, key=lambda g: sum(_volume(parts[i]) for i in g))
    result = []
    for group in groups:
        if group is main:
            continue
        distance, gap = _closest_gap(parts, group, main)
        result.append((group, distance, gap))
    return result

def _find_issues(parts):
    # Human-readable problems worth sending back to the model.
    issues = []
    if not parts:
        return ["no valid parts were found"]
    span = _span(parts)
    if len(parts) < MIN_DETAIL_PARTS:
        issues.append(
            f"the model has only {len(parts)} parts. Add the real visible features "
            "(screens, buttons, handles, windows, legs, colour regions) unless the "
            "object truly is a single simple shape"
        )
    for group, distance, _gap in _floating_groups(parts):
        if distance > 0.12 * span:
            names = ", ".join(parts[i]["id"] for i in group[:4])
            issues.append(
                f"part(s) {names} float about {distance:.1f} units away from the "
                "rest of the model; fix their center so they touch the object"
            )
    return issues

def _connect_floaters(parts, max_gap_ratio=0.4):
    # Slide detached parts along the shortest path until they touch the model.
    span = _span(parts) if parts else 1.0
    for _ in range(len(parts)):
        groups = _floating_groups(parts)
        movable = [g for g in groups if g[1] <= max_gap_ratio * span]
        if not movable:
            break
        group, _distance, gap = min(movable, key=lambda item: item[1])
        for index in group:
            parts[index]["center"] = parts[index]["center"] + gap
    return parts

def _drop_specks(parts):
    if len(parts) < 2:
        return parts
    span = _span(parts)
    kept = [p for p in parts if float((2 * _half_extents(p)).max()) >= 0.01 * span]
    return kept or parts

# Convert to the schema shapes.py renders
def _to_shapes_layout(parts):
    # Freeze absolute positions: every part is centred on the root plus an offset.
    # shapes.resolve_primitive_positions() then reproduces exactly the audited positions.
    root = max(parts, key=_volume)
    ordered = [root] + [part for part in parts if part is not root]
    primitives = []
    for part in ordered:
        rx, ry, rz = part["rot"]
        item = {
            "id": part["id"],
            "type": part["type"],
            "dimensions": dict(part["dims"]),
            "color": part["color"],
            "rotation": {"rx": rx, "ry": ry, "rz": rz},
        }
        if part is root:
            item.update(attach_to=None, position="center",
                        offset_x=0.0, offset_y=0.0, offset_z=0.0)
        else:
            delta = part["center"] - root["center"]
            item.update(attach_to=root["id"], position="center",
                        offset_x=round(float(delta[0]), 4),
                        offset_y=round(float(delta[1]), 4),
                        offset_z=round(float(delta[2]), 4))
        primitives.append(item)
    return {"primitives": primitives, "source": "camera"}

# Build pipeline: photo -> model JSON -> audited layout
def _request_model(messages, max_tokens=12000):
    # One Gemini call that must return a parseable JSON object.
    raw = _chat(messages, BUILD_MODEL_NAME, max_tokens)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        retry = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": (
                "That reply was cut off or invalid JSON. Return the complete JSON "
                "again with at most 28 parts and no extra text."
            )},
        ]
        return _parse_json(_chat(retry, BUILD_MODEL_NAME, max_tokens, temperature=0.0))

def build_model_layout(frame):
    # Return (layout_for_shapes, title) for the main object in a BGR frame.
    data_url = _encode_frame(frame)
    messages = [
        {"role": "system", "content": BUILD_SYSTEM_PROMPT},
        _image_message(BUILD_USER_PROMPT, data_url),
    ]
    data = _request_model(messages)
    title = str(data.get("object") or "object").strip() if isinstance(data, dict) else "object"
    if title.lower() in {"none", "nothing", "unknown", "no object"}:
        raise RuntimeError("No clear object was found in the frame.")
    parts = _coerce_parts(data)
    issues = _find_issues(parts)
    print(f"[camera] draft: '{title}' with {len(parts)} parts; issues: {issues or 'none'}")

    for round_number in range(MAX_REFINE_ROUNDS):
        if not issues:
            break
        repair_text = (
            "Here is your previous model:\n"
            + json.dumps(data, separators=(",", ":"))
            + "\n\nProblems found:\n- " + "\n- ".join(issues)
            + "\n\nLook at the photo again and return the corrected, complete JSON "
            "in the same schema. Keep everything that was already right."
        )
        try:
            new_data = _request_model([
                {"role": "system", "content": BUILD_SYSTEM_PROMPT},
                _image_message(repair_text, data_url),
            ])
        except Exception as error:
            print(f"[camera] repair pass {round_number + 1} failed: {error}")
            break
        new_parts = _coerce_parts(new_data)
        new_issues = _find_issues(new_parts)
        print(f"[camera] repair {round_number + 1}: {len(new_parts)} parts; "
              f"issues: {new_issues or 'none'}")
        if new_parts and len(new_issues) <= len(issues):
            data, parts, issues = new_data, new_parts, new_issues

    if not parts:
        raise RuntimeError("Gemini did not return any usable 3D parts.")
    parts = _drop_specks(_connect_floaters(parts))
    layout = _to_shapes_layout(parts)
    return layout, title[:48]

def describe_frame(frame):
    # Short keyword-safe description, used only if the direct build fails.
    text = _chat(
        [_image_message(DESCRIBE_PROMPT, _encode_frame(frame))],
        MODEL_NAME, 120, temperature=0.1, json_mode=False,
    )
    return re.sub(r"\s+", " ", text).strip().strip("\"'.")

# Viewer polish (worker process only; shapes.py itself is not modified)
_LIGHT = np.array([-0.35, -0.55, 0.75])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

def _face_normal(face):
    points = np.asarray(face, dtype=float)
    nxt = np.roll(points, -1, axis=0)
    normal = np.array([
        np.sum((points[:, 1] - nxt[:, 1]) * (points[:, 2] + nxt[:, 2])),
        np.sum((points[:, 2] - nxt[:, 2]) * (points[:, 0] + nxt[:, 0])),
        np.sum((points[:, 0] - nxt[:, 0]) * (points[:, 1] + nxt[:, 1])),
    ])
    length = np.linalg.norm(normal)
    return normal / length if length else np.array([0.0, 0.0, 1.0])

def _tile_triangle(face, cell, depth=0):
    # Midpoint-subdivide a large triangle into 4^n pieces for better depth sorting.
    p = [np.asarray(vertex, dtype=float) for vertex in face]
    longest = max(np.linalg.norm(p[(i + 1) % 3] - p[i]) for i in range(3))
    if longest <= 3 * cell or depth >= 3:
        return [face]
    m01, m12, m20 = (p[0] + p[1]) / 2, (p[1] + p[2]) / 2, (p[2] + p[0]) / 2
    pieces = []
    for corner in ([p[0], m01, m20], [m01, p[1], m12], [m20, m12, p[2]], [m01, m12, m20]):
        pieces.extend(_tile_triangle(corner, cell, depth + 1))
    return pieces

def _tile_quad(face, cell):
    # Split a large quad or triangle into small tiles (other polygons kept).
    if len(face) == 3:
        return _tile_triangle(face, cell)
    if len(face) != 4:
        return [face]
    p = np.asarray(face, dtype=float)
    edge_u, edge_v = p[1] - p[0], p[3] - p[0]
    if max(np.linalg.norm(edge_u), np.linalg.norm(edge_v)) <= 3 * cell:
        return [face]
    count_u = min(16, int(math.ceil(np.linalg.norm(edge_u) / cell)))
    count_v = min(16, int(math.ceil(np.linalg.norm(edge_v) / cell)))
    if count_u <= 1 and count_v <= 1:
        return [face]
    tiles = []
    for i in range(max(count_u, 1)):
        for j in range(max(count_v, 1)):
            a, b = i / max(count_u, 1), (i + 1) / max(count_u, 1)
            c, d = j / max(count_v, 1), (j + 1) / max(count_v, 1)
            tiles.append([
                p[0] + a * edge_u + c * edge_v,
                p[0] + b * edge_u + c * edge_v,
                p[0] + b * edge_u + d * edge_v,
                p[0] + a * edge_u + d * edge_v,
            ])
    return tiles

def install_shading(shapes_module):
    # Make the viewer draw detailed models correctly (worker process only).
    # 1. Lighting: each face gets a lit colour so boxes/cylinders read as 3D solids.
    # 2. Depth sorting: split large faces into tiles so small parts on them sort correctly.
    import matplotlib.colors as mcolors

    original = shapes_module._solid_collection
    cache = {}

    def base_color(color):
        if color not in cache:
            rgb = np.array(mcolors.to_rgb(color))
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            if luminance < 0.28:  # lift near-black so it stays visible on the dark panel
                rgb = np.clip(rgb + (0.28 - luminance), 0, 1)
            cache[color] = rgb
        return cache[color]

    def shaded_collection(faces, colors):
        cloud = np.concatenate([np.asarray(face, dtype=float) for face in faces])
        span = float((cloud.max(axis=0) - cloud.min(axis=0)).max())
        tiled_faces, tiled_colors, seamless = [], [], []
        for face, color in zip(faces, colors):
            pieces = _tile_quad(face, span / 24)
            tiled_faces.extend(pieces)
            tiled_colors.extend([color] * len(pieces))
            seamless.extend([len(pieces) > 1] * len(pieces))
        if len(tiled_faces) > 12000:  # keep rotation smooth on huge models
            tiled_faces, tiled_colors = list(faces), list(colors)
            seamless = [False] * len(tiled_faces)
        shaded = []
        for face, color in zip(tiled_faces, tiled_colors):
            shade = 0.45 + 0.55 * abs(float(_face_normal(face) @ _LIGHT))
            shaded.append(tuple(np.clip(base_color(color) * shade, 0, 1)))
        collection = original(tiled_faces, shaded)
        try:  # tile seams get the face colour so only real edges keep an outline
            outline = mcolors.to_rgba("#20242b")
            collection.set_edgecolor([
                (*rgb, 1.0) if hidden else outline
                for rgb, hidden in zip(shaded, seamless)
            ])
        except Exception:
            pass
        return collection

    shapes_module._solid_collection = shaded_collection

# Worker process: build the model and open the viewer
def run_worker(image_path, ready_path):
    # imdecode + fromfile (not imread) so non-ASCII Windows user paths work.
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read the captured frame at {image_path}")

    import shapes  # heavy imports (Matplotlib) stay out of the GUI process

    if ENABLE_SHADING:
        try:
            install_shading(shapes)
        except Exception as error:  # cosmetic only
            print(f"[camera] shading disabled: {error}")

    layout, title = None, "object"
    try:
        layout, title = build_model_layout(frame)
        errors = shapes._validate_layout(layout)
        if errors:
            raise ValueError("; ".join(errors))
    except Exception as error:
        layout = None
        print(f"[camera] direct build failed: {error}")
        traceback.print_exc()
        # Fallback: a short, keyword-safe phrase through shapes.py's text pipeline.
        title = describe_frame(frame) or "object"
        print(f"[camera] falling back to text pipeline with: {title!r}")
        layout = shapes.parse_description_with_gemini(title)

    plt = shapes.plt
    original_show = plt.show

    def show_and_signal(*args, **kwargs):
        try:
            Path(ready_path).write_text("ready", encoding="utf-8")
        except OSError:
            pass
        return original_show(*args, **kwargs)

    plt.show = show_and_signal
    shapes.display_diagram(title, layout)

# GUI
class CameraTo3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OTTO - Camera to 3D")
        self.root.configure(bg="#080B12")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(960, max(760, screen_width - 80))
        window_height = min(700, max(560, screen_height - 120))
        left = max(0, (screen_width - window_width) // 2)
        top = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{left}+{top}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.capture = None
        self.current_frame = None
        self.recent_frames = deque(maxlen=6)  # (sharpness, frame)
        self.running = True
        self.busy = False
        self.spinner_step = 0
        self.overlay = None
        self.image_reference = None
        self.model_process = None

        self._build_ui()
        self._open_camera()
        self._refresh_frame()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0D111C", height=58)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="CAMERA TO 3D",
            bg="#0D111C",
            fg="#67E8F9",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left", padx=22, pady=14)
        tk.Label(
            header,
            text="CAPTURE AN OBJECT TO BUILD A 3D DUPLICATE",
            bg="#0D111C",
            fg="#94A3B8",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=4)

        self.preview = tk.Canvas(
            self.root,
            width=1,
            height=1,
            bg="#02040A",
            highlightbackground="#263244",
            highlightthickness=1,
        )
        self.preview.pack(expand=True, fill="both", padx=28, pady=(22, 8))
        self.preview.bind("<Configure>", self._redraw_current_frame)

        controls = tk.Frame(self.root, bg="#080B12", height=92)
        controls.pack(fill="x", pady=(0, 16))
        controls.pack_propagate(False)
        self.capture_button = tk.Canvas(
            controls,
            width=72,
            height=72,
            bg="#080B12",
            highlightthickness=0,
            cursor="hand2",
        )
        self.capture_button.pack(pady=8)
        self.capture_button.create_oval(
            6, 6, 66, 66, fill="#EF4444", outline="#FCA5A5", width=2
        )
        self.capture_button.create_oval(
            19, 19, 53, 53, fill="#F8FAFC", outline=""
        )
        self.capture_button.bind("<Button-1>", self.capture_frame)

        self.status = tk.Label(
            self.root,
            text="LIVE  /  FRAME THE CLOSEST OBJECT, THEN CAPTURE",
            bg="#080B12",
            fg="#CBD5E1",
            font=("Segoe UI", 10),
        )
        self.status.pack(pady=(0, 14))

    def _open_camera(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.status.config(
                text="Camera could not be opened. Check the webcam and CAMERA_INDEX.",
                fg="#FCA5A5",
            )

    def _release_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _reopen_camera(self):
        if not self.running:
            return
        self._open_camera()

    @staticmethod
    def _sharpness(frame):
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _refresh_frame(self):
        if not self.running:
            return
        if not self.busy and self.capture is not None and self.capture.isOpened():
            ok, frame = self.capture.read()
            if ok:
                self.current_frame = frame
                self.recent_frames.append((self._sharpness(frame), frame))
                self._redraw_current_frame()
        self.root.after(30, self._refresh_frame)

    def _redraw_current_frame(self, _event=None):
        if self.current_frame is None or not self.preview.winfo_exists():
            return
        frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        target_width = max(self.preview.winfo_width(), 1)
        target_height = max(self.preview.winfo_height(), 1)
        scale = min(target_width / width, target_height / height)
        resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.image_reference = image
        self.preview.delete("webcam_frame")
        self.preview.create_image(
            target_width // 2,
            target_height // 2,
            image=image,
            anchor="center",
            tags="webcam_frame",
        )

    def capture_frame(self, _event=None):
        if self.busy or self.current_frame is None:
            return
        self.busy = True
        # Pick the sharpest of the last few frames to avoid motion blur.
        if self.recent_frames:
            frozen_frame = max(self.recent_frames, key=lambda item: item[0])[1].copy()
        else:
            frozen_frame = self.current_frame.copy()
        self.status.config(
            text="ANALYZING FRAME - the live display is paused",
            fg="#FBBF24",
        )
        self._show_loading_overlay()
        threading.Thread(
            target=self._analyze_and_generate,
            args=(frozen_frame,),
            daemon=True,
        ).start()

    def _show_loading_overlay(self):
        self.overlay = tk.Frame(self.root, bg="#030712")
        self.overlay.place(
            in_=self.preview,
            relx=0,
            rely=0,
            relwidth=1,
            relheight=1,
        )
        self.loading_label = tk.Label(
            self.overlay,
            text="ANALYZING\n.",
            bg="#030712",
            fg="#67E8F9",
            font=("Segoe UI", 18, "bold"),
            justify="center",
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self._animate_loading()

    def _animate_loading(self):
        if self.overlay is None or not self.overlay.winfo_exists():
            return
        dots = "." * (self.spinner_step % 4)
        self.loading_label.config(text=f"ANALYZING\n{dots}")
        self.spinner_step += 1
        self.root.after(180, self._animate_loading)

    def _analyze_and_generate(self, frame):
        image_path = ready_file = log_path = None
        try:
            token = uuid.uuid4().hex
            temp_dir = Path(tempfile.gettempdir())
            image_path = temp_dir / f"camera_capture_{token}.jpg"
            ready_file = temp_dir / f"shapes_ready_{token}.flag"
            log_path = ready_file.with_suffix(".log")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("The captured frame could not be saved.")
            image_path.write_bytes(encoded.tobytes())

            self._release_camera()
            log_file = open(log_path, "w", encoding="utf-8")
            self.model_process = subprocess.Popen(
                [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker", str(image_path), "--ready", str(ready_file),
                ],
                cwd=str(Path(__file__).resolve().parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            log_file.close()

            deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
            while (
                self.model_process.poll() is None
                and not ready_file.exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            if not ready_file.exists():
                if self.model_process.poll() is None:
                    self.model_process.terminate()
                return_code = self.model_process.wait(timeout=5)
                try:
                    details = log_path.read_text(encoding="utf-8", errors="replace")[-1800:]
                except OSError:
                    details = "No model log was available."
                raise RuntimeError(
                    "The 3D generation process did not display a model "
                    f"(exit code {return_code}).\n{details}"
                )
            ready_file.unlink(missing_ok=True)
            # Wait for the worker's viewer to close before reacquiring camera 0.
            self.model_process.wait()
            self.root.after(
                0,
                lambda: self._finish_capture(
                    "3D model closed. Camera is live again."
                ),
            )
        except Exception as error:
            self.root.after(
                0,
                lambda message=str(error): self._finish_capture(
                    f"Capture failed: {message}", error=True
                ),
            )
        finally:
            for path in (ready_file, image_path, log_path):
                if path is not None:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        pass

    def _finish_capture(self, message, error=False):
        if self.overlay is not None and self.overlay.winfo_exists():
            self.overlay.destroy()
        self.overlay = None
        self.recent_frames.clear()
        self._reopen_camera()
        self.busy = False
        if error:
            # Print the actual error to console; show a neutral message in the UI.
            print(f"Error: {message}")
            self.status.config(text="", fg="#CBD5E1")
        else:
            self.status.config(text=message, fg="#86EFAC")

    def close(self):
        self.running = False
        self._release_camera()
        if self.model_process is not None and self.model_process.poll() is None:
            self.model_process.terminate()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker")
    parser.add_argument("--ready")
    args, _unknown = parser.parse_known_args()

    if args.worker:
        try:
            run_worker(args.worker, args.ready or "")
        except Exception as error:
            traceback.print_exc()
            print(f"ERROR: {error}")
            sys.exit(1)
        return

    root = tk.Tk()
    CameraTo3DApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()