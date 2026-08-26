"""Shared parsers for KiCad PCB board files."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from typing import Any

from ..file_formats import GENERATED_SEXPR_DIALECT_VERSION
from ..utils import telemetry as otel
from ..utils.sexpr import _extract_block

BOARD_FILE_VERSION = GENERATED_SEXPR_DIALECT_VERSION
FLOAT_PATTERN = r"-?\d+(?:\.\d+)?"
STRING_PATTERN = r'"((?:\\.|[^"\\])*)"'


def _default_board_text() -> str:
    return (
        "(kicad_pcb\n"
        f"\t(version {BOARD_FILE_VERSION})\n"
        '\t(generator "kicad-mcp-pro")\n'
        "\t(general)\n"
        '\t(paper "A4")\n'
        ")\n"
    )


def _normalize_board_content(content: str) -> str:
    stripped = content.strip()
    if not stripped or stripped == "(kicad_pcb)":
        return _default_board_text()
    if "(version" not in content:
        return _default_board_text()
    return content


def _parse_root_at(block: str) -> tuple[float, float, int] | None:
    for line in block.splitlines()[:12]:
        match = re.match(
            rf"\s*\(at\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})(?:\s+({FLOAT_PATTERN}))?\)",
            line,
        )
        if match:
            rotation = int(round(float(match.group(3) or "0")))
            return float(match.group(1)), float(match.group(2)), rotation
    return None


def _iter_blocks(content: str, keyword: str) -> Iterable[str]:
    cursor = 0
    marker = f"({keyword}"
    while True:
        start = content.find(marker, cursor)
        if start < 0:
            return
        block, length = _extract_block(content, start)
        if block:
            yield block
            cursor = start + length
        else:
            cursor = start + len(marker)


def _bbox_extents_from_block(block: str) -> tuple[float, float, float, float]:
    """Return local footprint geometry as ``min_x, min_y, max_x, max_y``.

    KiCad footprint origins are not necessarily at the geometric center.  Keeping
    the extents, rather than only width and height, is required for correct edge
    anchoring and collision checks.  Pad rotation is included because an oblong
    pad can otherwise understate one axis of the physical footprint.
    """
    xs: list[float] = []
    ys: list[float] = []

    for rect in re.finditer(
        rf"\(fp_rect\s+\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)\s+\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)",
        block,
    ):
        xs.extend([float(rect.group(1)), float(rect.group(3))])
        ys.extend([float(rect.group(2)), float(rect.group(4))])

    for line in re.finditer(
        rf"\(fp_line\s+\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)\s+\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)",
        block,
    ):
        xs.extend([float(line.group(1)), float(line.group(3))])
        ys.extend([float(line.group(2)), float(line.group(4))])

    for circle in re.finditer(
        rf"\(fp_circle\s+\(center\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)\s+\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)",
        block,
    ):
        center_x = float(circle.group(1))
        center_y = float(circle.group(2))
        end_x = float(circle.group(3))
        end_y = float(circle.group(4))
        radius = math.hypot(end_x - center_x, end_y - center_y)
        xs.extend([center_x - radius, center_x + radius])
        ys.extend([center_y - radius, center_y + radius])

    for pad_block in _iter_blocks(block, "pad"):
        at_match = re.search(
            rf"\(at\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})(?:\s+({FLOAT_PATTERN}))?\)",
            pad_block,
        )
        size_match = re.search(rf"\(size\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", pad_block)
        if at_match and size_match:
            center_x = float(at_match.group(1))
            center_y = float(at_match.group(2))
            width = float(size_match.group(1))
            height = float(size_match.group(2))
            angle = math.radians(float(at_match.group(3) or 0.0))
            half_width = width / 2.0
            half_height = height / 2.0
            rotated_half_width = abs(half_width * math.cos(angle)) + abs(
                half_height * math.sin(angle)
            )
            rotated_half_height = abs(half_width * math.sin(angle)) + abs(
                half_height * math.cos(angle)
            )
            xs.extend([center_x - rotated_half_width, center_x + rotated_half_width])
            ys.extend([center_y - rotated_half_height, center_y + rotated_half_height])

    if not xs or not ys:
        return -2.54, -2.54, 2.54, 2.54

    return min(xs), min(ys), max(xs), max(ys)


def _bbox_from_block(block: str) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = _bbox_extents_from_block(block)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    return round(width, 4), round(height, 4)


def _footprint_net_names(block: str) -> list[str]:
    names: set[str] = set()
    for pad_block in _iter_blocks(block, "pad"):
        match = re.search(rf"\(net(?:\s+\d+)?\s+{STRING_PATTERN}\)", pad_block)
        if match is not None and match.group(1):
            names.add(match.group(1))
    return sorted(names)


def _footprint_pad_net_map(block: str) -> dict[str, str]:
    pad_map: dict[str, str] = {}
    for pad_block in _iter_blocks(block, "pad"):
        pad_match = re.match(rf"\(pad\s+{STRING_PATTERN}", pad_block.lstrip())
        net_match = re.search(rf"\(net(?:\s+\d+)?\s+{STRING_PATTERN}\)", pad_block)
        if pad_match is None or net_match is None:
            continue
        if not pad_match.group(1) or not net_match.group(1):
            continue
        pad_map[pad_match.group(1)] = net_match.group(1)
    return pad_map


def _parse_board_footprint_blocks(content: str) -> dict[str, dict[str, Any]]:
    started = time.perf_counter()
    footprints: dict[str, dict[str, Any]] = {}
    with otel.pcb_parse_span() as span:
        cursor = 0
        while cursor < len(content):
            if content[cursor:].startswith("(footprint"):
                block, length = _extract_block(content, cursor)
                if block:
                    ref_match = re.search(rf'\(property\s+"Reference"\s+{STRING_PATTERN}', block)
                    value_match = re.search(rf'\(property\s+"Value"\s+{STRING_PATTERN}', block)
                    name_match = re.match(rf"\(footprint\s+{STRING_PATTERN}", block.lstrip())
                    if ref_match and name_match:
                        root_at = _parse_root_at(block)
                        min_x, min_y, max_x, max_y = _bbox_extents_from_block(block)
                        width_mm = max(max_x - min_x, 1.0)
                        height_mm = max(max_y - min_y, 1.0)
                        layer_match = re.search(r'\(layer\s+"([^"]+)"\)', block)
                        footprints[ref_match.group(1)] = {
                            "name": name_match.group(1),
                            "block": block,
                            "start": cursor,
                            "end": cursor + length,
                            "value": value_match.group(1) if value_match else "",
                            "x_mm": root_at[0] if root_at else None,
                            "y_mm": root_at[1] if root_at else None,
                            "rotation": root_at[2] if root_at else 0,
                            "width_mm": round(width_mm, 4),
                            "height_mm": round(height_mm, 4),
                            "bbox_min_x_mm": round(min_x, 4),
                            "bbox_min_y_mm": round(min_y, 4),
                            "bbox_max_x_mm": round(max_x, 4),
                            "bbox_max_y_mm": round(max_y, 4),
                            "bbox_center_x_mm": round((min_x + max_x) / 2.0, 4),
                            "bbox_center_y_mm": round((min_y + max_y) / 2.0, 4),
                            "layer_name": layer_match.group(1) if layer_match else "F.Cu",
                            "net_names": _footprint_net_names(block),
                            "pad_nets": _footprint_pad_net_map(block),
                        }
                    cursor += length
                    continue
            cursor += 1
        otel.finish_pcb_parse_span(
            span,
            footprint_count=len(footprints),
            elapsed_seconds=time.perf_counter() - started,
        )
        return footprints


def _edge_cuts_bounds(content: str) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    patterns = [
        rf"\(gr_line\s+\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)\s+\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\).*?\(layer\s+\"Edge\.Cuts\"\)",
        rf"\(gr_rect\s+\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)\s+\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\).*?\(layer\s+\"Edge\.Cuts\"\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, content, flags=re.DOTALL):
            xs.extend([float(match.group(1)), float(match.group(3))])
            ys.extend([float(match.group(2)), float(match.group(4))])
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _board_frame_mm(
    content: str,
    footprints: dict[str, dict[str, Any]],
) -> tuple[float, float, float, float]:
    if (outline := _edge_cuts_bounds(content)) is not None:
        return outline

    xs: list[float] = []
    ys: list[float] = []
    for entry in footprints.values():
        if entry["x_mm"] is None or entry["y_mm"] is None:
            continue
        x_mm = float(entry["x_mm"])
        y_mm = float(entry["y_mm"])
        width_mm = float(entry["width_mm"])
        height_mm = float(entry["height_mm"])
        xs.extend([x_mm - (width_mm / 2), x_mm + (width_mm / 2)])
        ys.extend([y_mm - (height_mm / 2), y_mm + (height_mm / 2)])
    if xs and ys:
        return min(xs) - 10.0, min(ys) - 10.0, max(xs) + 10.0, max(ys) + 10.0
    return 0.0, 0.0, 100.0, 80.0


def _placement_boxes_overlap(
    x1_mm: float,
    y1_mm: float,
    width1_mm: float,
    height1_mm: float,
    x2_mm: float,
    y2_mm: float,
    width2_mm: float,
    height2_mm: float,
    margin_mm: float,
) -> bool:
    return (
        abs(x1_mm - x2_mm) < ((width1_mm + width2_mm) / 2) + margin_mm
        and abs(y1_mm - y2_mm) < ((height1_mm + height2_mm) / 2) + margin_mm
    )
