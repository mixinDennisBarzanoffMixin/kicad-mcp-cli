"""Bounded, geometry-first text views of one KiCad schematic sheet.

The deep project map is intentionally semantic: its zoom levels describe the
hierarchy, BOM, nets, and PCB.  This module complements it with a deterministic
spatial view read directly from one saved ``.kicad_sch`` page.  It is kept
separate so the existing map output remains a stable shell interface.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils.sexpr import _extract_block

type JsonRecord = dict[str, Any]

_FLOAT = r"-?\d+(?:\.\d+)?"
_STRING = r'"((?:\\.|[^"\\])*)"'
_SPATIAL_KEYWORDS = frozenset({"symbol", "label", "global_label", "hierarchical_label", "wire"})


@dataclass(frozen=True)
class SpatialSymbol:
    reference: str
    value: str
    x_mm: float
    y_mm: float
    rotation: int


@dataclass(frozen=True)
class SpatialLabel:
    name: str
    kind: str
    x_mm: float
    y_mm: float
    rotation: int
    uuid: str = ""


@dataclass(frozen=True)
class SpatialWire:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SchematicGeometry:
    symbols: tuple[SpatialSymbol, ...]
    labels: tuple[SpatialLabel, ...]
    wires: tuple[SpatialWire, ...]


def _unescape(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def _root_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Yield interesting direct children of the ``kicad_sch`` root.

    Looking for ``(symbol`` with a plain regex also finds cached library symbol
    definitions.  Tracking only root-level forms cleanly separates those from
    placed instances without requiring a full S-expression object tree.
    """

    depth = 0
    cursor = 0
    in_string = False
    escaped = False
    while cursor < len(text):
        character = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            cursor += 1
            continue
        if character == '"':
            in_string = True
            cursor += 1
            continue
        if character == "(":
            if depth == 1:
                keyword_match = re.match(r"\(([A-Za-z_][A-Za-z0-9_]*)", text[cursor:])
                if keyword_match and keyword_match.group(1) in _SPATIAL_KEYWORDS:
                    block, length = _extract_block(text, cursor)
                    if block and length:
                        yield keyword_match.group(1), block
                        cursor += length
                        continue
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        cursor += 1


def _root_at(block: str) -> tuple[float, float, int] | None:
    match = re.search(
        rf"^\([A-Za-z_][A-Za-z0-9_]*(?:\s+{_STRING})?\s+.*?"
        rf"\(at\s+({_FLOAT})\s+({_FLOAT})(?:\s+({_FLOAT}))?\)",
        block,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    # _STRING contributes capture group 1 when the root form has text.
    groups = match.groups()
    offset = 1 if len(groups) == 4 else 0
    return (
        float(groups[offset]),
        float(groups[offset + 1]),
        int(round(float(groups[offset + 2] or 0))),
    )


def parse_schematic_geometry(text: str) -> SchematicGeometry:
    """Parse placed symbols, named labels, and wire polylines from one sheet."""

    symbols: list[SpatialSymbol] = []
    labels: list[SpatialLabel] = []
    wires: list[SpatialWire] = []
    for kind, block in _root_blocks(text):
        if kind == "symbol":
            reference = re.search(rf'\(property\s+"Reference"\s+{_STRING}', block)
            value = re.search(rf'\(property\s+"Value"\s+{_STRING}', block)
            at = _root_at(block)
            if reference is None or at is None:
                continue
            symbols.append(
                SpatialSymbol(
                    reference=_unescape(reference.group(1)),
                    value=_unescape(value.group(1)) if value else "",
                    x_mm=at[0],
                    y_mm=at[1],
                    rotation=at[2],
                )
            )
        elif kind in {"label", "global_label", "hierarchical_label"}:
            name = re.match(rf"\({kind}\s+{_STRING}", block)
            at = _root_at(block)
            if name is not None and at is not None:
                uuid_match = re.search(r'\(uuid\s+"([^"]+)"\)', block)
                labels.append(
                    SpatialLabel(
                        name=_unescape(name.group(1)),
                        kind=kind,
                        x_mm=at[0],
                        y_mm=at[1],
                        rotation=at[2],
                        uuid=uuid_match.group(1) if uuid_match else "",
                    )
                )
        elif kind == "wire":
            points = tuple(
                (float(x), float(y))
                for x, y in re.findall(rf"\(xy\s+({_FLOAT})\s+({_FLOAT})\)", block)
            )
            if len(points) >= 2:
                wires.append(SpatialWire(points=points))
    return SchematicGeometry(
        symbols=tuple(sorted(symbols, key=lambda item: item.reference)),
        labels=tuple(sorted(labels, key=lambda item: (item.name, item.x_mm, item.y_mm))),
        wires=tuple(wires),
    )


def select_schematic_sheet(snapshot: JsonRecord, sheet: str, center_reference: str) -> JsonRecord:
    """Resolve exactly one saved page by sheet substring or component reference."""
    sheets = list(snapshot["schematic"]["sheets"])
    if sheet:
        query = sheet.casefold()
        exact = [
            item
            for item in sheets
            if query
            in {
                str(item.get("name", "")).strip("/").casefold(),
                Path(str(item.get("file", ""))).stem.casefold(),
            }
        ]
        matches = exact or [
            item
            for item in sheets
            if query in str(item.get("name", "")).casefold()
            or query in str(item.get("file", "")).casefold()
        ]
    elif center_reference:
        component = next(
            (
                item
                for item in snapshot["schematic"]["components"]
                if str(item.get("reference", "")).casefold() == center_reference.casefold()
            ),
            None,
        )
        if component is None:
            raise ValueError(
                f"center reference was not found in the project snapshot: {center_reference}"
            )
        component_sheet = str(component.get("sheet", "")) if component else ""
        matches = [item for item in sheets if str(item.get("name", "")) == component_sheet]
    else:
        matches = [item for item in sheets if str(item.get("name", "")) == "/"]
        if not matches and sheets:
            matches = [sheets[0]]
    if not matches:
        target = sheet or center_reference
        raise ValueError(f"no schematic page matched spatial target: {target}")
    if len(matches) > 1:
        names = ", ".join(str(item.get("name") or item.get("file")) for item in matches)
        raise ValueError(f"spatial target is ambiguous: {names}")
    return matches[0]


def _extent(geometry: SchematicGeometry) -> tuple[float, float, float, float]:
    points = [(item.x_mm, item.y_mm) for item in (*geometry.symbols, *geometry.labels)] + [
        point for wire in geometry.wires for point in wire.points
    ]
    if not points:
        return (0.0, 0.0, 100.0, 60.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = 5.0
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _viewport(
    bounds: tuple[float, float, float, float],
    geometry: SchematicGeometry,
    zoom: int,
    center_reference: str,
) -> tuple[float, float, float, float]:
    if zoom == 0:
        return bounds
    focus = next(
        (
            (symbol.x_mm, symbol.y_mm)
            for symbol in geometry.symbols
            if symbol.reference.casefold() == center_reference.casefold()
        ),
        None,
    )
    if center_reference and focus is None:
        raise ValueError(f"center reference was not found on selected page: {center_reference}")
    x1, y1, x2, y2 = bounds
    if focus is None:
        focus = ((x1 + x2) / 2, (y1 + y2) / 2)
    scale = 1.0 / (1.0 + zoom * 0.75)
    half_width = max(8.0, (x2 - x1) * scale / 2)
    half_height = max(6.0, (y2 - y1) * scale / 2)
    return (
        focus[0] - half_width,
        focus[1] - half_height,
        focus[0] + half_width,
        focus[1] + half_height,
    )


def _zone(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bounds
    column = min(3, max(0, int((point[0] - x1) / max(x2 - x1, 0.001) * 4)))
    row = min(3, max(0, int((point[1] - y1) / max(y2 - y1, 0.001) * 4)))
    return f"{chr(ord('A') + column)}{row + 1}"


def _truncate(value: str, width: int) -> str:
    return value if len(value) <= width else value[: max(1, width - 1)] + "…"


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a line to the viewport with the Liang-Barsky algorithm."""

    x1, y1 = start
    x2, y2 = end
    left, top, right, bottom = bounds
    dx, dy = x2 - x1, y2 - y1
    lower, upper = 0.0, 1.0
    for denominator, numerator in (
        (-dx, x1 - left),
        (dx, right - x1),
        (-dy, y1 - top),
        (dy, bottom - y1),
    ):
        if denominator == 0:
            if numerator < 0:
                return None
            continue
        ratio = numerator / denominator
        if denominator < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return ((x1 + lower * dx, y1 + lower * dy), (x1 + upper * dx, y1 + upper * dy))


def schematic_spatial_map(
    snapshot: JsonRecord,
    project_dir: str | Path,
    *,
    zoom: int = 0,
    width: int = 100,
    height: int = 32,
    sheet: str = "",
    center_reference: str = "",
) -> str:
    """Render one bounded schematic page with geometry and local relationships."""

    if zoom not in range(4):
        raise ValueError("zoom must be between 0 and 3")
    columns = max(48, min(int(width), 200))
    rows = max(12, min(int(height), 80))
    selected_sheet = select_schematic_sheet(snapshot, sheet, center_reference)
    source = Path(project_dir).expanduser().resolve() / str(selected_sheet["file"])
    if not source.is_file():
        raise FileNotFoundError(f"schematic page is missing: {source}")
    geometry = parse_schematic_geometry(source.read_text(encoding="utf-8", errors="ignore"))
    full_bounds = _extent(geometry)
    view = _viewport(full_bounds, geometry, zoom, center_reference)
    vx1, vy1, vx2, vy2 = view
    grid = [[" " for _ in range(columns)] for _ in range(rows)]

    def visible(point: tuple[float, float]) -> bool:
        return vx1 <= point[0] <= vx2 and vy1 <= point[1] <= vy2

    def cell(point: tuple[float, float]) -> tuple[int, int]:
        column = round((point[0] - vx1) / max(vx2 - vx1, 0.001) * (columns - 1))
        row = round((point[1] - vy1) / max(vy2 - vy1, 0.001) * (rows - 1))
        return max(0, min(columns - 1, column)), max(0, min(rows - 1, row))

    def stroke(start: tuple[float, float], end: tuple[float, float]) -> None:
        start_column, start_row = cell(start)
        end_column, end_row = cell(end)
        steps = max(abs(end_column - start_column), abs(end_row - start_row), 1)
        horizontal = abs(end_column - start_column) >= abs(end_row - start_row) * 2
        vertical = abs(end_row - start_row) >= abs(end_column - start_column) * 2
        character = "─" if horizontal else "│" if vertical else "·"
        for index in range(steps + 1):
            column = round(start_column + (end_column - start_column) * index / steps)
            row = round(start_row + (end_row - start_row) * index / steps)
            if 0 < column < columns - 1 and 0 < row < rows - 1:
                old = grid[row][column]
                grid[row][column] = "┼" if {old, character} == {"─", "│"} else character

    for wire in geometry.wires:
        for start, end in zip(wire.points, wire.points[1:], strict=False):
            clipped = _clip_segment(start, end, view)
            if clipped is not None:
                stroke(*clipped)

    visible_labels = [label for label in geometry.labels if visible((label.x_mm, label.y_mm))]
    for label in visible_labels:
        column, row = cell((label.x_mm, label.y_mm))
        if 0 < column < columns - 1 and 0 < row < rows - 1:
            grid[row][column] = "@"
            if zoom >= 2:
                for offset, character in enumerate(_truncate(label.name, 14), start=1):
                    target = column + offset
                    if target >= columns - 1 or grid[row][target] not in {" ", "─", "│", "·"}:
                        break
                    grid[row][target] = character

    allowed_refs = {
        str(component["reference"]) for component in snapshot["schematic"]["components"]
    }
    visible_symbols = sorted(
        [
            symbol
            for symbol in geometry.symbols
            if symbol.reference in allowed_refs and visible((symbol.x_mm, symbol.y_mm))
        ],
        key=lambda item: (item.y_mm, item.x_mm, item.reference),
    )
    claimed_component_cells: set[tuple[int, int]] = set()
    for symbol in visible_symbols:
        column, row = cell((symbol.x_mm, symbol.y_mm))
        label = f"[{_truncate(symbol.reference, 12)}]"
        start = max(1, min(columns - len(label) - 1, column - len(label) // 2))
        candidates = [row, row - 1, row + 1, row - 2, row + 2]
        target_row = next(
            (
                candidate
                for candidate in candidates
                if 0 < candidate < rows - 1
                and not any(
                    (target, candidate) in claimed_component_cells
                    for target in range(start, start + len(label))
                )
            ),
            None,
        )
        if target_row is not None:
            for offset, character in enumerate(label):
                target_column = start + offset
                grid[target_row][target_column] = character
                claimed_component_cells.add((target_column, target_row))

    for column in range(columns):
        grid[0][column] = "─"
        grid[-1][column] = "─"
    for row in range(rows):
        grid[row][0] = "│"
        grid[row][-1] = "│"
    grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1] = "┌", "┐", "└", "┘"

    page_name = str(selected_sheet.get("name") or selected_sheet.get("file"))
    header = (
        f"{snapshot['project']['name']}  SCHEMATIC SPATIAL MAP  sheet={page_name} zoom={zoom}  "
        f"view=({vx1:.1f},{vy1:.1f})-({vx2:.1f},{vy2:.1f})mm"
    )
    lines = [
        header,
        "zones=A1..D4 (stable over full page)  [REF]=component  @=named label",
        *("".join(row) for row in grid),
    ]

    nets_by_reference: dict[str, list[str]] = {}
    for net in snapshot["schematic"]["nets"]:
        for node in net["nodes"]:
            nets_by_reference.setdefault(str(node["reference"]), []).append(str(net["name"]))
    detail_limit = min(24, 6 + zoom * 6)
    lines.append(
        f"COMPONENTS visible={len(visible_symbols)}/{len(geometry.symbols)} details<={detail_limit}"
    )
    detail_symbols = sorted(
        visible_symbols,
        key=lambda item: (
            item.reference.casefold() != center_reference.casefold() if center_reference else False,
            item.y_mm,
            item.x_mm,
        ),
    )
    for symbol in detail_symbols[:detail_limit]:
        nearest_by_name: dict[str, float] = {}
        for label in geometry.labels:
            distance = math.hypot(symbol.x_mm - label.x_mm, symbol.y_mm - label.y_mm)
            nearest_by_name[label.name] = min(distance, nearest_by_name.get(label.name, math.inf))
        distances = sorted((distance, name) for name, distance in nearest_by_name.items())
        nearby = [f"{name}@{distance:.1f}mm" for distance, name in distances[: 2 + zoom]]
        net_names = sorted(set(nets_by_reference.get(symbol.reference, [])))
        lines.append(
            f"{_zone((symbol.x_mm, symbol.y_mm), full_bounds):2} "
            f"{symbol.reference:<8} ({symbol.x_mm:7.2f},{symbol.y_mm:7.2f}) "
            f"value={_truncate(symbol.value, 18)!r} "
            f"nets=[{_truncate(','.join(net_names), 36)}] "
            f"near=[{','.join(nearby)}]"
        )
    hidden = len(visible_symbols) - detail_limit
    if hidden > 0:
        lines.append(f"… {hidden} more visible component(s); raise zoom or use --center-ref REF")
    return "\n".join(lines)
