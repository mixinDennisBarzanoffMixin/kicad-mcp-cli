"""Read-only exact-pin railway rewiring plans for one schematic sheet.

The planner consumes a KiCad netlist snapshot but reads geometry directly from the
saved ``.kicad_sch`` page.  It resolves electrical pin tips from that page's cached
``lib_symbols`` definitions, classifies local nets separately from shared rails, and
emits explicit wire/label operations.  It never applies those operations.

Safety is intentionally one-sided: unresolved pins, ambiguous wire ownership,
symbol-body intersections, and crossings of another net all refuse a route.  This
makes the result useful as an LLM planning primitive without pretending that a
geometrically plausible line is electrically proven.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models.contract_verifier import extract_balanced_block
from .models.visual_qa import LabelItem, parse_labels
from .schematic_rewire_plan import (
    _point_on_segment,
    _same_point,
    _segment_crosses_box,
    _segment_intersection,
)
from .schematic_spatial import _root_blocks, parse_schematic_geometry, select_schematic_sheet
from .utils.geometry import Box

type JsonRecord = dict[str, Any]
type Point = tuple[float, float]
type Segment = tuple[Point, Point]
type _RouteState = tuple[Point, int]

_FLOAT = r"-?\d+(?:\.\d+)?"
_TOLERANCE = 1e-4
_GRID_MM = 2.54
_SCHEMATIC_GRID_MM = 1.27
_BODY_CLEARANCE_MM = 0.4
_LABEL_CLEARANCE_MM = 0.2
_LOCAL_ROUTE_MARGIN_MM = 4 * _GRID_MM
_BEND_PENALTY_MM = 0.25
_MAX_LOCAL_AXIS_COORDS = 96


@dataclass(frozen=True, slots=True)
class _PinDefinition:
    number: str
    name: str
    x: float
    y: float
    angle: int


@dataclass(frozen=True, slots=True)
class _Placed:
    reference: str
    lib_id: str
    x: float
    y: float
    angle: int
    unit: int
    block: str


@dataclass(frozen=True, slots=True)
class _ResolvedPin:
    reference: str
    pin: str
    name: str
    anchor: Point
    lib_id: str
    unit: int


@dataclass(frozen=True, slots=True)
class _WireOwnership:
    net: str | None
    ambiguous: bool


def _canonical_net_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _point_record(point: Point) -> list[float]:
    return [round(point[0], 4), round(point[1], 4)]


def _segment_key(segment: Segment) -> tuple[Point, Point]:
    first = (round(segment[0][0], 4), round(segment[0][1], 4))
    second = (round(segment[1][0], 4), round(segment[1][1], 4))
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _direct_children(block: str, keywords: set[str] | None = None) -> list[tuple[str, str]]:
    """Return direct child S-expressions without parsing nested KiCad data."""

    children: list[tuple[str, str]] = []
    depth = 0
    cursor = 0
    in_string = False
    escaped = False
    while cursor < len(block):
        character = block[cursor]
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
                match = re.match(r"\(([A-Za-z_][A-Za-z0-9_]*)", block[cursor:])
                if match and (keywords is None or match.group(1) in keywords):
                    child = extract_balanced_block(block, cursor)
                    if child:
                        children.append((match.group(1), child))
                        cursor += len(child)
                        continue
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        cursor += 1
    return children


def _library_entries(text: str) -> dict[str, str]:
    match = re.search(r"\(lib_symbols\b", text)
    if match is None:
        return {}
    block = extract_balanced_block(text, match.start())
    entries: dict[str, str] = {}
    for kind, child in _direct_children(block, {"symbol"}):
        del kind
        name = re.match(r'\(symbol\s+"([^"]+)"', child)
        if name and ":" in name.group(1):
            entries[name.group(1)] = child
    return entries


def _placed_symbols(text: str) -> dict[str, list[_Placed]]:
    placed: dict[str, list[_Placed]] = {}
    for kind, block in _root_blocks(text):
        if kind != "symbol":
            continue
        reference = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', block)
        lib_id = re.search(r'\(lib_id\s+"([^"]+)"', block)
        at = re.search(
            rf"^\(symbol\b.*?\(at\s+({_FLOAT})\s+({_FLOAT})(?:\s+({_FLOAT}))?\)", block, re.DOTALL
        )
        unit = re.search(r"\(unit\s+(\d+)\)", block)
        if reference is None or lib_id is None or at is None:
            continue
        item = _Placed(
            reference=reference.group(1),
            lib_id=lib_id.group(1),
            x=float(at.group(1)),
            y=float(at.group(2)),
            angle=int(round(float(at.group(3) or 0))),
            unit=int(unit.group(1)) if unit else 1,
            block=block,
        )
        placed.setdefault(item.reference.casefold(), []).append(item)
    for blocks in placed.values():
        blocks.sort(key=lambda item: (item.unit, item.lib_id, item.x, item.y))
    return placed


def _child_unit(name: str) -> int | None:
    parts = name.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        return None
    return int(parts[1])


def _pin_definition(block: str) -> _PinDefinition | None:
    at = re.search(rf"\(at\s+({_FLOAT})\s+({_FLOAT})(?:\s+({_FLOAT}))?\)", block)
    number = re.search(r'\(number\s+"([^"]*)"', block)
    name = re.search(r'\(name\s+"([^"]*)"', block)
    if at is None or number is None:
        return None
    return _PinDefinition(
        number=number.group(1),
        name=name.group(1) if name else "",
        x=float(at.group(1)),
        y=float(at.group(2)),
        angle=int(round(float(at.group(3) or 0))),
    )


def _unit_blocks(entry: str, unit: int) -> list[str]:
    blocks = [entry]
    for kind, child in _direct_children(entry, {"symbol"}):
        del kind
        name = re.match(r'\(symbol\s+"([^"]+)"', child)
        child_unit = _child_unit(name.group(1)) if name else None
        if child_unit in {0, unit}:
            blocks.append(child)
    return blocks


def _definitions_for_unit(entry: str, unit: int) -> dict[str, _PinDefinition | None]:
    found: dict[str, _PinDefinition | None] = {}
    for candidate in _unit_blocks(entry, unit):
        for kind, pin_block in _direct_children(candidate, {"pin"}):
            del kind
            definition = _pin_definition(pin_block)
            if definition is None:
                continue
            previous = found.get(definition.number)
            if previous is not None and previous != definition:
                found[definition.number] = None
            elif definition.number not in found:
                found[definition.number] = definition
    return found


def _transform_point(point: Point, placed: _Placed) -> Point:
    """Transform a library point to a schematic point like KiCad pin placement."""

    radians = math.radians(placed.angle)
    screen_x, screen_y = point[0], -point[1]
    rotated_x = screen_x * math.cos(radians) - screen_y * math.sin(radians)
    rotated_y = screen_x * math.sin(radians) + screen_y * math.cos(radians)
    return round(placed.x + rotated_x, 4), round(placed.y + rotated_y, 4)


def _resolve_pin(
    reference: str,
    pin: str,
    placed: dict[str, list[_Placed]],
    libraries: dict[str, str],
) -> tuple[_ResolvedPin | None, str | None]:
    candidates = placed.get(reference.casefold(), [])
    if not candidates:
        return None, "reference_not_placed_on_selected_sheet"
    resolutions: list[_ResolvedPin] = []
    reasons: list[str] = []
    for item in candidates:
        if re.search(r"\(mirror\s+[xy]\)", item.block):
            reasons.append("mirrored_symbol_transform_not_proven")
            continue
        entry = libraries.get(item.lib_id)
        if entry is None:
            reasons.append("cached_library_symbol_missing")
            continue
        definitions = _definitions_for_unit(entry, item.unit)
        if pin not in definitions:
            reasons.append("pin_number_absent_from_cached_symbol")
            continue
        definition = definitions[pin]
        if definition is None:
            reasons.append("duplicate_pin_number_has_conflicting_anchors")
            continue
        resolutions.append(
            _ResolvedPin(
                reference=item.reference,
                pin=pin,
                name=definition.name,
                anchor=_transform_point((definition.x, definition.y), item),
                lib_id=item.lib_id,
                unit=item.unit,
            )
        )
    unique = {(item.anchor, item.unit, item.name) for item in resolutions}
    if len(unique) == 1:
        return resolutions[0], None
    if len(unique) > 1:
        return None, "multi_unit_pin_anchor_is_ambiguous"
    return None, sorted(reasons)[0] if reasons else "pin_anchor_unresolved"


def _shape_points(block: str) -> list[Point]:
    points: list[Point] = []
    for kind, child in _direct_children(
        block, {"rectangle", "polyline", "circle", "arc", "bezier"}
    ):
        if kind == "circle":
            center = re.search(rf"\(center\s+({_FLOAT})\s+({_FLOAT})\)", child)
            radius = re.search(rf"\(radius\s+({_FLOAT})\)", child)
            if center and radius:
                cx, cy, value = (
                    float(center.group(1)),
                    float(center.group(2)),
                    float(radius.group(1)),
                )
                points.extend(((cx - value, cy - value), (cx + value, cy + value)))
            continue
        points.extend(
            (float(x), float(y))
            for x, y in re.findall(rf"\((?:start|end|mid|xy)\s+({_FLOAT})\s+({_FLOAT})\)", child)
        )
    return points


def _body_box(item: _Placed, entry: str | None) -> Box:
    if entry is None:
        return Box(item.x - 1.27, item.y - 1.27, item.x + 1.27, item.y + 1.27)
    local: list[Point] = []
    for block in _unit_blocks(entry, item.unit):
        local.extend(_shape_points(block))
    if not local:
        return Box(item.x - 1.27, item.y - 1.27, item.x + 1.27, item.y + 1.27)
    transformed = [_transform_point(point, item) for point in local]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return Box(min(xs), min(ys), max(xs), max(ys))


def _sheet_path(schematic_path: str | Path, selected_sheet: JsonRecord) -> Path:
    supplied = Path(schematic_path).expanduser().resolve()
    selected_file = Path(str(selected_sheet["file"]))
    if supplied.is_dir():
        result = selected_file if selected_file.is_absolute() else supplied / selected_file
    elif supplied.name == selected_file.name:
        result = supplied
    else:
        result = selected_file if selected_file.is_absolute() else supplied.parent / selected_file
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"selected schematic page is missing: {result}")
    return result


def _normalized_netlist(snapshot: JsonRecord) -> list[JsonRecord]:
    records = []
    for net in snapshot["schematic"]["nets"]:
        nodes = sorted(
            {
                (str(node.get("reference", "")), str(node.get("pin", "")))
                for node in net.get("nodes", [])
            }
        )
        records.append(
            {
                "name": str(net.get("name", "")),
                "nodes": [[reference, pin] for reference, pin in nodes],
            }
        )
    return sorted(records, key=lambda item: (item["name"], item["nodes"]))


def _merged_snapshot_nets(snapshot: JsonRecord) -> list[JsonRecord]:
    """Merge hierarchy-export duplicates before proposing physical edits.

    KiCad's hierarchy snapshot may repeat one logical net as multiple records.
    That duplication is valid evidence for the connectivity fingerprint, but it
    must never cause the railway planner to emit the same physical edit twice.
    """

    nodes_by_name: dict[str, dict[tuple[str, str], JsonRecord]] = {}
    for net in snapshot["schematic"]["nets"]:
        name = str(net.get("name", ""))
        merged = nodes_by_name.setdefault(name, {})
        for node in net.get("nodes", []):
            key = (str(node.get("reference", "")), str(node.get("pin", "")))
            merged.setdefault(key, dict(node))
    return [
        {
            "name": name,
            "nodes": [nodes[key] for key in sorted(nodes)],
        }
        for name, nodes in sorted(nodes_by_name.items())
    ]


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _wire_segments(text: str) -> list[Segment]:
    geometry = parse_schematic_geometry(text)
    return [
        (start, end)
        for wire in geometry.wires
        for start, end in zip(wire.points, wire.points[1:], strict=False)
    ]


def _wire_components(wires: list[Segment]) -> list[int]:
    parents = list(range(len(wires)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[max(root_first, root_second)] = min(root_first, root_second)

    for first_index, first in enumerate(wires):
        for second_index in range(first_index + 1, len(wires)):
            second = wires[second_index]
            # Endpoints and T-junctions are electrically connected.  A pure
            # interior crossing is deliberately not inferred without a junction.
            if any(_point_on_segment(point, second) for point in first) or any(
                _point_on_segment(point, first) for point in second
            ):
                union(first_index, second_index)
    return [find(index) for index in range(len(wires))]


def _wire_ownership(
    wires: list[Segment],
    pin_nets: list[tuple[Point, str]],
    labels: list[LabelItem],
    snapshot: JsonRecord,
) -> list[_WireOwnership]:
    if not wires:
        return []
    components = _wire_components(wires)
    names_by_canonical: dict[str, set[str]] = {}
    planning_nets = _merged_snapshot_nets(snapshot)
    for net in planning_nets:
        name = str(net["name"])
        names_by_canonical.setdefault(_canonical_net_name(name), set()).add(name)
    seeds: dict[int, set[str]] = {root: set() for root in set(components)}
    for index, segment in enumerate(wires):
        root = components[index]
        for point, net_name in pin_nets:
            if _point_on_segment(point, segment):
                seeds[root].add(net_name)
        for label in labels:
            if not _point_on_segment((label.x, label.y), segment):
                continue
            matches = names_by_canonical.get(label.text, set())
            if len(matches) == 1:
                seeds[root].update(matches)
    result = []
    for root in components:
        names = seeds[root]
        result.append(
            _WireOwnership(
                net=next(iter(names)) if len(names) == 1 else None,
                ambiguous=len(names) > 1,
            )
        )
    return result


def _deduplicate_segments(segments: Iterable[Segment]) -> tuple[Segment, ...]:
    unique: dict[tuple[Point, Point], Segment] = {}
    for start, end in segments:
        if _same_point(start, end):
            continue
        unique.setdefault(_segment_key((start, end)), (start, end))
    return tuple(unique[key] for key in sorted(unique))


def _point_inside_box(point: Point, box: Box) -> bool:
    return box.x_min < point[0] < box.x_max and box.y_min < point[1] < box.y_max


def _merge_safe_path_segments(
    points: list[Point],
    *,
    net_name: str,
    blocking_boxes: list[Box],
    existing_wires: list[Segment],
    ownership: list[_WireOwnership],
    accepted_other_nets: list[tuple[str, Segment]],
) -> tuple[Segment, ...]:
    """Merge collinear A* edges only while the merged edge passes every gate."""

    segments: list[Segment] = []
    for edge in zip(points, points[1:], strict=False):
        if not segments:
            segments.append(edge)
            continue
        previous = segments[-1]
        same_column = (
            abs(previous[0][0] - previous[1][0]) <= _TOLERANCE
            and abs(previous[1][0] - edge[1][0]) <= _TOLERANCE
        )
        same_row = (
            abs(previous[0][1] - previous[1][1]) <= _TOLERANCE
            and abs(previous[1][1] - edge[1][1]) <= _TOLERANCE
        )
        merged = (previous[0], edge[1])
        if (same_column or same_row) and _route_edge_is_safe(
            merged,
            net_name=net_name,
            blocking_boxes=blocking_boxes,
            existing_wires=existing_wires,
            ownership=ownership,
            accepted_other_nets=accepted_other_nets,
        ):
            segments[-1] = merged
        else:
            segments.append(edge)
    return _deduplicate_segments(segments)


def _local_route_window(anchors: list[Point]) -> Box:
    xs = [point[0] for point in anchors]
    ys = [point[1] for point in anchors]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    margin = max(_LOCAL_ROUTE_MARGIN_MM, min(2 * _LOCAL_ROUTE_MARGIN_MM, span / 2))
    return Box(
        math.floor((min(xs) - margin) / _SCHEMATIC_GRID_MM) * _SCHEMATIC_GRID_MM,
        math.floor((min(ys) - margin) / _SCHEMATIC_GRID_MM) * _SCHEMATIC_GRID_MM,
        math.ceil((max(xs) + margin) / _SCHEMATIC_GRID_MM) * _SCHEMATIC_GRID_MM,
        math.ceil((max(ys) + margin) / _SCHEMATIC_GRID_MM) * _SCHEMATIC_GRID_MM,
    )


def _box_touches(first: Box, second: Box) -> bool:
    return not (
        first.x_max < second.x_min
        or second.x_max < first.x_min
        or first.y_max < second.y_min
        or second.y_max < first.y_min
    )


def _segment_touches_box(segment: Segment, box: Box) -> bool:
    (x1, y1), (x2, y2) = segment
    return not (
        max(x1, x2) < box.x_min
        or min(x1, x2) > box.x_max
        or max(y1, y2) < box.y_min
        or min(y1, y2) > box.y_max
    )


def _bounded_axis_coordinates(
    values: set[float],
    *,
    required: set[float],
    anchor_low: float,
    anchor_high: float,
) -> list[float]:
    """Bound compressed-grid size while preserving anchors and window limits."""

    if len(values) <= _MAX_LOCAL_AXIS_COORDS:
        return sorted(values)
    remaining = _MAX_LOCAL_AXIS_COORDS - len(required)
    if remaining <= 0:
        return sorted(required)

    def corridor_distance(value: float) -> float:
        if value < anchor_low:
            return anchor_low - value
        if value > anchor_high:
            return value - anchor_high
        return 0.0

    optional = sorted(values - required, key=lambda value: (corridor_distance(value), value))
    return sorted({*required, *optional[:remaining]})


def _route_edge_is_safe(
    segment: Segment,
    *,
    net_name: str,
    blocking_boxes: list[Box],
    existing_wires: list[Segment],
    ownership: list[_WireOwnership],
    accepted_other_nets: list[tuple[str, Segment]],
) -> bool:
    if any(_segment_crosses_box(segment, box) for box in blocking_boxes):
        return False
    for wire_index, wire in enumerate(existing_wires):
        kind, _ = _segment_intersection(segment, wire)
        if kind == "none":
            continue
        owner = ownership[wire_index]
        if owner.net != net_name:
            return False
        if kind == "overlap" and not (
            _point_on_segment(segment[0], wire) and _point_on_segment(segment[1], wire)
        ):
            # A partial collinear overlap is electrically plausible but the
            # existing evaluator intentionally refuses it.  Do not let the
            # search manufacture a candidate which cannot pass the same gate.
            return False
    return not any(
        _segment_intersection(segment, other_segment)[0] != "none"
        for _, other_segment in accepted_other_nets
    )


def _bounded_orthogonal_path(
    anchors: list[Point],
    *,
    net_name: str,
    blocking_boxes: list[Box],
    existing_wires: list[Segment],
    ownership: list[_WireOwnership],
    accepted_other_nets: list[tuple[str, Segment]],
) -> tuple[Segment, ...] | None:
    """Find one short local two-pin route on a coordinate-compressed A* grid."""

    if len(anchors) != 2:
        return None
    start, goal = anchors
    window = _local_route_window(anchors)
    local_boxes = [box for box in blocking_boxes if _box_touches(box, window)]
    local_wires = [wire for wire in existing_wires if _segment_touches_box(wire, window)]
    x_values = {start[0], goal[0], window.x_min, window.x_max}
    y_values = {start[1], goal[1], window.y_min, window.y_max}

    def add_lane(value: float, values: set[float], low: float, high: float) -> None:
        for candidate in (value - _GRID_MM, value, value + _GRID_MM):
            snapped = round(round(candidate / _SCHEMATIC_GRID_MM) * _SCHEMATIC_GRID_MM, 4)
            if low - _TOLERANCE <= snapped <= high + _TOLERANCE:
                values.add(snapped)

    for box in local_boxes:
        add_lane(box.x_min, x_values, window.x_min, window.x_max)
        add_lane(box.x_max, x_values, window.x_min, window.x_max)
        add_lane(box.y_min, y_values, window.y_min, window.y_max)
        add_lane(box.y_max, y_values, window.y_min, window.y_max)
    for wire in local_wires:
        for point in wire:
            add_lane(point[0], x_values, window.x_min, window.x_max)
            add_lane(point[1], y_values, window.y_min, window.y_max)

    required_x = {start[0], goal[0], window.x_min, window.x_max}
    required_y = {start[1], goal[1], window.y_min, window.y_max}
    xs = _bounded_axis_coordinates(
        x_values,
        required=required_x,
        anchor_low=min(start[0], goal[0]),
        anchor_high=max(start[0], goal[0]),
    )
    ys = _bounded_axis_coordinates(
        y_values,
        required=required_y,
        anchor_low=min(start[1], goal[1]),
        anchor_high=max(start[1], goal[1]),
    )
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    points = {
        (x, y)
        for x in xs
        for y in ys
        if not any(_point_inside_box((x, y), box) for box in local_boxes)
    }
    points.update(anchors)

    # Direction is part of the state so equal-length alternatives prefer fewer
    # bends while remaining stable under repeated runs.
    start_state: _RouteState = (start, 0)
    queue: list[tuple[float, int, float, float, float, int, _RouteState]] = [
        (math.dist(start, goal), 0, 0.0, start[1], start[0], 0, start_state)
    ]
    best: dict[_RouteState, tuple[float, int]] = {start_state: (0.0, 0)}
    previous: dict[_RouteState, _RouteState] = {}
    final: _RouteState | None = None
    while queue:
        _, bends, cost, _, _, _, state = heapq.heappop(queue)
        point, direction = state
        if best.get(state) != (cost, bends):
            continue
        if _same_point(point, goal):
            final = state
            break
        xi, yi = x_index[point[0]], y_index[point[1]]
        neighbours: list[tuple[Point, int]] = []
        if xi:
            neighbours.append(((xs[xi - 1], point[1]), 1))
        if xi + 1 < len(xs):
            neighbours.append(((xs[xi + 1], point[1]), 1))
        if yi:
            neighbours.append(((point[0], ys[yi - 1]), 2))
        if yi + 1 < len(ys):
            neighbours.append(((point[0], ys[yi + 1]), 2))
        for neighbour, next_direction in sorted(
            neighbours, key=lambda item: (item[0][1], item[0][0], item[1])
        ):
            if neighbour not in points:
                continue
            edge = (point, neighbour)
            if not _route_edge_is_safe(
                edge,
                net_name=net_name,
                blocking_boxes=local_boxes,
                existing_wires=existing_wires,
                ownership=ownership,
                accepted_other_nets=accepted_other_nets,
            ):
                continue
            bend = int(direction not in {0, next_direction})
            next_bends = bends + bend
            next_cost = round(
                cost + math.dist(point, neighbour) + bend * _BEND_PENALTY_MM,
                6,
            )
            next_state = (neighbour, next_direction)
            if (next_cost, next_bends) >= best.get(next_state, (math.inf, 2**31)):
                continue
            best[next_state] = (next_cost, next_bends)
            previous[next_state] = state
            heuristic = abs(neighbour[0] - goal[0]) + abs(neighbour[1] - goal[1])
            heapq.heappush(
                queue,
                (
                    next_cost + heuristic,
                    next_bends,
                    next_cost,
                    neighbour[1],
                    neighbour[0],
                    next_direction,
                    next_state,
                ),
            )
    if final is None:
        return None
    path = [final[0]]
    while final != start_state:
        final = previous[final]
        path.append(final[0])
    path.reverse()
    return _merge_safe_path_segments(
        path,
        net_name=net_name,
        blocking_boxes=local_boxes,
        existing_wires=existing_wires,
        ownership=ownership,
        accepted_other_nets=accepted_other_nets,
    )


def _route_candidates(
    anchors: list[Point],
    obstacles: list[Box],
    *,
    net_name: str,
    existing_wires: list[Segment],
    ownership: list[_WireOwnership],
    label_obstacles: list[tuple[str, str, Box]],
    removable_label_ids: set[str],
    accepted_other_nets: list[tuple[str, Segment]],
) -> list[tuple[Segment, ...]]:
    if len(anchors) < 2:
        return []
    x_values = {point[0] for point in anchors}
    y_values = {point[1] for point in anchors}
    if obstacles:
        x_values.update(
            {
                min(box.x_min for box in obstacles) - _GRID_MM,
                max(box.x_max for box in obstacles) + _GRID_MM,
            }
        )
        y_values.update(
            {
                min(box.y_min for box in obstacles) - _GRID_MM,
                max(box.y_max for box in obstacles) + _GRID_MM,
            }
        )
    candidates: list[tuple[Segment, ...]] = []
    for rail_y in sorted(y_values):
        drops = [((x, y), (x, rail_y)) for x, y in anchors]
        candidates.append(
            _deduplicate_segments(
                [
                    *drops,
                    (
                        (min(point[0] for point in anchors), rail_y),
                        (max(point[0] for point in anchors), rail_y),
                    ),
                ]
            )
        )
    for rail_x in sorted(x_values):
        drops = [((x, y), (rail_x, y)) for x, y in anchors]
        candidates.append(
            _deduplicate_segments(
                [
                    *drops,
                    (
                        (rail_x, min(point[1] for point in anchors)),
                        (rail_x, max(point[1] for point in anchors)),
                    ),
                ]
            )
        )
    blocking_boxes = [
        *obstacles,
        *[box for label_id, _, box in label_obstacles if label_id not in removable_label_ids],
    ]
    local_path = _bounded_orthogonal_path(
        anchors,
        net_name=net_name,
        blocking_boxes=blocking_boxes,
        existing_wires=existing_wires,
        ownership=ownership,
        accepted_other_nets=accepted_other_nets,
    )
    if local_path:
        candidates.append(local_path)
    unique = {
        _json_sha256([[_point_record(a), _point_record(b)] for a, b in item]): item
        for item in candidates
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            round(sum(math.dist(start, end) for start, end in item), 4),
            len(item),
            tuple(_segment_key(segment) for segment in item),
        ),
    )


def _evaluate_candidate(
    segments: tuple[Segment, ...],
    *,
    net_name: str,
    existing_wires: list[Segment],
    ownership: list[_WireOwnership],
    symbol_obstacles: list[tuple[str, Box]],
    label_obstacles: list[tuple[str, str, Box]],
    removable_label_ids: set[str],
    accepted_other_nets: list[tuple[str, Segment]],
) -> tuple[list[JsonRecord], set[int], set[Point]]:
    refusals: list[JsonRecord] = []
    covered: set[int] = set()
    junctions: set[Point] = set()
    for segment_index, segment in enumerate(segments):
        for reference, box in symbol_obstacles:
            if _segment_crosses_box(segment, box):
                refusals.append(
                    {
                        "code": "symbol_body_intersection",
                        "segment": segment_index,
                        "reference": reference,
                    }
                )
        for label_id, label_name, box in label_obstacles:
            if label_id not in removable_label_ids and _segment_crosses_box(segment, box):
                refusals.append(
                    {
                        "code": "retained_label_intersection",
                        "segment": segment_index,
                        "label_id": label_id,
                        "label": label_name,
                    }
                )
        for wire_index, wire in enumerate(existing_wires):
            kind, point = _segment_intersection(segment, wire)
            if kind == "none":
                continue
            owner = ownership[wire_index]
            if kind == "overlap":
                fully_covered = _point_on_segment(segment[0], wire) and _point_on_segment(
                    segment[1], wire
                )
                if owner.net == net_name and fully_covered:
                    covered.add(segment_index)
                else:
                    refusals.append(
                        {
                            "code": "existing_wire_overlap_unproven",
                            "segment": segment_index,
                            "existing_wire": wire_index,
                            "owner": owner.net,
                            "ambiguous": owner.ambiguous,
                        }
                    )
            elif point is not None:
                if owner.net == net_name:
                    junctions.add(point)
                else:
                    refusals.append(
                        {
                            "code": "unrelated_net_crossing"
                            if owner.net
                            else "unknown_wire_crossing",
                            "segment": segment_index,
                            "existing_wire": wire_index,
                            "at_mm": _point_record(point),
                            "owner": owner.net,
                            "ambiguous": owner.ambiguous,
                        }
                    )
        for other_net, other_segment in accepted_other_nets:
            kind, point = _segment_intersection(segment, other_segment)
            if kind != "none":
                refusal: JsonRecord = {
                    "code": "planned_unrelated_net_crossing",
                    "segment": segment_index,
                    "other_net": other_net,
                }
                if point is not None:
                    refusal["at_mm"] = _point_record(point)
                refusals.append(refusal)
    deduplicated = {json.dumps(item, sort_keys=True): item for item in refusals}
    return list(deduplicated.values()), covered, junctions


def _label_id(label: LabelItem, ordinal: int) -> str:
    return f"{label.kind}:{label.text}@{label.x:.4f},{label.y:.4f}#{ordinal}"


def _safe_new_label_anchor(
    name: str,
    segments: tuple[Segment, ...],
    *,
    symbol_obstacles: list[tuple[str, Box]],
    other_label_obstacles: list[tuple[str, str, Box]],
) -> Point | None:
    """Choose a deterministic route point where a replacement label fits."""

    points = {
        point
        for start, end in segments
        for point in (start, end, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
    }
    for point in sorted(points, key=lambda item: (item[1], item[0])):
        label_box = LabelItem(name, point[0], point[1], "local").box().expanded(_LABEL_CLEARANCE_MM)
        if any(label_box.overlaps(box) for _, box in symbol_obstacles):
            continue
        if any(label_box.overlaps(box) for _, _, box in other_label_obstacles):
            continue
        return round(point[0], 4), round(point[1], 4)
    return None


def _net_name_label_choice(
    net_name: str,
    segments: tuple[Segment, ...],
    matching_labels: list[tuple[str, LabelItem]],
    *,
    symbol_obstacles: list[tuple[str, Box]],
    label_obstacles: list[tuple[str, str, Box]],
) -> tuple[tuple[str, LabelItem] | None, Point | None]:
    """Return one attached existing local label, or a safe replacement anchor."""

    local_labels = sorted(
        ((label_id, label) for label_id, label in matching_labels if label.kind == "local"),
        key=lambda item: (item[1].y, item[1].x, item[0]),
    )
    attached = [
        item
        for item in local_labels
        if any(_point_on_segment((item[1].x, item[1].y), segment) for segment in segments)
    ]
    if attached:
        return attached[0], None
    local_ids = {label_id for label_id, _ in local_labels}
    replacement = _safe_new_label_anchor(
        _canonical_net_name(net_name),
        segments,
        symbol_obstacles=symbol_obstacles,
        other_label_obstacles=[
            obstacle for obstacle in label_obstacles if obstacle[0] not in local_ids
        ],
    )
    return None, replacement


def _classify_net(
    nodes: list[JsonRecord],
    cluster: set[str],
    selected_sheet_name: str,
    component_sheets: dict[str, str],
) -> str:
    node_refs = {str(node.get("reference", "")).casefold() for node in nodes}
    node_sheets = {component_sheets.get(reference, "") for reference in node_refs}
    if any(sheet and sheet != selected_sheet_name for sheet in node_sheets):
        return "cross_sheet_rail"
    if node_refs - cluster:
        return "shared_sheet_rail"
    if len(nodes) >= 2:
        return "local_intra_cluster"
    return "cluster_terminal"


def plan_railway_rewire(
    schematic_path: str | Path,
    snapshot: JsonRecord,
    *,
    sheet: str = "",
    cluster_refs: Iterable[str] | None = None,
) -> JsonRecord:
    """Return an exact-pin, deterministic, non-applying rewiring plan.

    ``schematic_path`` may identify the project directory, root schematic, or
    selected child page.  ``snapshot`` is the ordinary :func:`project_snapshot`
    result.  If ``cluster_refs`` is omitted, every component on the selected
    sheet is in scope.  Existing symbol positions are evidence and are never
    changed.
    """

    requested = sorted(
        {str(reference).strip() for reference in cluster_refs or () if str(reference).strip()}
    )
    center = requested[0] if requested else ""
    selected = select_schematic_sheet(snapshot, sheet, center)
    source = _sheet_path(schematic_path, selected)
    before_bytes = source.read_bytes()
    text = before_bytes.decode("utf-8", errors="ignore")
    libraries = _library_entries(text)
    placed = _placed_symbols(text)

    component_sheets = {
        str(item.get("reference", "")).casefold(): str(item.get("sheet", ""))
        for item in snapshot["schematic"]["components"]
    }
    selected_name = str(selected["name"])
    selected_refs = {
        reference
        for reference, component_sheet in component_sheets.items()
        if component_sheet == selected_name
    }
    cluster = {reference.casefold() for reference in requested} if requested else selected_refs
    missing = sorted(reference for reference in cluster if reference not in placed)
    if missing:
        raise ValueError(
            f"cluster references are not placed on selected page: {', '.join(missing)}"
        )

    planning_nets = _merged_snapshot_nets(snapshot)
    resolved_by_node: dict[tuple[str, str], _ResolvedPin] = {}
    unresolved: list[JsonRecord] = []
    for net in planning_nets:
        for node in net.get("nodes", []):
            reference = str(node.get("reference", ""))
            pin = str(node.get("pin", ""))
            if reference.casefold() not in selected_refs:
                continue
            key = (reference.casefold(), pin)
            if key in resolved_by_node or any(
                item["reference"].casefold() == key[0] and item["pin"] == pin for item in unresolved
            ):
                continue
            resolved, reason = _resolve_pin(reference, pin, placed, libraries)
            if resolved is None:
                unresolved.append(
                    {
                        "reference": reference,
                        "pin": pin,
                        "reason": reason or "pin_anchor_unresolved",
                    }
                )
            else:
                resolved_by_node[key] = resolved

    all_pin_nets: list[tuple[Point, str]] = []
    for net in planning_nets:
        for node in net.get("nodes", []):
            resolved = resolved_by_node.get(
                (str(node.get("reference", "")).casefold(), str(node.get("pin", "")))
            )
            if resolved is not None:
                all_pin_nets.append((resolved.anchor, str(net["name"])))

    labels = parse_labels(text)
    wires = _wire_segments(text)
    ownership = _wire_ownership(wires, all_pin_nets, labels, snapshot)
    placed_items = [item for blocks in placed.values() for item in blocks]
    symbol_obstacles = [
        (item.reference, _body_box(item, libraries.get(item.lib_id)).expanded(_BODY_CLEARANCE_MM))
        for item in placed_items
    ]
    label_records = [
        (_label_id(label, index), label) for index, label in enumerate(labels, start=1)
    ]
    label_obstacles = [
        (label_id, label.text, label.box().expanded(_LABEL_CLEARANCE_MM))
        for label_id, label in label_records
    ]

    net_plans: list[JsonRecord] = []
    operations: list[JsonRecord] = []
    accepted_segments: list[tuple[str, Segment]] = []
    refused_count = 0
    routed_count = 0
    for net in planning_nets:
        nodes = [dict(node) for node in net.get("nodes", [])]
        cluster_nodes = [
            node for node in nodes if str(node.get("reference", "")).casefold() in cluster
        ]
        if not cluster_nodes:
            continue
        net_name = str(net["name"])
        classification = _classify_net(nodes, cluster, selected_name, component_sheets)
        exact_pins: list[_ResolvedPin] = []
        net_unresolved: list[JsonRecord] = []
        for node in cluster_nodes:
            key = (str(node.get("reference", "")).casefold(), str(node.get("pin", "")))
            resolved = resolved_by_node.get(key)
            if resolved is None:
                matching = next(
                    (
                        item
                        for item in unresolved
                        if item["reference"].casefold() == key[0] and item["pin"] == key[1]
                    ),
                    {"reference": node.get("reference", ""), "pin": key[1], "reason": "unresolved"},
                )
                net_unresolved.append(matching)
            else:
                exact_pins.append(resolved)
        matching_labels = [
            (label_id, label)
            for label_id, label in label_records
            if label.text == _canonical_net_name(net_name)
        ]
        plan: JsonRecord = {
            "net": net_name,
            "classification": classification,
            "cluster_nodes": [
                {
                    "reference": pin.reference,
                    "pin": pin.pin,
                    "pin_name": pin.name,
                    "anchor_mm": _point_record(pin.anchor),
                    "lib_id": pin.lib_id,
                    "unit": pin.unit,
                }
                for pin in sorted(
                    exact_pins, key=lambda item: (item.reference, item.pin, item.unit)
                )
            ],
            "labels": [],
            "candidate_routes": [],
        }
        if net_unresolved:
            plan["status"] = "refused"
            plan["refusals"] = [{"code": "unresolved_exact_pin", **item} for item in net_unresolved]
            refused_count += 1
            net_plans.append(plan)
            continue

        if classification != "local_intra_cluster":
            for label_id, label in matching_labels:
                operation = {
                    "op": "retain_label",
                    "id": label_id,
                    "net": net_name,
                    "kind": label.kind,
                    "anchor_mm": _point_record((label.x, label.y)),
                    "reason": "shared_or_cross_sheet_rail_contract",
                }
                plan["labels"].append(operation)
                operations.append(operation)
            plan["status"] = "retained"
            if not matching_labels:
                plan["notes"] = ["no matching label was parsed on this page; no rail edit proposed"]
            net_plans.append(plan)
            continue

        if len(exact_pins) < 2:
            plan["status"] = "refused"
            plan["refusals"] = [
                {"code": "insufficient_local_pin_anchors", "resolved_count": len(exact_pins)}
            ]
            refused_count += 1
            net_plans.append(plan)
            continue

        removable = {label_id for label_id, label in matching_labels if label.kind == "local"}
        for label_id, label in matching_labels:
            if label_id not in removable:
                operation = {
                    "op": "retain_label",
                    "id": label_id,
                    "net": net_name,
                    "kind": label.kind,
                    "anchor_mm": _point_record((label.x, label.y)),
                    "reason": "nonlocal_label_semantics_are_conserved",
                }
                plan["labels"].append(operation)
                operations.append(operation)

        anchors = sorted({pin.anchor for pin in exact_pins})
        chosen: tuple[Segment, ...] | None = None
        chosen_covered: set[int] = set()
        chosen_junctions: set[Point] = set()
        chosen_retained_label: tuple[str, LabelItem] | None = None
        chosen_added_label_anchor: Point | None = None
        for candidate in _route_candidates(
            anchors,
            [box for _, box in symbol_obstacles],
            net_name=net_name,
            existing_wires=wires,
            ownership=ownership,
            label_obstacles=label_obstacles,
            removable_label_ids=removable,
            accepted_other_nets=accepted_segments,
        ):
            refusals, covered, junctions = _evaluate_candidate(
                candidate,
                net_name=net_name,
                existing_wires=wires,
                ownership=ownership,
                symbol_obstacles=symbol_obstacles,
                label_obstacles=label_obstacles,
                removable_label_ids=removable,
                accepted_other_nets=accepted_segments,
            )
            retained_label, added_label_anchor = _net_name_label_choice(
                net_name,
                candidate,
                matching_labels,
                symbol_obstacles=symbol_obstacles,
                label_obstacles=label_obstacles,
            )
            if retained_label is None and added_label_anchor is None:
                refusals.append(
                    {
                        "code": "net_name_label_unplaceable",
                        "reason": (
                            "no existing same-name local label is attached to the route and "
                            "no collision-free replacement anchor was found"
                        ),
                    }
                )
            record = {
                "segments": [
                    {"start_mm": _point_record(start), "end_mm": _point_record(end)}
                    for start, end in candidate
                ],
                "length_mm": round(sum(math.dist(start, end) for start, end in candidate), 4),
                "safe": not refusals,
                "refusals": refusals,
            }
            plan["candidate_routes"].append(record)
            if chosen is None and not refusals:
                chosen = candidate
                chosen_covered = covered
                chosen_junctions = junctions
                chosen_retained_label = retained_label
                chosen_added_label_anchor = added_label_anchor
        if chosen is None:
            plan["status"] = "refused"
            plan["refusals"] = [
                reason for candidate in plan["candidate_routes"] for reason in candidate["refusals"]
            ]
            refused_count += 1
            net_plans.append(plan)
            continue

        selected_operations: list[JsonRecord] = []
        for index, (start, end) in enumerate(chosen):
            if index in chosen_covered:
                continue
            operation = {
                "op": "add_wire",
                "net": net_name,
                "start_mm": _point_record(start),
                "end_mm": _point_record(end),
            }
            selected_operations.append(operation)
            operations.append(operation)
            accepted_segments.append((net_name, (start, end)))
        for point in sorted(chosen_junctions):
            operation = {"op": "ensure_junction", "net": net_name, "at_mm": _point_record(point)}
            selected_operations.append(operation)
            operations.append(operation)
        retained_local_id = chosen_retained_label[0] if chosen_retained_label else None
        if chosen_retained_label is not None:
            label_id, label = chosen_retained_label
            operation = {
                "op": "retain_label",
                "id": label_id,
                "net": net_name,
                "kind": label.kind,
                "anchor_mm": _point_record((label.x, label.y)),
                "reason": "preserve_the_snapshot_net_name_on_the_new_continuous_route",
            }
            plan["labels"].append(operation)
            selected_operations.append(operation)
            operations.append(operation)
        elif chosen_added_label_anchor is not None:
            operation = {
                "op": "add_label",
                "net": net_name,
                "name": _canonical_net_name(net_name),
                "kind": "local",
                "anchor_mm": _point_record(chosen_added_label_anchor),
                "reason": "preserve_the_snapshot_net_name_on_the_new_continuous_route",
            }
            plan["labels"].append(operation)
            selected_operations.append(operation)
            operations.append(operation)
        for label_id, label in matching_labels:
            if label_id not in removable or label_id == retained_local_id:
                continue
            operation = {
                "op": "remove_label",
                "id": label_id,
                "net": net_name,
                "kind": label.kind,
                "anchor_mm": _point_record((label.x, label.y)),
                "reason": "exact pins are directly connected by selected railway route",
            }
            plan["labels"].append(operation)
            selected_operations.append(operation)
            operations.append(operation)
        plan["status"] = "planned"
        plan["selected_operations"] = selected_operations
        routed_count += 1
        net_plans.append(plan)

    after_bytes = source.read_bytes()
    if after_bytes != before_bytes:
        raise RuntimeError("read-only railway planner observed source mutation during planning")
    normalized = _normalized_netlist(snapshot)
    netlist_hash = _json_sha256(normalized)
    positions = [
        {
            "reference": item.reference,
            "lib_id": item.lib_id,
            "unit": item.unit,
            "at_mm": [item.x, item.y],
            "rotation_deg": item.angle,
        }
        for item in sorted(placed_items, key=lambda item: (item.reference, item.unit, item.lib_id))
        if item.reference.casefold() in cluster
    ]
    if routed_count and not refused_count:
        status = "planned"
    elif routed_count:
        status = "review"
    elif refused_count:
        status = "blocked"
    else:
        status = "noop"
    return {
        "schema_version": "1.0",
        "status": status,
        "read_only": True,
        "apply_supported": False,
        "source": {
            "path": str(source),
            "sha256_before": hashlib.sha256(before_bytes).hexdigest(),
            "sha256_after": hashlib.sha256(after_bytes).hexdigest(),
            "unchanged": before_bytes == after_bytes,
        },
        "sheet": {"name": selected_name, "file": str(selected["file"])},
        "cluster_refs": sorted(
            item.reference
            for blocks in placed.values()
            for item in blocks
            if item.reference.casefold() in cluster
        ),
        "preserved_symbol_positions": positions,
        "pin_resolution": {
            "resolved_count": len(resolved_by_node),
            "unresolved": sorted(unresolved, key=lambda item: (item["reference"], item["pin"])),
        },
        "nets": net_plans,
        "operations": operations,
        "connectivity_fingerprint_expectation": {
            "algorithm": "sha256(canonical-net-name-and-ref-pin-pairs-v1)",
            "before": netlist_hash,
            "expected_after": netlist_hash,
            "must_match": True,
            "canonical_netlist": normalized,
        },
        "position_fingerprint_expectation": {
            "algorithm": "sha256(cluster-symbol-placement-v1)",
            "before": _json_sha256(positions),
            "expected_after": _json_sha256(positions),
            "must_match": True,
        },
        "invariants": [
            "planning_only_no_cad_mutation",
            "all_routed_cluster_nodes_use_exact_cached_library_pin_tips",
            "symbol_positions_and_rotations_are_preserved",
            "shared_and_cross_sheet_labels_are_retained",
            "every_new_local_railway_retains_or_adds_one_attached_same_name_label",
            "routes_crossing_symbol_bodies_or_unrelated_or_unknown nets are refused",
            "future_apply_must_reexport_and_match_the_connectivity_fingerprint",
        ],
    }


def format_railway_rewire_plan(plan: JsonRecord) -> str:
    """Render a compact deterministic terminal view of a railway plan."""

    lines = [
        f"RAILWAY REWIRE  status={plan['status']} read_only=true apply=false",
        f"sheet={plan['sheet']['name']} cluster={','.join(plan['cluster_refs']) or '-'}",
    ]
    for net in plan["nets"]:
        lines.append(
            f"{net['status'].upper():8} {net['classification']:20} {net['net']} "
            f"pins={len(net['cluster_nodes'])}"
        )
        for operation in net.get("selected_operations", []):
            if operation["op"] == "add_wire":
                lines.append(f"  WIRE {operation['start_mm']} -> {operation['end_mm']}")
            elif operation["op"] == "remove_label":
                lines.append(f"  REMOVE LABEL {operation['id']}")
        if net["status"] == "refused":
            codes = sorted({item["code"] for item in net.get("refusals", [])})
            lines.append(f"  REFUSE {','.join(codes)}")
    fingerprint = plan["connectivity_fingerprint_expectation"]["before"]
    lines.append(f"EXPECT connectivity sha256={fingerprint}")
    lines.append("NO APPLY: re-exported netlist and symbol-placement hashes must remain identical")
    return "\n".join(lines)
