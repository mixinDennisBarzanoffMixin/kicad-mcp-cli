"""Headless project inspection, ASCII maps, and conservative route planning.

The normal MCP backend remains the authority for KiCad mutations.  This module
adds a fast, file-oriented planning surface that works with KiCad closed and
emits ordinary Python/JSON values for shell pipelines.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .tools.board_file import (
    FLOAT_PATTERN,
    STRING_PATTERN,
    _edge_cuts_bounds,
    _iter_blocks,
    _normalize_board_content,
    _parse_board_footprint_blocks,
)

type JsonRecord = dict[str, Any]


def _project_files(project_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(project_dir).expanduser().resolve()
    projects = sorted(root.glob("*.kicad_pro"))
    if not projects:
        raise FileNotFoundError(f"no .kicad_pro file found in {root}")
    if len(projects) > 1:
        names = ", ".join(path.name for path in projects)
        raise ValueError(f"multiple KiCad projects found in {root}: {names}")
    project = projects[0]
    schematic = project.with_suffix(".kicad_sch")
    board = project.with_suffix(".kicad_pcb")
    for path in (schematic, board):
        if not path.is_file():
            raise FileNotFoundError(f"project companion file is missing: {path}")
    return project, schematic, board


def _kicad_cli() -> str:
    configured = os.environ.get("KICAD_CLI", "").strip()
    candidates = [
        configured,
        shutil.which("kicad-cli") or "",
        "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("kicad-cli was not found; set KICAD_CLI")


def _export_netlist(schematic: Path) -> ET.Element:
    with tempfile.TemporaryDirectory(prefix="kicadq-netlist-") as temporary:
        output = Path(temporary) / "project.xml"
        process = subprocess.run(
            [
                _kicad_cli(),
                "sch",
                "export",
                "netlist",
                "--format",
                "kicadxml",
                str(schematic),
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0 or not output.is_file():
            diagnostic = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"KiCad netlist export failed: {diagnostic}")
        # The XML is generated locally by the selected KiCad executable, not supplied by a caller.
        return ET.parse(output).getroot()  # noqa: S314


def _text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    return (child.text or "").strip() if child is not None else ""


def _schematic_snapshot(
    root: ET.Element,
) -> tuple[list[JsonRecord], list[JsonRecord], list[JsonRecord]]:
    sheets = [
        {
            "number": int(sheet.get("number", "0")),
            "name": sheet.get("name", ""),
            "file": _text(sheet.find("title_block") or ET.Element("empty"), "source"),
        }
        for sheet in root.findall("./design/sheet")
    ]
    components: list[JsonRecord] = []
    for component in root.findall("./components/comp"):
        sheetpath = component.find("sheetpath")
        components.append(
            {
                "reference": component.get("ref", ""),
                "value": _text(component, "value"),
                "footprint": _text(component, "footprint"),
                "datasheet": _text(component, "datasheet"),
                "sheet": sheetpath.get("names", "") if sheetpath is not None else "",
            }
        )
    nets: list[JsonRecord] = []
    for net in root.findall("./nets/net"):
        nodes = [
            {
                "reference": node.get("ref", ""),
                "pin": node.get("pin", ""),
                "function": node.get("pinfunction", ""),
                "type": node.get("pintype", ""),
            }
            for node in net.findall("node")
        ]
        name = net.get("name", "")
        nets.append(
            {
                "code": int(net.get("code", "0")),
                "name": name,
                "class": net.get("class", ""),
                "nodes": nodes,
                "unconnected": name.startswith("unconnected-("),
            }
        )
    return sheets, components, nets


def _track_records(content: str) -> list[JsonRecord]:
    tracks: list[JsonRecord] = []
    for segment in _iter_blocks(content, "segment"):
        start = re.search(rf"\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", segment)
        end = re.search(rf"\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", segment)
        width = re.search(rf"\(width\s+({FLOAT_PATTERN})\)", segment)
        layer = re.search(r'\(layer\s+"([^"]+)"\)', segment)
        net = re.search(r"\(net\s+(\d+)\)", segment)
        if not (start and end and width and layer and net):
            continue
        x1, y1, x2, y2 = map(float, (*start.groups(), *end.groups()))
        tracks.append(
            {
                "start": [x1, y1],
                "end": [x2, y2],
                "width_mm": float(width.group(1)),
                "layer": layer.group(1),
                "net_code": int(net.group(1)),
                "length_mm": round(math.hypot(x2 - x1, y2 - y1), 4),
            }
        )
    return tracks


def _via_records(content: str) -> list[JsonRecord]:
    vias: list[JsonRecord] = []
    for via in _iter_blocks(content, "via"):
        at = re.search(rf"\(at\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", via)
        size = re.search(rf"\(size\s+({FLOAT_PATTERN})\)", via)
        drill = re.search(rf"\(drill\s+({FLOAT_PATTERN})\)", via)
        net = re.search(r"\(net\s+(\d+)\)", via)
        if at and size and drill and net:
            vias.append(
                {
                    "at": [float(at.group(1)), float(at.group(2))],
                    "diameter_mm": float(size.group(1)),
                    "drill_mm": float(drill.group(1)),
                    "net_code": int(net.group(1)),
                }
            )
    return vias


def _pad_positions(footprint: JsonRecord) -> list[JsonRecord]:
    root_x = float(footprint.get("x_mm") or 0.0)
    root_y = float(footprint.get("y_mm") or 0.0)
    root_rotation = math.radians(float(footprint.get("rotation") or 0.0))
    back = str(footprint.get("layer_name", "F.Cu")) == "B.Cu"
    pads: list[JsonRecord] = []
    for pad in _iter_blocks(str(footprint["block"]), "pad"):
        number = re.match(rf"\(pad\s+{STRING_PATTERN}", pad.lstrip())
        at = re.search(
            rf"\(at\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})(?:\s+{FLOAT_PATTERN})?\)", pad
        )
        size = re.search(rf"\(size\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", pad)
        net = re.search(rf"\(net(?:\s+(\d+))?\s+{STRING_PATTERN}\)", pad)
        if not (number and at and size):
            continue
        local_x = float(at.group(1)) * (-1 if back else 1)
        local_y = float(at.group(2))
        global_x = root_x + local_x * math.cos(root_rotation) - local_y * math.sin(root_rotation)
        global_y = root_y + local_x * math.sin(root_rotation) + local_y * math.cos(root_rotation)
        pads.append(
            {
                "number": number.group(1),
                "at": [round(global_x, 4), round(global_y, 4)],
                "size": [float(size.group(1)), float(size.group(2))],
                "net_code": int(net.group(1) or 0) if net else 0,
                "net": net.group(2) if net else "",
            }
        )
    return pads


def project_snapshot(project_dir: str | Path) -> JsonRecord:
    """Return one complete file-backed snapshot of a KiCad project."""
    project, schematic, board = _project_files(project_dir)
    sheets, components, nets = _schematic_snapshot(_export_netlist(schematic))
    board_content = _normalize_board_content(board.read_text(encoding="utf-8", errors="ignore"))
    parsed_footprints = _parse_board_footprint_blocks(board_content)
    footprints: list[JsonRecord] = []
    for reference, raw in sorted(parsed_footprints.items()):
        footprint = {
            key: value
            for key, value in raw.items()
            if key not in {"block", "start", "end", "pad_nets"}
        }
        footprint["reference"] = reference
        footprint["pads"] = _pad_positions(raw)
        footprints.append(footprint)
    tracks = _track_records(board_content)
    vias = _via_records(board_content)
    board_nets = {
        int(code): name
        for code, name in re.findall(rf"\(net\s+(\d+)\s+{STRING_PATTERN}\)", board_content)
    }
    for track in tracks:
        track["net"] = board_nets.get(int(track["net_code"]), "")
    for via in vias:
        via["net"] = board_nets.get(int(via["net_code"]), "")
    bounds = _edge_cuts_bounds(board_content)
    copper_layers = re.findall(r'^\s*\(\d+ "(?:F|B|In\d+)\.Cu" ', board_content, re.MULTILINE)
    return {
        "schema_version": "1.0",
        "project": {"name": project.stem, "directory": str(project.parent)},
        "schematic": {
            "sheets": sheets,
            "components": components,
            "nets": nets,
            "counts": {
                "sheets": len(sheets),
                "components": len(components),
                "nets": len(nets),
                "unconnected_nets": sum(bool(net["unconnected"]) for net in nets),
                "missing_footprints": sum(not component["footprint"] for component in components),
            },
        },
        "board": {
            "bounds_mm": list(bounds) if bounds else None,
            "copper_layers": len(copper_layers),
            "footprints": footprints,
            "tracks": tracks,
            "vias": vias,
            "zones": sum(1 for _ in _iter_blocks(board_content, "zone")),
            "counts": {
                "footprints": len(footprints),
                "tracks": len(tracks),
                "vias": len(vias),
                "nets": len(board_nets),
            },
        },
    }


def filter_snapshot(
    snapshot: JsonRecord, *, sheet: str = "", net: str = "", reference: str = ""
) -> JsonRecord:
    """Return a filtered copy useful for JSON/JSONL inspection."""
    if not any((sheet, net, reference)):
        return snapshot
    schematic = snapshot["schematic"]
    components = list(schematic["components"])
    nets = list(schematic["nets"])
    board = snapshot["board"]
    footprints = list(board["footprints"])
    if sheet:
        components = [item for item in components if sheet.casefold() in item["sheet"].casefold()]
        refs = {item["reference"] for item in components}
        nets = [item for item in nets if any(node["reference"] in refs for node in item["nodes"])]
    if net:
        nets = [item for item in nets if net.casefold() in item["name"].casefold()]
        refs = {node["reference"] for item in nets for node in item["nodes"]}
        components = [item for item in components if item["reference"] in refs]
        footprints = [item for item in footprints if item["reference"] in refs]
    if reference:
        components = [
            item for item in components if item["reference"].casefold() == reference.casefold()
        ]
        footprints = [
            item for item in footprints if item["reference"].casefold() == reference.casefold()
        ]
        nets = [
            item
            for item in nets
            if any(node["reference"].casefold() == reference.casefold() for node in item["nodes"])
        ]
    return {
        **snapshot,
        "schematic": {**schematic, "components": components, "nets": nets},
        "board": {**board, "footprints": footprints},
    }


def _truncate(value: str, width: int) -> str:
    return value if len(value) <= width else value[: max(1, width - 1)] + "…"


def ascii_map(snapshot: JsonRecord, *, zoom: int = 0, width: int = 100) -> str:
    """Render a stable multi-resolution text map suitable for terminals and diffs."""
    schematic = snapshot["schematic"]
    components = list(schematic["components"])
    nets = list(schematic["nets"])
    if zoom <= 0:
        grouped: dict[str, list[JsonRecord]] = {}
        for component in components:
            grouped.setdefault(component["sheet"] or "/", []).append(component)
        lines = [f"{snapshot['project']['name']}  SYSTEM MAP"]
        for sheet_name, items in sorted(grouped.items()):
            connected = sum(
                any(
                    node["reference"] in {item["reference"] for item in items}
                    for node in net["nodes"]
                )
                for net in nets
                if not net["unconnected"]
            )
            refs = " ".join(item["reference"] for item in items)
            lines.append(
                f"├─ {_truncate(sheet_name, 30):30} [{len(items):2} parts, {connected:2} nets]"
            )
            lines.append(f"│  {_truncate(refs, max(20, width - 4))}")
        return "\n".join(lines)
    if zoom == 1:
        lines = [f"{snapshot['project']['name']}  COMPONENT MAP"]
        for component in components:
            footprint = component["footprint"] or "MISSING_FOOTPRINT"
            lines.append(
                f"{component['sheet']:<28} {component['reference']:<7} "
                f"{_truncate(component['value'], 28):<28} {_truncate(footprint, 34)}"
            )
        return "\n".join(lines)
    if zoom == 2:
        lines = [f"{snapshot['project']['name']}  NET MAP"]
        for net in nets:
            nodes = " ─ ".join(
                f"{node['reference']}.{node['pin']}:{node['function'] or '?'}"
                for node in net["nodes"]
            )
            marker = "!" if net["unconnected"] else "•"
            lines.append(
                f"{marker} {_truncate(net['name'], 38):38} {_truncate(nodes, max(20, width - 42))}"
            )
        return "\n".join(lines)
    return _ascii_board(snapshot, width=width)


def _ascii_board(snapshot: JsonRecord, *, width: int) -> str:
    board = snapshot["board"]
    bounds = board["bounds_mm"]
    if not bounds:
        return f"{snapshot['project']['name']}  PCB MAP\n(no Edge.Cuts outline)"
    x1, y1, x2, y2 = map(float, bounds)
    columns = max(30, min(width, 160))
    rows = max(12, min(48, round(columns * (y2 - y1) / max(x2 - x1, 1.0) * 0.45)))
    grid = [[" " for _ in range(columns)] for _ in range(rows)]

    def cell(point: Iterable[float]) -> tuple[int, int]:
        x, y = point
        column = round((float(x) - x1) / max(x2 - x1, 0.001) * (columns - 1))
        row = round((float(y) - y1) / max(y2 - y1, 0.001) * (rows - 1))
        return max(0, min(columns - 1, column)), max(0, min(rows - 1, row))

    for column in range(columns):
        grid[0][column] = "─"
        grid[-1][column] = "─"
    for row in range(rows):
        grid[row][0] = "│"
        grid[row][-1] = "│"
    for track in board["tracks"]:
        start_column, start_row = cell(track["start"])
        end_column, end_row = cell(track["end"])
        steps = max(abs(end_column - start_column), abs(end_row - start_row), 1)
        for index in range(steps + 1):
            column = round(start_column + (end_column - start_column) * index / steps)
            row = round(start_row + (end_row - start_row) * index / steps)
            if 0 < column < columns - 1 and 0 < row < rows - 1:
                grid[row][column] = "·"
    for footprint in board["footprints"]:
        column, row = cell([footprint["x_mm"], footprint["y_mm"]])
        label = str(footprint["reference"])
        for offset, character in enumerate(label):
            target = column + offset
            if 0 < target < columns - 1 and 0 < row < rows - 1:
                grid[row][target] = character
    grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1] = "┌", "┐", "└", "┘"
    header = (
        f"{snapshot['project']['name']}  PCB MAP  "
        f"{x2 - x1:.1f}×{y2 - y1:.1f} mm  "
        f"FP={len(board['footprints'])} TRACK={len(board['tracks'])} VIA={len(board['vias'])}"
    )
    return header + "\n" + "\n".join("".join(row) for row in grid)


CRITICAL_NET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|[+_-])(GND|BAT|VBAT|VCC|3V3|3V8|5V|POWER)([+_-]|$)",
        r"USB.*(?:DP|DM|D\+|D-)",
        r"(?:RF|ANT|VCOM|VGH|VGL)",
        r"(?:CLK|XTAL|OSC)",
    )
)


def route_plan(
    snapshot: JsonRecord,
    net_name: str,
    *,
    layer: str = "F.Cu",
    width_mm: float = 0.25,
    clearance_mm: float = 0.5,
    allow_critical: bool = False,
) -> JsonRecord:
    """Plan simple Manhattan segments between PCB pads; never mutates a board."""
    critical = any(pattern.search(net_name) for pattern in CRITICAL_NET_PATTERNS)
    if critical and not allow_critical:
        return {
            "status": "refused",
            "net": net_name,
            "critical": True,
            "reason": "critical/high-current/differential/RF net requires --allow-critical",
            "segments": [],
        }
    board = snapshot["board"]
    endpoints: list[JsonRecord] = []
    for footprint in board["footprints"]:
        for pad in footprint["pads"]:
            if pad["net"] == net_name:
                endpoints.append(
                    {"reference": footprint["reference"], "pad": pad["number"], "at": pad["at"]}
                )
    if len(endpoints) < 2:
        return {
            "status": "blocked",
            "net": net_name,
            "critical": critical,
            "reason": f"need at least two placed pads on net; found {len(endpoints)}",
            "endpoints": endpoints,
            "segments": [],
        }

    remaining = endpoints[1:]
    connected = [endpoints[0]]
    edges: list[tuple[JsonRecord, JsonRecord]] = []
    while remaining:
        start, end = min(
            ((a, b) for a in connected for b in remaining),
            key=lambda pair: (
                abs(pair[0]["at"][0] - pair[1]["at"][0]) + abs(pair[0]["at"][1] - pair[1]["at"][1])
            ),
        )
        edges.append((start, end))
        connected.append(end)
        remaining.remove(end)

    segments: list[JsonRecord] = []
    for start, end in edges:
        x1, y1 = start["at"]
        x2, y2 = end["at"]
        candidates = [([x1, y1], [x2, y1], [x2, y2]), ([x1, y1], [x1, y2], [x2, y2])]
        points = min(
            candidates,
            key=lambda candidate: _route_collision_score(candidate, board, clearance_mm),
        )
        for first, second in zip(points, points[1:], strict=False):
            if first == second:
                continue
            segments.append(
                {
                    "x1": first[0],
                    "y1": first[1],
                    "x2": second[0],
                    "y2": second[1],
                    "width": width_mm,
                    "layer": layer,
                    "net": net_name,
                }
            )
    collisions = sum(
        _route_collision_score(
            [[segment["x1"], segment["y1"]], [segment["x2"], segment["y2"]]],
            board,
            clearance_mm,
        )
        for segment in segments
    )
    return {
        "status": "planned",
        "net": net_name,
        "critical": critical,
        "layer": layer,
        "width_mm": width_mm,
        "clearance_mm": clearance_mm,
        "endpoints": endpoints,
        "segments": segments,
        "notes": [
            "Manhattan/MST proposal only; KiCad DRC remains authoritative.",
            "Review geometry before applying; no differential-pair or impedance solver is used.",
        ],
        "collision_score": collisions,
    }


def _route_collision_score(
    points: list[list[float]] | tuple[list[float], ...],
    board: JsonRecord,
    clearance: float,
) -> int:
    if len(points) < 2:
        return 0
    score = 0
    for first, second in zip(points, points[1:], strict=False):
        sx1, sx2 = sorted((first[0], second[0]))
        sy1, sy2 = sorted((first[1], second[1]))
        for footprint in board["footprints"]:
            cx = float(footprint["x_mm"])
            cy = float(footprint["y_mm"])
            half_width = float(footprint["width_mm"]) / 2 + clearance
            half_height = float(footprint["height_mm"]) / 2 + clearance
            overlaps = (
                sx2 >= cx - half_width
                and sx1 <= cx + half_width
                and sy2 >= cy - half_height
                and sy1 <= cy + half_height
            )
            if overlaps:
                score += 1
    return score
