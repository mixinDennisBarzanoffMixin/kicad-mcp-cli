"""Headless project inspection, ASCII maps, and conservative route planning.

The normal MCP backend remains the authority for KiCad mutations.  This module
adds a fast, file-oriented planning surface that works with KiCad closed and
emits ordinary Python/JSON values for shell pipelines.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from heapq import heappop, heappush
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
from .utils.placement import (
    ForceDirectedConfig,
    PlacementComponent,
    PlacementNet,
    force_directed_placement,
    legalize_placement,
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
        "/Volumes/Apps/User Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
        "/Volumes/KiCad/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
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
            cwd=schematic.parent,
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


def _net_identity(block: str) -> tuple[int, str]:
    """Read either KiCad 9 numeric or KiCad 10 name-based net syntax."""
    named = re.search(rf"\(net(?:\s+(\d+))?\s+{STRING_PATTERN}\)", block)
    if named:
        return int(named.group(1) or 0), named.group(2)
    numeric = re.search(r"\(net\s+(\d+)\)", block)
    return (int(numeric.group(1)), "") if numeric else (0, "")


def _board_net_names(
    content: str,
    footprints: dict[str, JsonRecord] | None = None,
) -> list[str]:
    """Return semantic board nets across KiCad serialization versions."""
    names = {
        name
        for _code, name in re.findall(
            rf"\(net\s+(\d+)\s+{STRING_PATTERN}\)", content
        )
        if name
    }
    parsed = footprints if footprints is not None else _parse_board_footprint_blocks(content)
    for footprint in parsed.values():
        names.update(
            name for name in dict(footprint.get("pad_nets", {})).values() if name
        )
    for kind in ("segment", "arc", "via", "zone"):
        for block in _iter_blocks(content, kind):
            _code, name = _net_identity(block)
            if name:
                names.add(name)
            zone_name = re.search(rf"\(net_name\s+{STRING_PATTERN}\)", block)
            if zone_name and zone_name.group(1):
                names.add(zone_name.group(1))
    return sorted(names)


def _track_records(content: str) -> list[JsonRecord]:
    tracks: list[JsonRecord] = []
    for segment in _iter_blocks(content, "segment"):
        start = re.search(rf"\(start\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", segment)
        end = re.search(rf"\(end\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\)", segment)
        width = re.search(rf"\(width\s+({FLOAT_PATTERN})\)", segment)
        layer = re.search(r'\(layer\s+"([^"]+)"\)', segment)
        net_code, net_name = _net_identity(segment)
        if not (start and end and width and layer and (net_code or net_name)):
            continue
        x1, y1, x2, y2 = map(float, (*start.groups(), *end.groups()))
        tracks.append(
            {
                "start": [x1, y1],
                "end": [x2, y2],
                "width_mm": float(width.group(1)),
                "layer": layer.group(1),
                "net_code": net_code,
                "net": net_name,
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
        net_code, net_name = _net_identity(via)
        if at and size and drill and (net_code or net_name):
            vias.append(
                {
                    "at": [float(at.group(1)), float(at.group(2))],
                    "diameter_mm": float(size.group(1)),
                    "drill_mm": float(drill.group(1)),
                    "net_code": net_code,
                    "net": net_name,
                }
            )
    return vias


def _board_semantic_state(content: str) -> JsonRecord:
    """Return a formatting-insensitive board state for live/disk divergence checks."""
    normalized = _normalize_board_content(content)
    footprints = _parse_board_footprint_blocks(normalized)
    footprint_state = []
    for reference, footprint in sorted(footprints.items()):
        footprint_state.append(
            {
                "reference": reference,
                "x_mm": footprint.get("x_mm"),
                "y_mm": footprint.get("y_mm"),
                "rotation": footprint.get("rotation"),
                "layer": footprint.get("layer_name"),
                "pads": sorted(
                    (str(number), str(name))
                    for number, name in dict(footprint.get("pad_nets", {})).items()
                ),
            }
        )
    tracks = [
        {
            key: item[key]
            for key in ("start", "end", "width_mm", "layer", "net_code", "net")
        }
        for item in _track_records(normalized)
    ]
    vias = _via_records(normalized)
    zones = []
    for block in _iter_blocks(normalized, "zone"):
        uuid_match = re.search(r'\(uuid\s+"([^"]+)"\)', block)
        net_match = re.search(rf"\(net_name\s+{STRING_PATTERN}\)", block)
        layer_match = re.search(r'\(layer\s+"([^"]+)"\)', block)
        layers_match = re.search(r"\(layers\s+([^\)]+)\)", block)
        zones.append(
            {
                "uuid": uuid_match.group(1) if uuid_match else "",
                "net": net_match.group(1) if net_match else "",
                "layer": layer_match.group(1) if layer_match else "",
                "layers": " ".join(layers_match.group(1).split()) if layers_match else "",
            }
        )
    nets = _board_net_names(normalized, footprints)
    return {
        "bounds_mm": list(bounds) if (bounds := _edge_cuts_bounds(normalized)) else None,
        "footprints": footprint_state,
        "tracks": tracks,
        "vias": vias,
        "zones": sorted(zones, key=lambda item: (item["uuid"], item["net"])),
        "nets": nets,
    }


def _live_board_probe(project: Path, disk_content: str) -> JsonRecord:
    """Probe the official KiCad IPC API and compare its active board with disk."""
    try:
        from kipy import KiCad
        from kipy.proto.common.types.base_types_pb2 import DocumentType

        client = KiCad(timeout_ms=1500)
        documents = client.get_open_documents(DocumentType.DOCTYPE_PCB)
        document = next(
            (
                item
                for item in documents
                if Path(str(item.project.path)).resolve() == project.parent.resolve()
                and str(item.board_filename) == project.with_suffix(".kicad_pcb").name
            ),
            None,
        )
        if document is None:
            return {
                "status": "wrong-or-closed-project",
                "authority": "native-ipc",
                "api_version": str(client.get_api_version()),
                "kicad_version": str(client.get_version()),
                "open_boards": [
                    {"project": str(item.project.path), "file": str(item.board_filename)}
                    for item in documents
                ],
                "semantic_match": None,
            }
        board = client.get_board()
        live_state = _board_semantic_state(board.get_as_string())
        disk_state = _board_semantic_state(disk_content)
        return {
            "status": "connected",
            "authority": "native-ipc",
            "api_version": str(client.get_api_version()),
            "kicad_version": str(client.get_version()),
            "document": {
                "project": str(document.project.path),
                "file": str(document.board_filename),
            },
            "semantic_match": live_state == disk_state,
            "counts": {
                "footprints": len(live_state["footprints"]),
                "tracks": len(live_state["tracks"]),
                "vias": len(live_state["vias"]),
                "zones": len(live_state["zones"]),
                "nets": len(live_state["nets"]),
            },
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "authority": "native-ipc",
            "error": f"{type(exc).__name__}: {exc}",
            "semantic_match": None,
        }


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
        net_code, net_name = _net_identity(pad)
        if not (number and at and size):
            continue
        local_x = float(at.group(1)) * (-1 if back else 1)
        local_y = float(at.group(2))
        # KiCad board coordinates have +Y downward. At +90 degrees local +Y
        # becomes board +X, while local +X becomes board -Y.
        global_x = root_x + local_x * math.cos(root_rotation) + local_y * math.sin(root_rotation)
        global_y = root_y - local_x * math.sin(root_rotation) + local_y * math.cos(root_rotation)
        pads.append(
            {
                "number": number.group(1),
                "at": [round(global_x, 4), round(global_y, 4)],
                "size": [float(size.group(1)), float(size.group(2))],
                "net_code": net_code,
                "net": net_name,
            }
        )
    return pads


_GROUND_NET_NAMES = {"GND", "GNDA", "GNDD", "VSS", "VSSA", "VSSD", "PGND", "AGND"}


def _is_ground_net(net_name: str) -> bool:
    """Return whether a net name is an intentional current-return rail."""
    normalized = net_name.strip().upper().rsplit("/", 1)[-1]
    return normalized in _GROUND_NET_NAMES or normalized.endswith("_GND")


def _is_actionable_pad_net(net_name: str) -> bool:
    normalized = net_name.strip().casefold()
    return bool(normalized) and not normalized.startswith("unconnected-(") and normalized not in {
        "nc",
        "n/c",
        "no_connect",
    }


def _closest_pad_pair(
    left_pads: Iterable[JsonRecord],
    right_pads: Iterable[JsonRecord],
) -> JsonRecord | None:
    """Return the shortest same-net pad-center pair and its physical distance."""
    best: JsonRecord | None = None
    for left in left_pads:
        left_net = str(left.get("net", ""))
        if not _is_actionable_pad_net(left_net):
            continue
        for right in right_pads:
            if str(right.get("net", "")) != left_net:
                continue
            left_at = list(left.get("at", []))
            right_at = list(right.get("at", []))
            if len(left_at) != 2 or len(right_at) != 2:
                continue
            dx_mm = float(right_at[0]) - float(left_at[0])
            dy_mm = float(right_at[1]) - float(left_at[1])
            distance_mm = math.hypot(dx_mm, dy_mm)
            candidate: JsonRecord = {
                "net": left_net,
                "host_pad": str(left.get("number", "")),
                "cap_pad": str(right.get("number", "")),
                "host_at_mm": [round(float(left_at[0]), 4), round(float(left_at[1]), 4)],
                "cap_at_mm": [round(float(right_at[0]), 4), round(float(right_at[1]), 4)],
                "dx_mm": round(dx_mm, 4),
                "dy_mm": round(dy_mm, 4),
                "distance_mm": round(distance_mm, 4),
                "manhattan_mm": round(abs(dx_mm) + abs(dy_mm), 4),
            }
            if best is None or float(candidate["distance_mm"]) < float(best["distance_mm"]):
                best = candidate
    return best


def power_loop_report(
    snapshot: JsonRecord,
    decoupling_pairs: Iterable[JsonRecord],
    *,
    reference: str = "",
) -> JsonRecord:
    """Inspect every declared decoupler using actual same-net pad coordinates.

    Forward distance is the nearest host-power-pad to capacitor-pad distance on
    the same rail. Return distance is the corresponding ground-pad distance.
    Their sum is a simple current-loop proxy, not a routed-copper sign-off.
    """
    footprints = {
        str(item.get("reference", "")): item
        for item in snapshot.get("board", {}).get("footprints", [])
        if item.get("reference")
    }
    groups: list[JsonRecord] = []
    checked_caps = 0
    passing_caps = 0
    missing_refs: set[str] = set()

    for raw_pair in decoupling_pairs:
        host_ref = str(raw_pair.get("ic_ref", ""))
        if reference and host_ref != reference:
            continue
        max_distance_mm = float(raw_pair.get("max_distance_mm", 3.0))
        host = footprints.get(host_ref)
        members: list[JsonRecord] = []
        if host is None:
            missing_refs.add(host_ref)
        host_pads = list(host.get("pads", [])) if host else []
        host_signal_pads = [
            pad for pad in host_pads if not _is_ground_net(str(pad.get("net", "")))
        ]
        host_ground_pads = [
            pad for pad in host_pads if _is_ground_net(str(pad.get("net", "")))
        ]

        for cap_ref_value in raw_pair.get("cap_refs", []):
            cap_ref = str(cap_ref_value)
            cap = footprints.get(cap_ref)
            checked_caps += 1
            if cap is None or host is None:
                if cap is None:
                    missing_refs.add(cap_ref)
                members.append(
                    {
                        "reference": cap_ref,
                        "status": "missing",
                        "reason": (
                            "host footprint is missing"
                            if host is None
                            else "capacitor is missing"
                        ),
                    }
                )
                continue

            cap_pads = list(cap.get("pads", []))
            cap_signal_pads = [
                pad for pad in cap_pads if not _is_ground_net(str(pad.get("net", "")))
            ]
            cap_ground_pads = [
                pad for pad in cap_pads if _is_ground_net(str(pad.get("net", "")))
            ]
            forward = _closest_pad_pair(host_signal_pads, cap_signal_pads)
            return_path = _closest_pad_pair(host_ground_pads, cap_ground_pads)
            origin_distance_mm = math.hypot(
                float(cap.get("x_mm") or 0.0) - float(host.get("x_mm") or 0.0),
                float(cap.get("y_mm") or 0.0) - float(host.get("y_mm") or 0.0),
            )
            if forward is None:
                status = "fail"
                reason = "no shared non-ground pad net"
            elif float(forward["distance_mm"]) > max_distance_mm:
                status = "fail"
                reason = (
                    f"power-pad distance {float(forward['distance_mm']):.2f} mm exceeds "
                    f"{max_distance_mm:.2f} mm"
                )
            else:
                status = "pass"
                reason = "power-pad distance is within the declared limit"
                passing_caps += 1
            loop_proxy_mm = (
                round(float(forward["distance_mm"]) + float(return_path["distance_mm"]), 4)
                if forward is not None and return_path is not None
                else None
            )
            members.append(
                {
                    "reference": cap_ref,
                    "value": str(cap.get("value", "")),
                    "status": status,
                    "reason": reason,
                    "origin_distance_mm": round(origin_distance_mm, 4),
                    "forward": forward,
                    "return": return_path,
                    "loop_proxy_mm": loop_proxy_mm,
                }
            )

        group_status = (
            "pass"
            if members and all(item["status"] == "pass" for item in members)
            else "fail"
        )
        groups.append(
            {
                "host_reference": host_ref,
                "host_value": str(host.get("value", "")) if host else "",
                "status": group_status,
                "max_power_pad_distance_mm": max_distance_mm,
                "members": members,
                "counts": {
                    "capacitors": len(members),
                    "passing": sum(item["status"] == "pass" for item in members),
                    "failing": sum(item["status"] != "pass" for item in members),
                },
            }
        )

    failing_caps = checked_caps - passing_caps
    return {
        "schema_version": "1.0",
        "status": "pass" if groups and failing_caps == 0 else "fail",
        "authority": "saved-board-pad-geometry",
        "measurement": {
            "forward": "nearest same-net host-pad to capacitor-pad center distance",
            "return": "nearest ground-pad center distance",
            "loop_proxy": "forward distance plus return distance; routed copper not yet considered",
        },
        "summary": {
            "groups": len(groups),
            "capacitors": checked_caps,
            "passing": passing_caps,
            "failing": failing_caps,
            "missing_references": sorted(missing_refs),
        },
        "groups": groups,
    }


def format_power_loop_report(report: JsonRecord) -> str:
    """Render a compact terminal report while preserving JSON as authority."""
    summary = report["summary"]
    lines = [
        f"POWER LOOPS {str(report['status']).upper()} | "
        f"groups={summary['groups']} caps={summary['capacitors']} "
        f"pass={summary['passing']} fail={summary['failing']}",
        "distance = actual same-net pad centers; loop = forward + GND return proxy",
    ]
    for group in report["groups"]:
        lines.append(
            f"{group['host_reference']} {str(group['status']).upper()} "
            f"limit={float(group['max_power_pad_distance_mm']):.2f}mm"
        )
        for member in group["members"]:
            forward = member.get("forward")
            if forward is None:
                lines.append(
                    f"  {member['reference']}: {str(member['status']).upper()} — {member['reason']}"
                )
                continue
            loop = member.get("loop_proxy_mm")
            loop_text = f" loop={float(loop):.2f}mm" if loop is not None else " loop=n/a"
            lines.append(
                f"  {member['reference']} {str(member['status']).upper()} "
                f"{forward['net']} {forward['host_pad']}→{forward['cap_pad']} "
                f"forward={float(forward['distance_mm']):.2f}mm{loop_text}"
            )
    return "\n".join(lines)


def _rotate_local_offset(x_mm: float, y_mm: float, rotation_deg: float) -> tuple[float, float]:
    angle = math.radians(rotation_deg)
    return (
        x_mm * math.cos(angle) + y_mm * math.sin(angle),
        -x_mm * math.sin(angle) + y_mm * math.cos(angle),
    )


def _pad_local_offset(footprint: JsonRecord, pad: JsonRecord) -> tuple[float, float]:
    """Recover one pad's unrotated local offset from saved world geometry."""
    root_x = float(footprint.get("x_mm") or 0.0)
    root_y = float(footprint.get("y_mm") or 0.0)
    pad_at = list(pad.get("at", []))
    dx_mm = float(pad_at[0]) - root_x
    dy_mm = float(pad_at[1]) - root_y
    angle = math.radians(float(footprint.get("rotation") or 0.0))
    return (
        dx_mm * math.cos(angle) - dy_mm * math.sin(angle),
        dx_mm * math.sin(angle) + dy_mm * math.cos(angle),
    )


def _footprint_bounds_for_transform(
    footprint: JsonRecord,
    x_mm: float,
    y_mm: float,
    rotation_deg: float,
    *,
    margin_mm: float = 0.0,
    body_only: bool = False,
) -> tuple[float, float, float, float]:
    min_x_key = "body_bbox_min_x_mm" if body_only else "bbox_min_x_mm"
    min_y_key = "body_bbox_min_y_mm" if body_only else "bbox_min_y_mm"
    max_x_key = "body_bbox_max_x_mm" if body_only else "bbox_max_x_mm"
    max_y_key = "body_bbox_max_y_mm" if body_only else "bbox_max_y_mm"
    min_x = float(
        footprint.get(
            min_x_key,
            footprint.get("bbox_min_x_mm", -float(footprint.get("width_mm", 1.0)) / 2.0),
        )
    )
    min_y = float(
        footprint.get(
            min_y_key,
            footprint.get("bbox_min_y_mm", -float(footprint.get("height_mm", 1.0)) / 2.0),
        )
    )
    max_x = float(
        footprint.get(
            max_x_key,
            footprint.get("bbox_max_x_mm", float(footprint.get("width_mm", 1.0)) / 2.0),
        )
    )
    max_y = float(
        footprint.get(
            max_y_key,
            footprint.get("bbox_max_y_mm", float(footprint.get("height_mm", 1.0)) / 2.0),
        )
    )
    corners = [
        _rotate_local_offset(local_x, local_y, rotation_deg)
        for local_x, local_y in (
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        )
    ]
    xs = [x_mm + point[0] for point in corners]
    ys = [y_mm + point[1] for point in corners]
    return (
        min(xs) - margin_mm,
        min(ys) - margin_mm,
        max(xs) + margin_mm,
        max(ys) + margin_mm,
    )


def _rectangles_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def _transform_courtyard_polygons(
    footprint: JsonRecord,
    x_mm: float,
    y_mm: float,
    rotation_deg: float,
    *,
    margin_mm: float = 0.0,
) -> list[list[tuple[float, float]]]:
    """Return world-space courtyard loops, falling back to the parsed bounds."""
    raw_polygons = footprint.get("courtyard_polygons")
    polygons: list[list[tuple[float, float]]] = []
    if isinstance(raw_polygons, list):
        for raw_polygon in raw_polygons:
            if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
                continue
            polygon: list[tuple[float, float]] = []
            for raw_point in raw_polygon:
                if not isinstance(raw_point, list) or len(raw_point) != 2:
                    polygon = []
                    break
                offset_x, offset_y = _rotate_local_offset(
                    float(raw_point[0]), float(raw_point[1]), rotation_deg
                )
                polygon.append((x_mm + offset_x, y_mm + offset_y))
            if polygon:
                polygons.append(polygon)
    if polygons and margin_mm <= 0.0:
        return polygons
    # A true polygon offset is deliberately not approximated by moving each
    # vertex away from the centroid: that fails for concave courtyards.  When
    # the caller requests extra clearance, use the conservative expanded
    # bounding rectangle instead.
    min_x, min_y, max_x, max_y = _footprint_bounds_for_transform(
        footprint,
        x_mm,
        y_mm,
        rotation_deg,
        margin_mm=margin_mm,
    )
    return [[(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]]


def _polygon_bounds(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    cross = (point[1] - start[1]) * (end[0] - start[0]) - (
        point[0] - start[0]
    ) * (end[1] - start[1])
    if abs(cross) > epsilon:
        return False
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def _polygon_segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def orientation(
        left: tuple[float, float],
        middle: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (middle[1] - left[1]) * (right[0] - middle[0]) - (
            middle[0] - left[0]
        ) * (right[1] - middle[1])

    values = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True
    return any(
        abs(value) <= 1e-9 and _point_on_segment(point, segment_start, segment_end)
        for value, point, segment_start, segment_end in (
            (values[0], second_start, first_start, first_end),
            (values[1], second_end, first_start, first_end),
            (values[2], first_start, second_start, second_end),
            (values[3], first_end, second_start, second_end),
        )
    )


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            x_intersection = (end[0] - start[0]) * (point[1] - start[1]) / (
                end[1] - start[1]
            ) + start[0]
            if point[0] < x_intersection:
                inside = not inside
    return inside


def _polygons_overlap(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> bool:
    if not _rectangles_overlap(_polygon_bounds(left), _polygon_bounds(right)):
        return False
    for left_index, left_start in enumerate(left):
        left_end = left[(left_index + 1) % len(left)]
        for right_index, right_start in enumerate(right):
            right_end = right[(right_index + 1) % len(right)]
            if _polygon_segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return _point_in_polygon(left[0], right) or _point_in_polygon(right[0], left)


def power_loop_placement_plan(
    snapshot: JsonRecord,
    decoupling_pairs: Iterable[JsonRecord],
    *,
    reference: str = "",
    grid_mm: float = 0.25,
    courtyard_margin_mm: float = 0.0,
    yield_references: Iterable[str] = (),
    repack: bool = False,
) -> JsonRecord:
    """Plan capacitor root transforms around matching host power pads.

    This is intentionally a dry-run geometry proposal. It considers all four
    orthogonal capacitor rotations, keeps other footprints collision-free by
    courtyard proxy, and minimizes the forward-plus-ground-return distance.
    KiCad DRC remains the final acceptance authority when the plan is applied.
    """
    if grid_mm <= 0.0:
        raise ValueError("grid_mm must be greater than zero")
    pair_list = [dict(pair) for pair in decoupling_pairs]
    yielded_refs = {str(reference) for reference in yield_references}
    before = power_loop_report(snapshot, pair_list, reference=reference)
    footprints = {
        str(item.get("reference", "")): item
        for item in snapshot.get("board", {}).get("footprints", [])
        if item.get("reference")
    }
    board_bounds_raw = snapshot.get("board", {}).get("bounds_mm")
    if not isinstance(board_bounds_raw, list) or len(board_bounds_raw) != 4:
        return {
            "schema_version": "1.0",
            "status": "blocked",
            "reason": "board has no rectangular Edge.Cuts bounds",
            "placements": [],
            "before": before,
        }
    board_bounds = tuple(float(value) for value in board_bounds_raw)
    failing_hosts = {
        str(group["host_reference"])
        for group in before["groups"]
        if group["status"] != "pass"
    }
    selected_hosts = (
        {reference}
        if reference
        else {str(group["host_reference"]) for group in before["groups"]}
    )
    active_hosts = selected_hosts if repack else failing_hosts
    movable_cap_refs = {
        str(member["reference"])
        for group in before["groups"]
        if str(group["host_reference"]) in active_hosts
        for member in group["members"]
    }
    occupied: list[tuple[str, list[list[tuple[float, float]]]]] = []
    for footprint_ref, footprint in footprints.items():
        if footprint_ref in movable_cap_refs or footprint_ref in yielded_refs:
            continue
        occupied.append(
            (
                footprint_ref,
                _transform_courtyard_polygons(
                    footprint,
                    float(footprint.get("x_mm") or 0.0),
                    float(footprint.get("y_mm") or 0.0),
                    float(footprint.get("rotation") or 0.0),
                ),
            )
        )

    placements: list[JsonRecord] = []
    unresolved: list[JsonRecord] = []
    search_diagnostics: list[JsonRecord] = []
    group_by_host = {str(group["host_reference"]): group for group in before["groups"]}
    ordered_pairs = sorted(
        pair_list,
        key=lambda pair: (
            -len(pair.get("cap_refs", [])),
            -sum(
                float(footprints.get(str(cap_ref), {}).get("width_mm", 0.0))
                * float(footprints.get(str(cap_ref), {}).get("height_mm", 0.0))
                for cap_ref in pair.get("cap_refs", [])
            ),
            str(pair.get("ic_ref", "")),
        ),
    )
    for pair in ordered_pairs:
        host_ref = str(pair.get("ic_ref", ""))
        if reference and host_ref != reference:
            continue
        host = footprints.get(host_ref)
        group = group_by_host.get(host_ref)
        if host is None or group is None:
            unresolved.append({"reference": host_ref, "reason": "host footprint is missing"})
            continue
        if group["status"] == "pass" and not repack:
            continue
        member_by_ref = {str(member["reference"]): member for member in group["members"]}
        max_distance_mm = float(pair.get("max_distance_mm", 3.0))
        host_pads = list(host.get("pads", []))
        host_ground_pads = [
            pad for pad in host_pads if _is_ground_net(str(pad.get("net", "")))
        ]

        ordered_cap_refs = sorted(
            (str(value) for value in pair.get("cap_refs", [])),
            key=lambda cap_ref: -(
                float(footprints.get(cap_ref, {}).get("width_mm", 0.0))
                * float(footprints.get(cap_ref, {}).get("height_mm", 0.0))
            ),
        )
        group_candidates: list[
            tuple[str, list[tuple[float, JsonRecord, list[list[tuple[float, float]]]]]]
        ] = []
        for cap_ref in ordered_cap_refs:
            member = member_by_ref.get(cap_ref)
            cap = footprints.get(cap_ref)
            if cap is None or member is None or member.get("forward") is None:
                unresolved.append(
                    {
                        "reference": cap_ref,
                        "host_reference": host_ref,
                        "reason": str(member.get("reason", "capacitor is missing"))
                        if member
                        else "capacitor is missing",
                    }
                )
                continue
            rail = str(member["forward"]["net"])
            matching_host_pads = [pad for pad in host_pads if str(pad.get("net", "")) == rail]
            cap_pads = list(cap.get("pads", []))
            matching_cap_pads = [pad for pad in cap_pads if str(pad.get("net", "")) == rail]
            cap_ground_pads = [
                pad for pad in cap_pads if _is_ground_net(str(pad.get("net", "")))
            ]
            candidates: list[
                tuple[float, JsonRecord, list[list[tuple[float, float]]]]
            ] = []
            blocker_hits: Counter[str] = Counter()
            in_bounds_transforms = 0
            for host_pad in matching_host_pads:
                host_at = list(host_pad.get("at", []))
                outward_x = float(host_at[0]) - float(host.get("x_mm") or 0.0)
                outward_y = float(host_at[1]) - float(host.get("y_mm") or 0.0)
                base_angle = math.atan2(outward_y, outward_x) if outward_x or outward_y else 0.0
                angle_offsets = (0, 45, -45, 90, -90, 135, -135, 180)
                radius_steps = max(1, int(math.floor(max_distance_mm / grid_mm)))
                radii = [
                    max(1.25, grid_mm * step)
                    for step in range(1, radius_steps + 1)
                    if max(1.25, grid_mm * step) <= max_distance_mm
                ]
                radii = sorted(set(round(value, 6) for value in radii))
                for rotation_deg in (0.0, 90.0, 180.0, 270.0):
                    for cap_pad in matching_cap_pads:
                        local_pad_x, local_pad_y = _pad_local_offset(cap, cap_pad)
                        pad_offset_x, pad_offset_y = _rotate_local_offset(
                            local_pad_x, local_pad_y, rotation_deg
                        )
                        for angle_offset in angle_offsets:
                            angle = base_angle + math.radians(angle_offset)
                            unit_x, unit_y = math.cos(angle), math.sin(angle)
                            for radius_mm in radii:
                                desired_pad_x = float(host_at[0]) + unit_x * radius_mm
                                desired_pad_y = float(host_at[1]) + unit_y * radius_mm
                                root_x = round((desired_pad_x - pad_offset_x) / grid_mm) * grid_mm
                                root_y = round((desired_pad_y - pad_offset_y) / grid_mm) * grid_mm
                                cap_pad_x = root_x + pad_offset_x
                                cap_pad_y = root_y + pad_offset_y
                                forward_mm = math.hypot(
                                    cap_pad_x - float(host_at[0]),
                                    cap_pad_y - float(host_at[1]),
                                )
                                if forward_mm > max_distance_mm + 1e-6:
                                    continue
                                bounds = _footprint_bounds_for_transform(
                                    cap,
                                    root_x,
                                    root_y,
                                    rotation_deg,
                                    margin_mm=courtyard_margin_mm,
                                )
                                if (
                                    bounds[0] < board_bounds[0]
                                    or bounds[1] < board_bounds[1]
                                    or bounds[2] > board_bounds[2]
                                    or bounds[3] > board_bounds[3]
                                ):
                                    continue
                                in_bounds_transforms += 1
                                candidate_polygons = _transform_courtyard_polygons(
                                    cap,
                                    root_x,
                                    root_y,
                                    rotation_deg,
                                    margin_mm=courtyard_margin_mm,
                                )
                                blocking_refs = {
                                    occupied_ref
                                    for occupied_ref, occupied_polygons in occupied
                                    if any(
                                        _polygons_overlap(candidate_polygon, occupied_polygon)
                                        for candidate_polygon in candidate_polygons
                                        for occupied_polygon in occupied_polygons
                                    )
                                }
                                if blocking_refs:
                                    blocker_hits.update(blocking_refs)
                                    continue
                                return_mm: float | None = None
                                for cap_ground_pad in cap_ground_pads:
                                    local_ground = _pad_local_offset(cap, cap_ground_pad)
                                    ground_offset = _rotate_local_offset(
                                        *local_ground, rotation_deg
                                    )
                                    cap_ground_x = root_x + ground_offset[0]
                                    cap_ground_y = root_y + ground_offset[1]
                                    for host_ground_pad in host_ground_pads:
                                        ground_at = list(host_ground_pad.get("at", []))
                                        distance = math.hypot(
                                            cap_ground_x - float(ground_at[0]),
                                            cap_ground_y - float(ground_at[1]),
                                        )
                                        if return_mm is None or distance < return_mm:
                                            return_mm = distance
                                body_outward = (
                                    (root_x - float(host_at[0])) * unit_x
                                    + (root_y - float(host_at[1])) * unit_y
                                )
                                move_mm = math.hypot(
                                    root_x - float(cap.get("x_mm") or 0.0),
                                    root_y - float(cap.get("y_mm") or 0.0),
                                )
                                current_rotation = float(cap.get("rotation") or 0.0)
                                rotation_delta = abs(
                                    (rotation_deg - current_rotation + 180.0) % 360.0 - 180.0
                                )
                                score = (
                                    forward_mm
                                    + (return_mm if return_mm is not None else max_distance_mm)
                                    + max(0.0, -body_outward) * 20.0
                                    + move_mm * 0.002
                                    # A rotation also swings reference/value fields and can
                                    # create silkscreen regressions even when the copper and
                                    # courtyard remain legal. Prefer the saved orientation
                                    # unless rotation materially improves the electrical loop.
                                    + (8.0 if rotation_delta > 1e-3 else 0.0)
                                )
                                candidates.append(
                                    (
                                        score,
                                        {
                                            "reference": cap_ref,
                                            "host_reference": host_ref,
                                            "rail": rail,
                                            "from": [
                                                float(cap.get("x_mm") or 0.0),
                                                float(cap.get("y_mm") or 0.0),
                                            ],
                                            "to": [round(root_x, 4), round(root_y, 4)],
                                            "from_rotation": float(cap.get("rotation") or 0.0),
                                            "rotation": rotation_deg,
                                            "host_pad": str(host_pad.get("number", "")),
                                            "cap_pad": str(cap_pad.get("number", "")),
                                            "forward_distance_mm": round(forward_mm, 4),
                                            "return_distance_mm": (
                                                round(return_mm, 4)
                                                if return_mm is not None
                                                else None
                                            ),
                                            "loop_proxy_mm": (
                                                round(forward_mm + return_mm, 4)
                                                if return_mm is not None
                                                else None
                                            ),
                                        },
                                        candidate_polygons,
                                    )
                                )
            search_diagnostics.append(
                {
                    "reference": cap_ref,
                    "host_reference": host_ref,
                    "rail": rail,
                    "in_bounds_transforms_checked": in_bounds_transforms,
                    "collision_free_candidates": len(candidates),
                    "blocking_footprints": [
                        {"reference": blocker, "hits": hits}
                        for blocker, hits in blocker_hits.most_common(12)
                    ],
                }
            )
            if not candidates:
                unresolved.append(
                    {
                        "reference": cap_ref,
                        "host_reference": host_ref,
                        "rail": rail,
                        "reason": "no in-bounds collision-free pad-aware candidate was found",
                        "in_bounds_transforms_checked": in_bounds_transforms,
                        "blocking_footprints": [
                            {"reference": blocker, "hits": hits}
                            for blocker, hits in blocker_hits.most_common(12)
                        ],
                    }
                )
                continue
            # The same root transform may be reached from several host pads or
            # angular samples. Keep its best electrical interpretation before
            # group packing so the bounded search remains small and diverse.
            by_transform: dict[
                tuple[float, float, float],
                tuple[float, JsonRecord, list[list[tuple[float, float]]]],
            ] = {}
            for candidate in candidates:
                candidate_to = candidate[1]["to"]
                transform = (
                    float(candidate_to[0]),
                    float(candidate_to[1]),
                    float(candidate[1]["rotation"]),
                )
                current = by_transform.get(transform)
                if current is None or candidate[0] < current[0]:
                    by_transform[transform] = candidate
            ranked_candidates = sorted(
                by_transform.values(),
                key=lambda item: (
                    item[0],
                    float(item[1]["to"][1]),
                    float(item[1]["to"][0]),
                    float(item[1]["rotation"]),
                ),
            )[:1024]
            group_candidates.append((cap_ref, ranked_candidates))

        # Capacitors around one IC are a coupled placement problem. A greedy
        # first choice can consume the only legal location for a second, larger
        # capacitor. Keep a bounded beam of complete non-overlapping cluster
        # arrangements so earlier choices can be reconsidered deterministically.
        beam: list[
            tuple[
                float,
                list[tuple[str, JsonRecord, list[list[tuple[float, float]]]]],
                list[list[tuple[float, float]]],
            ]
        ] = [(0.0, [], [])]
        packing_failed = False
        for cap_ref, candidates in sorted(group_candidates, key=lambda item: len(item[1])):
            next_beam: list[
                tuple[
                    float,
                    list[tuple[str, JsonRecord, list[list[tuple[float, float]]]]],
                    list[list[tuple[float, float]]],
                ]
            ] = []
            for accumulated_score, selected_items, selected_polygons in beam:
                for candidate_score, candidate, candidate_polygons in candidates:
                    if any(
                        _polygons_overlap(candidate_polygon, selected_polygon)
                        for candidate_polygon in candidate_polygons
                        for selected_polygon in selected_polygons
                    ):
                        continue
                    next_beam.append(
                        (
                            accumulated_score + candidate_score,
                            [*selected_items, (cap_ref, candidate, candidate_polygons)],
                            [*selected_polygons, *candidate_polygons],
                        )
                    )
            if not next_beam:
                unresolved.append(
                    {
                        "reference": cap_ref,
                        "host_reference": host_ref,
                        "reason": "no collision-free complete capacitor-cluster packing was found",
                        "candidate_counts": {
                            reference: len(cap_candidates)
                            for reference, cap_candidates in group_candidates
                        },
                        "candidate_samples": {
                            reference: [
                                {
                                    "to": candidate[1]["to"],
                                    "rotation": candidate[1]["rotation"],
                                    "score": round(candidate[0], 4),
                                }
                                for candidate in cap_candidates[:12]
                            ]
                            for reference, cap_candidates in group_candidates
                        },
                    }
                )
                packing_failed = True
                break
            next_beam.sort(
                key=lambda item: (
                    item[0],
                    tuple(
                        (
                            str(selected[1]["reference"]),
                            float(selected[1]["to"][1]),
                            float(selected[1]["to"][0]),
                            float(selected[1]["rotation"]),
                        )
                        for selected in item[1]
                    ),
                )
            )
            beam = next_beam[:2048]
        if group_candidates and not packing_failed:
            _score, selected_items, _selected_polygons = beam[0]
            for cap_ref, selected, selected_polygons in selected_items:
                placements.append(selected)
                occupied.append((cap_ref, selected_polygons))

    proposed_snapshot = copy.deepcopy(snapshot)
    proposed_by_ref = {
        str(item.get("reference", "")): item
        for item in proposed_snapshot.get("board", {}).get("footprints", [])
    }
    for placement in placements:
        cap = proposed_by_ref[str(placement["reference"])]
        new_x, new_y = (float(value) for value in placement["to"])
        new_rotation = float(placement["rotation"])
        for pad in cap.get("pads", []):
            local_x, local_y = _pad_local_offset(cap, pad)
            offset_x, offset_y = _rotate_local_offset(local_x, local_y, new_rotation)
            pad["at"] = [round(new_x + offset_x, 4), round(new_y + offset_y, 4)]
        cap["x_mm"] = new_x
        cap["y_mm"] = new_y
        cap["rotation"] = new_rotation
    after = power_loop_report(proposed_snapshot, pair_list, reference=reference)
    status = "planned" if not unresolved and after["status"] == "pass" else "blocked"
    return {
        "schema_version": "1.0",
        "status": status,
        "authority": "saved-board-pad-geometry",
        "reason": (
            "all declared failing capacitors received a pad-aware proposal"
            if status == "planned"
            else "one or more declared power-loop constraints remain unresolved"
        ),
        "grid_mm": grid_mm,
        "courtyard_margin_mm": courtyard_margin_mm,
        "placements": placements,
        "unresolved": unresolved,
        "search_diagnostics": search_diagnostics,
        "yielded_references": sorted(yielded_refs),
        "repack": repack,
        "requires_relocation": sorted(yielded_refs),
        "before": before,
        "after": after,
    }


def project_snapshot(
    project_dir: str | Path,
    *,
    board_content: str | None = None,
    board_source: str = "",
) -> JsonRecord:
    """Return one complete file-backed snapshot of a KiCad project."""
    project, schematic, board = _project_files(project_dir)
    sheets, components, nets = _schematic_snapshot(_export_netlist(schematic))
    candidate_supplied = board_content is not None
    normalized_board_content = _normalize_board_content(
        board_content
        if board_content is not None
        else board.read_text(encoding="utf-8", errors="ignore")
    )
    live_board = (
        {
            "status": "not-applicable",
            "authority": "offline-candidate",
            "semantic_match": None,
            "source": board_source or "supplied board content",
        }
        if candidate_supplied
        else _live_board_probe(project, normalized_board_content)
    )
    parsed_footprints = _parse_board_footprint_blocks(normalized_board_content)
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
    tracks = _track_records(normalized_board_content)
    vias = _via_records(normalized_board_content)
    board_nets = {
        int(code): name
        for code, name in re.findall(
            rf"\(net\s+(\d+)\s+{STRING_PATTERN}\)", normalized_board_content
        )
    }
    for track in tracks:
        if not track.get("net"):
            track["net"] = board_nets.get(int(track["net_code"]), "")
    for via in vias:
        if not via.get("net"):
            via["net"] = board_nets.get(int(via["net_code"]), "")
    board_net_names = _board_net_names(normalized_board_content, parsed_footprints)
    bounds = _edge_cuts_bounds(normalized_board_content)
    copper_layers = re.findall(
        r'^\s*\(\d+ "(?:F|B|In\d+)\.Cu" ', normalized_board_content, re.MULTILINE
    )
    return {
        "schema_version": "1.1",
        "project": {"name": project.stem, "directory": str(project.parent)},
        "schematic": {
            "authority": "kicad-cli-netlist",
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
            "authority": "offline-candidate-kicad-pcb"
            if candidate_supplied
            else "file-backed-kicad-pcb",
            "source": board_source or str(board),
            "live_ipc": live_board,
            "bounds_mm": list(bounds) if bounds else None,
            "copper_layers": len(copper_layers),
            "footprints": footprints,
            "tracks": tracks,
            "vias": vias,
            "zones": sum(1 for _ in _iter_blocks(normalized_board_content, "zone")),
            "counts": {
                "footprints": len(footprints),
                "tracks": len(tracks),
                "vias": len(vias),
                "nets": len(board_net_names),
            },
        },
    }


def authority_report(project_dir: str | Path) -> JsonRecord:
    """Explain which KiCad authority backs each operation in the current runtime."""
    snapshot = project_snapshot(project_dir)
    live = snapshot["board"]["live_ipc"]
    live_ready = live.get("status") == "connected"
    synchronized = live.get("semantic_match") is True
    return {
        "schema_version": "1.0",
        "status": "pass" if live_ready and synchronized else "review",
        "project": snapshot["project"],
        "authorities": {
            "schematic_read": "kicad-cli-netlist",
            "schematic_write": "transactional-file-fallback",
            "schematic_erc": "kicad-cli",
            "schematic_render": "kicad-cli",
            "board_read": "native-ipc" if live_ready else "file-backed-kicad-pcb",
            "board_write": "native-ipc" if live_ready else "unavailable",
            "board_drc": "kicad-cli",
            "board_export": "kicad-cli",
        },
        "live_ipc": live,
        "policy": {
            "board_mutation_allowed": live_ready and synchronized,
            "schematic_mutation_requires_transaction": True,
            "post_edit_verify_required": True,
        },
        "limitations": [
            "KiCad 10 IPC requires a running GUI instance.",
            "KiCad 10 does not expose the required schematic editing surface; guarded file "
            "transactions remain necessary for schematic authoring.",
            "Board planning is blocked when live IPC and the saved board diverge.",
        ],
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


def connectivity_proof(
    snapshot: JsonRecord,
    *,
    sheet: str = "",
    reference: str = "",
    net: str = "",
) -> JsonRecord:
    """Return deterministic pin-to-net-to-peer evidence for shell review.

    KiCad's exported netlist is the authority here.  The report intentionally
    repeats peer information for each selected pin so an LLM or a jq pipeline
    can answer connectivity questions without reconstructing a graph from raw
    schematic S-expressions.
    """
    schematic = snapshot["schematic"]
    components = list(schematic["components"])
    if sheet:
        components = [item for item in components if sheet.casefold() in item["sheet"].casefold()]
    if reference:
        components = [
            item for item in components if item["reference"].casefold() == reference.casefold()
        ]
    selected_refs = {item["reference"] for item in components}
    footprints = {item["reference"]: item for item in snapshot["board"]["footprints"]}
    rows: list[JsonRecord] = []
    seen_pins: dict[tuple[str, str], list[str]] = {}
    singleton_nets: set[str] = set()
    unconnected_pins = 0
    intentional_no_connects = 0
    matched_requested_nets: set[str] = set()

    for candidate in schematic["nets"]:
        net_name = str(candidate["name"])
        if net and net.casefold() not in net_name.casefold():
            continue
        nodes = list(candidate["nodes"])
        selected_nodes = [node for node in nodes if node["reference"] in selected_refs]
        if net and selected_nodes:
            matched_requested_nets.add(net_name)
        intentional_nodes = [
            node for node in selected_nodes if "+no_connect" in str(node.get("type", ""))
        ]
        if len(nodes) == 1 and selected_nodes and not intentional_nodes:
            singleton_nets.add(net_name)
        for node in selected_nodes:
            key = (str(node["reference"]), str(node["pin"]))
            seen_pins.setdefault(key, []).append(net_name)
            peers = [
                {
                    "reference": str(peer["reference"]),
                    "pin": str(peer["pin"]),
                    "function": str(peer.get("function", "")),
                    "type": str(peer.get("type", "")),
                }
                for peer in nodes
                if (str(peer["reference"]), str(peer["pin"])) != key
            ]
            is_unconnected = bool(candidate["unconnected"])
            intentional_no_connect = "+no_connect" in str(node.get("type", ""))
            unconnected_pins += int(is_unconnected and not intentional_no_connect)
            intentional_no_connects += int(intentional_no_connect)
            rows.append(
                {
                    "reference": key[0],
                    "pin": key[1],
                    "function": str(node.get("function", "")),
                    "type": str(node.get("type", "")),
                    "net": net_name,
                    "unconnected": is_unconnected,
                    "intentional_no_connect": intentional_no_connect,
                    "peer_count": len(peers),
                    "peers": peers,
                    "board_pad_net": next(
                        (
                            str(pad.get("net", ""))
                            for pad in footprints.get(key[0], {}).get("pads", [])
                            if str(pad.get("number", "")) == key[1]
                        ),
                        "",
                    ),
                }
            )

    duplicate_assignments = [
        {"reference": key[0], "pin": key[1], "nets": names}
        for key, names in sorted(seen_pins.items())
        if len(set(names)) > 1
    ]
    missing_footprints = sorted(
        item["reference"] for item in components if not item.get("footprint")
    )

    def comparable_net_name(name: object) -> str:
        # KiCad serializes '/' inside auto-generated unconnected net names as
        # the literal token ``{slash}`` when a pin function contains a slash.
        return str(name).replace("{slash}", "/")

    board_mismatches = [
        {
            "reference": row["reference"],
            "pin": row["pin"],
            "schematic_net": row["net"],
            "board_net": row["board_pad_net"],
        }
        for row in rows
        if row["board_pad_net"]
        and comparable_net_name(row["board_pad_net"]) != comparable_net_name(row["net"])
    ]
    requested_net_not_found = []
    if net and (not matched_requested_nets or not rows):
        requested_net_not_found.append(
            {
                "code": "requested_net_not_found",
                "requested": net,
                "match_semantics": "case-insensitive substring",
                "selected_components": len(components),
                "reason": "explicit net filter matched no net pins in the selected scope",
            }
        )
    requested_reference_not_found = []
    if reference and not components:
        requested_reference_not_found.append(
            {
                "code": "requested_reference_not_found",
                "requested": reference,
                "match_semantics": "case-insensitive exact reference",
                "reason": "explicit reference filter matched no component in the selected scope",
            }
        )
    elif reference and not net and not rows:
        requested_reference_not_found.append(
            {
                "code": "requested_reference_has_no_pins",
                "requested": reference,
                "match_semantics": "case-insensitive exact reference",
                "reason": "explicit reference matched a component but produced no netlisted pins",
            }
        )
    filter_failure = bool(requested_net_not_found or requested_reference_not_found)
    status = "fail" if duplicate_assignments or board_mismatches or filter_failure else "review"
    if (
        not unconnected_pins
        and not singleton_nets
        and not missing_footprints
        and status == "review"
    ):
        status = "pass"
    return {
        "schema_version": "1.0",
        "project": snapshot["project"]["name"],
        "filters": {"sheet": sheet, "reference": reference, "net": net},
        "status": status,
        "filter_semantics": {
            "sheet": "case-insensitive substring",
            "reference": "case-insensitive exact reference",
            "net": "case-insensitive substring",
        },
        "summary": {
            "components": len(components),
            "pins": len(rows),
            "nets": len({row["net"] for row in rows}),
            "unconnected_pins": unconnected_pins,
            "intentional_no_connects": intentional_no_connects,
            "singleton_nets": len(singleton_nets),
            "missing_footprints": len(missing_footprints),
            "duplicate_pin_assignments": len(duplicate_assignments),
            "board_net_mismatches": len(board_mismatches),
        },
        "findings": {
            "singleton_nets": sorted(singleton_nets),
            "missing_footprints": missing_footprints,
            "duplicate_pin_assignments": duplicate_assignments,
            "board_net_mismatches": board_mismatches,
            "requested_net_not_found": requested_net_not_found,
            "requested_reference_not_found": requested_reference_not_found,
        },
        "pins": sorted(rows, key=lambda row: (row["reference"], row["pin"], row["net"])),
    }


def _erc_evidence(schematic: Path, *, sheet: str = "") -> JsonRecord:
    """Run KiCad ERC and return stable, optionally sheet-scoped evidence."""
    with tempfile.TemporaryDirectory(prefix="kicadq-erc-") as temporary:
        output = Path(temporary) / "erc.json"
        process = subprocess.run(
            [
                _kicad_cli(),
                "sch",
                "erc",
                str(schematic),
                "--format",
                "json",
                "--severity-all",
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=schematic.parent,
        )
        if process.returncode != 0 or not output.is_file():
            diagnostic = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"KiCad ERC failed: {diagnostic}")
        payload = json.loads(output.read_text(encoding="utf-8"))

    findings: list[JsonRecord] = []
    matched_sheets: list[str] = []
    for page in payload.get("sheets", []):
        path = str(page.get("path", ""))
        if sheet and sheet.casefold() not in path.casefold():
            continue
        matched_sheets.append(path)
        for violation in page.get("violations", []):
            findings.append({"sheet": path, **violation})
    errors = sum(item.get("severity") == "error" for item in findings)
    warnings = sum(item.get("severity") == "warning" for item in findings)
    return {
        "status": "fail" if errors else "review" if warnings else "pass",
        "summary": {
            "sheets": len(matched_sheets),
            "violations": len(findings),
            "errors": errors,
            "warnings": warnings,
        },
        "findings": findings,
        "matched_sheets": matched_sheets,
        "kicad_version": payload.get("kicad_version", ""),
    }


def _drc_finding_key(item: JsonRecord) -> str:
    """Identify one DRC relation independent of order and moved coordinates."""
    canonical = {
        key: item.get(key)
        for key in ("kind", "type", "severity")
        if item.get(key) is not None
    }
    raw_items = item.get("items")
    if isinstance(raw_items, list):
        children: list[JsonRecord] = []
        for child in raw_items:
            if not isinstance(child, dict):
                continue
            # UUIDs identify the actual pad, field, graphic, or footprint.
            # Positions change during a legal root transform and must not turn
            # an unchanged intrinsic footprint violation into a new finding.
            if child.get("uuid"):
                children.append({"uuid": str(child["uuid"])})
            else:
                children.append({"description": str(child.get("description", ""))})
        canonical["items"] = sorted(
            children,
            key=lambda child: json.dumps(child, sort_keys=True, separators=(",", ":")),
        )
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def board_drc_evidence(project_dir: str | Path, *, board_content: str | None = None) -> JsonRecord:
    """Run KiCad DRC on saved or supplied serialized board state."""
    project, _schematic, board = _project_files(project_dir)
    with tempfile.TemporaryDirectory(prefix="kicadq-drc-") as temporary:
        stage = Path(temporary)
        staged_project = stage / project.name
        staged_board = stage / board.name
        shutil.copy2(project, staged_project)
        for rules in project.parent.glob("*.kicad_dru"):
            shutil.copy2(rules, stage / rules.name)
        staged_board.write_text(
            board_content
            if board_content is not None
            else board.read_text(encoding="utf-8", errors="ignore"),
            encoding="utf-8",
        )
        output = stage / "drc.json"
        process = subprocess.run(
            [
                _kicad_cli(),
                "pcb",
                "drc",
                str(staged_board),
                "--format",
                "json",
                "--severity-all",
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0 or not output.is_file():
            diagnostic = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"KiCad DRC failed: {diagnostic}")
        payload = json.loads(output.read_text(encoding="utf-8"))
    findings = [{"kind": "violation", **item} for item in payload.get("violations", [])] + [
        {"kind": "unconnected", **item} for item in payload.get("unconnected_items", [])
    ]
    keys = sorted(_drc_finding_key(item) for item in findings)
    return {
        "status": "fail" if findings else "pass",
        "summary": {
            "violations": len(payload.get("violations", [])),
            "unconnected_items": len(payload.get("unconnected_items", [])),
        },
        "findings": findings,
        "finding_keys": keys,
        "kicad_version": payload.get("kicad_version", ""),
    }


def _source_integrity_evidence(
    project: Path, snapshot: JsonRecord, *, sheet: str = ""
) -> JsonRecord:
    """Check selected schematic sources for transaction-breaking corruption."""
    from .tools.schematic import _duplicate_uuids, _validate_schematic_text

    findings: list[JsonRecord] = []
    checked: list[str] = []
    for page in snapshot["schematic"]["sheets"]:
        if sheet and sheet.casefold() not in str(page["name"]).casefold():
            continue
        path = project.parent / str(page["file"])
        checked.append(str(path))
        try:
            content = path.read_text(encoding="utf-8")
            _validate_schematic_text(content)
        except (OSError, ValueError) as exc:
            findings.append({"file": str(path), "type": "invalid_schematic", "detail": str(exc)})
            continue
        duplicates = sorted(_duplicate_uuids(content))
        if duplicates:
            findings.append(
                {
                    "file": str(path),
                    "type": "duplicate_element_uuids",
                    "detail": duplicates,
                }
            )
        whitespace_lines = len(re.findall(r"(?m)^[ \t]+$", content))
        if whitespace_lines:
            findings.append(
                {
                    "file": str(path),
                    "type": "whitespace_only_lines",
                    "detail": whitespace_lines,
                }
            )
        if "<<<<<<<" in content or ">>>>>>>" in content:
            findings.append({"file": str(path), "type": "merge_conflict_markers", "detail": True})
    return {
        "status": "fail" if findings else "pass",
        "summary": {"files": len(checked), "findings": len(findings)},
        "findings": findings,
        "files": checked,
    }


def _render_verification_svgs(
    schematic: Path,
    snapshot: JsonRecord,
    output_dir: Path,
    *,
    sheet: str = "",
) -> list[str]:
    """Render the selected schematic sources with visible hop-overs."""
    selected_pages = [
        page
        for page in snapshot["schematic"]["sheets"]
        if not sheet or sheet.casefold() in str(page["name"]).casefold()
    ]
    if sheet and not selected_pages:
        raise ValueError(f"sheet filter matched no pages: {sheet}")
    targets = (
        [schematic.parent / str(page["file"]) for page in selected_pages] if sheet else [schematic]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for target in targets:
        expected = output_dir / f"{target.stem}.svg"
        expected.unlink(missing_ok=True)
        command = [
            _kicad_cli(),
            "sch",
            "export",
            "svg",
            str(target),
            "-o",
            str(output_dir),
            "--exclude-drawing-sheet",
            "--draw-hop-over",
        ]
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=target.parent,
        )
        if process.returncode != 0 or not expected.is_file():
            diagnostic = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"KiCad SVG export failed for {target.name}: {diagnostic}")
        artifacts.append(str(expected))
    return artifacts


def verification_report(
    project_dir: str | Path,
    *,
    sheet: str = "",
    reference: str = "",
    net: str = "",
    artifacts_dir: str | Path | None = None,
) -> JsonRecord:
    """Bundle parse integrity, connectivity proof, ERC, and optional SVG evidence."""
    project, schematic, _board = _project_files(project_dir)
    snapshot = project_snapshot(project_dir)
    integrity = _source_integrity_evidence(project, snapshot, sheet=sheet)
    proof = connectivity_proof(snapshot, sheet=sheet, reference=reference, net=net)
    erc = _erc_evidence(schematic, sheet=sheet)
    artifacts: list[str] = []
    if artifacts_dir is not None:
        artifacts = _render_verification_svgs(
            schematic,
            snapshot,
            Path(artifacts_dir).expanduser().resolve(),
            sheet=sheet,
        )
    statuses = {integrity["status"], proof["status"], erc["status"]}
    status = "fail" if "fail" in statuses else "review" if "review" in statuses else "pass"
    return {
        "schema_version": "1.0",
        "project": snapshot["project"],
        "filters": {"sheet": sheet, "reference": reference, "net": net},
        "status": status,
        "checks": {"source_integrity": integrity, "connectivity": proof, "erc": erc},
        "artifacts": artifacts,
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
    board_x1, board_y1, board_x2, board_y2 = map(float, bounds)
    x1, y1, x2, y2 = board_x1, board_y1, board_x2, board_y2
    footprints = list(board["footprints"])
    total_footprints = int(board.get("counts", {}).get("footprints", len(footprints)))
    focused = bool(footprints) and len(footprints) < total_footprints
    if focused:
        footprint_xs = [float(item["x_mm"]) for item in footprints if item.get("x_mm") is not None]
        footprint_ys = [float(item["y_mm"]) for item in footprints if item.get("y_mm") is not None]
        if footprint_xs and footprint_ys:
            center_x = sum(footprint_xs) / len(footprint_xs)
            center_y = sum(footprint_ys) / len(footprint_ys)
            span_x = max(
                30.0,
                max(float(item.get("width_mm", 2.0)) for item in footprints) + 10.0,
                max(footprint_xs) - min(footprint_xs) + 16.0,
            )
            span_y = max(
                24.0,
                max(float(item.get("height_mm", 2.0)) for item in footprints) + 10.0,
                max(footprint_ys) - min(footprint_ys) + 16.0,
            )
            x1 = max(board_x1, center_x - span_x / 2.0)
            x2 = min(board_x2, center_x + span_x / 2.0)
            y1 = max(board_y1, center_y - span_y / 2.0)
            y2 = min(board_y2, center_y + span_y / 2.0)
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
    for footprint in footprints:
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
        f"FP={len(footprints)}/{total_footprints} "
        f"TRACK={len(board['tracks'])} VIA={len(board['vias'])}"
    )
    if focused:
        header += f"  view=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"
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
    live = snapshot.get("board", {}).get("live_ipc", {})
    if live.get("status") == "connected" and live.get("semantic_match") is not True:
        return {
            "status": "blocked",
            "net": net_name,
            "reason": "live KiCad board differs from the saved board; save/reload before planning",
            "authority": live,
            "segments": [],
        }
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
    collisions = 0
    routing_methods: list[str] = []
    for start, end in edges:
        excluded_refs = {str(start["reference"]), str(end["reference"])}
        points, method = _orthogonal_route(
            start["at"],
            end["at"],
            board,
            clearance=clearance_mm,
            width_mm=width_mm,
            excluded_refs=excluded_refs,
            net_name=net_name,
        )
        routing_methods.append(method)
        collisions += _route_collision_score(
            points,
            board,
            clearance_mm,
            width_mm=width_mm,
            excluded_refs=excluded_refs,
            net_name=net_name,
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
            "Obstacle-aware orthogonal/MST proposal only; KiCad DRC remains authoritative.",
            "Review geometry before applying; no differential-pair or impedance solver is used.",
        ],
        "collision_score": collisions,
        "routing_methods": routing_methods,
    }


def critical_placement_report(
    snapshot: JsonRecord,
    constraints: Iterable[JsonRecord],
) -> JsonRecord:
    """Measure named-net pad distance across critical two-component placements."""
    footprints = {
        str(item.get("reference", "")): item
        for item in snapshot.get("board", {}).get("footprints", [])
    }
    findings: list[JsonRecord] = []
    for raw in constraints:
        reference_a = str(raw.get("reference_a", raw.get("a", "")))
        reference_b = str(raw.get("reference_b", raw.get("b", "")))
        max_distance = float(raw.get("max_pad_distance_mm", 3.0))
        nets = [str(net) for net in raw.get("nets", [])]
        first = footprints.get(reference_a)
        second = footprints.get(reference_b)
        measurements: list[JsonRecord] = []
        missing: list[str] = []
        if first is None:
            missing.append(reference_a)
        if second is None:
            missing.append(reference_b)
        if first is not None and second is not None:
            for net in nets:
                first_pads = [pad for pad in first.get("pads", []) if pad.get("net") == net]
                second_pads = [pad for pad in second.get("pads", []) if pad.get("net") == net]
                if not first_pads or not second_pads:
                    missing.append(f"{reference_a}<->{reference_b}:{net}")
                    continue
                first_pad, second_pad = min(
                    ((a_pad, b_pad) for a_pad in first_pads for b_pad in second_pads),
                    key=lambda pair: math.hypot(
                        float(pair[0]["at"][0]) - float(pair[1]["at"][0]),
                        float(pair[0]["at"][1]) - float(pair[1]["at"][1]),
                    ),
                )
                dx = float(second_pad["at"][0]) - float(first_pad["at"][0])
                dy = float(second_pad["at"][1]) - float(first_pad["at"][1])
                distance = math.hypot(dx, dy)
                measurements.append(
                    {
                        "net": net,
                        "distance_mm": round(distance, 4),
                        "max_distance_mm": max_distance,
                        "status": "pass" if distance <= max_distance else "fail",
                        "first": {
                            "reference": reference_a,
                            "pad": str(first_pad.get("number", "")),
                            "at_mm": first_pad["at"],
                        },
                        "second": {
                            "reference": reference_b,
                            "pad": str(second_pad.get("number", "")),
                            "at_mm": second_pad["at"],
                        },
                    }
                )
        failed = bool(missing) or not measurements or any(
            measurement["status"] == "fail" for measurement in measurements
        )
        findings.append(
            {
                "name": str(raw.get("name", f"{reference_a}<->{reference_b}")),
                "reference_a": reference_a,
                "reference_b": reference_b,
                "reason": str(raw.get("reason", "")),
                "status": "fail" if failed else "pass",
                "max_pad_distance_mm": max_distance,
                "measurements": measurements,
                "missing": missing,
            }
        )
    failing = sum(finding["status"] == "fail" for finding in findings)
    return {
        "schema_version": "1.0",
        "status": "fail" if failing else "pass",
        "summary": {
            "constraints": len(findings),
            "passing": len(findings) - failing,
            "failing": failing,
        },
        "findings": findings,
    }


def critical_placement_plan(
    snapshot: JsonRecord,
    constraints: Iterable[JsonRecord],
    *,
    grid_mm: float = 0.25,
    courtyard_margin_mm: float = 0.25,
    yield_references: Iterable[str] = (),
) -> JsonRecord:
    """Move declared member footprints near fixed hosts using named-net pad geometry."""
    if grid_mm <= 0.0:
        raise ValueError("grid_mm must be greater than zero")
    constraint_list = [dict(item) for item in constraints]
    before = critical_placement_report(snapshot, constraint_list)
    footprints = {
        str(item.get("reference", "")): item
        for item in snapshot.get("board", {}).get("footprints", [])
        if item.get("reference")
    }
    bounds_raw = snapshot.get("board", {}).get("bounds_mm")
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
        return {
            "schema_version": "1.0",
            "status": "blocked",
            "reason": "board has no rectangular Edge.Cuts bounds",
            "placements": [],
            "before": before,
        }
    board_bounds = tuple(float(value) for value in bounds_raw)
    movable_refs = {
        str(item.get("reference_b", item.get("b", "")))
        for item in constraint_list
    }
    yielded_refs = {str(reference) for reference in yield_references}
    occupied: list[tuple[str, tuple[float, float, float, float]]] = []
    for reference, footprint in footprints.items():
        if reference in movable_refs or reference in yielded_refs:
            continue
        occupied.append(
            (
                reference,
                _footprint_bounds_for_transform(
                    footprint,
                    float(footprint.get("x_mm") or 0.0),
                    float(footprint.get("y_mm") or 0.0),
                    float(footprint.get("rotation") or 0.0),
                    margin_mm=courtyard_margin_mm,
                ),
            )
        )

    placements: list[JsonRecord] = []
    unresolved: list[JsonRecord] = []
    for constraint in constraint_list:
        reference_a = str(constraint.get("reference_a", constraint.get("a", "")))
        reference_b = str(constraint.get("reference_b", constraint.get("b", "")))
        host = footprints.get(reference_a)
        member = footprints.get(reference_b)
        nets = [str(net) for net in constraint.get("nets", [])]
        max_distance = float(constraint.get("max_pad_distance_mm", 3.0))
        if host is None or member is None or not nets:
            unresolved.append(
                {
                    "reference": reference_b,
                    "host_reference": reference_a,
                    "reason": "host/member footprint or named-net constraints are missing",
                }
            )
            continue
        net_pad_pairs: list[tuple[str, JsonRecord, JsonRecord]] = []
        for net in nets:
            host_pads = [pad for pad in host.get("pads", []) if pad.get("net") == net]
            member_pads = [pad for pad in member.get("pads", []) if pad.get("net") == net]
            if not host_pads or not member_pads:
                continue
            host_pad, member_pad = min(
                ((a_pad, b_pad) for a_pad in host_pads for b_pad in member_pads),
                key=lambda pair: math.hypot(
                    float(pair[0]["at"][0]) - float(pair[1]["at"][0]),
                    float(pair[0]["at"][1]) - float(pair[1]["at"][1]),
                ),
            )
            net_pad_pairs.append((net, host_pad, member_pad))
        if len(net_pad_pairs) != len(nets):
            unresolved.append(
                {
                    "reference": reference_b,
                    "host_reference": reference_a,
                    "reason": "one or more declared nets lack pads on both footprints",
                }
            )
            continue

        candidates: list[tuple[tuple[float, float, float, float], JsonRecord]] = []
        blocker_hits: Counter[str] = Counter()
        search_radius = max(4.0, max_distance * 1.5)
        steps = int(math.ceil(search_radius / grid_mm))
        offsets = sorted(
            (
                (dx * grid_mm, dy * grid_mm)
                for dx in range(-steps, steps + 1)
                for dy in range(-steps, steps + 1)
            ),
            key=lambda offset: (
                abs(offset[0]) + abs(offset[1]),
                abs(offset[1]),
                abs(offset[0]),
                offset[1],
                offset[0],
            ),
        )
        allowed_rotations = tuple(
            float(value)
            for value in constraint.get("allowed_rotations", (0.0, 90.0, 180.0, 270.0))
        )
        for rotation in allowed_rotations:
            ideal_roots: list[tuple[float, float]] = []
            local_offsets: list[tuple[str, JsonRecord, float, float]] = []
            for net, host_pad, member_pad in net_pad_pairs:
                local_x, local_y = _pad_local_offset(member, member_pad)
                offset_x, offset_y = _rotate_local_offset(local_x, local_y, rotation)
                ideal_roots.append(
                    (
                        float(host_pad["at"][0]) - offset_x,
                        float(host_pad["at"][1]) - offset_y,
                    )
                )
                local_offsets.append((net, host_pad, offset_x, offset_y))
            ideal_x = round(
                (sum(root[0] for root in ideal_roots) / len(ideal_roots)) / grid_mm
            ) * grid_mm
            ideal_y = round(
                (sum(root[1] for root in ideal_roots) / len(ideal_roots)) / grid_mm
            ) * grid_mm
            for offset_x, offset_y in offsets:
                root_x = round(ideal_x + offset_x, 6)
                root_y = round(ideal_y + offset_y, 6)
                member_bounds = _footprint_bounds_for_transform(
                    member,
                    root_x,
                    root_y,
                    rotation,
                    margin_mm=courtyard_margin_mm,
                )
                if (
                    member_bounds[0] < board_bounds[0]
                    or member_bounds[1] < board_bounds[1]
                    or member_bounds[2] > board_bounds[2]
                    or member_bounds[3] > board_bounds[3]
                ):
                    continue
                blockers = [
                    reference
                    for reference, occupied_bounds in occupied
                    if _rectangles_overlap(member_bounds, occupied_bounds)
                ]
                if blockers:
                    blocker_hits.update(blockers)
                    continue
                measurements: list[JsonRecord] = []
                distances: list[float] = []
                for net, host_pad, pad_offset_x, pad_offset_y in local_offsets:
                    member_at = [root_x + pad_offset_x, root_y + pad_offset_y]
                    distance = math.hypot(
                        member_at[0] - float(host_pad["at"][0]),
                        member_at[1] - float(host_pad["at"][1]),
                    )
                    distances.append(distance)
                    measurements.append(
                        {
                            "net": net,
                            "distance_mm": round(distance, 4),
                            "host_pad": str(host_pad.get("number", "")),
                            "member_at_mm": [round(value, 4) for value in member_at],
                        }
                    )
                if any(distance > max_distance + 1e-6 for distance in distances):
                    continue
                movement = math.hypot(
                    root_x - float(member.get("x_mm") or 0.0),
                    root_y - float(member.get("y_mm") or 0.0),
                )
                score = (max(distances), sum(distances), movement, rotation)
                candidates.append(
                    (
                        score,
                        {
                            "reference": reference_b,
                            "host_reference": reference_a,
                            "from": [
                                float(member.get("x_mm") or 0.0),
                                float(member.get("y_mm") or 0.0),
                            ],
                            "to": [root_x, root_y],
                            "from_rotation": float(member.get("rotation") or 0.0),
                            "rotation": rotation,
                            "max_pad_distance_mm": max_distance,
                            "measurements": measurements,
                        },
                    )
                )
        if not candidates:
            unresolved.append(
                {
                    "reference": reference_b,
                    "host_reference": reference_a,
                    "reason": "no collision-free transform satisfies every named-net distance",
                    "blocking_footprints": [
                        {"reference": reference, "hits": hits}
                        for reference, hits in blocker_hits.most_common(12)
                    ],
                }
            )
            continue
        _score, selected = min(candidates, key=lambda candidate: candidate[0])
        placements.append(selected)
        occupied.append(
            (
                reference_b,
                _footprint_bounds_for_transform(
                    member,
                    float(selected["to"][0]),
                    float(selected["to"][1]),
                    float(selected["rotation"]),
                    margin_mm=courtyard_margin_mm,
                ),
            )
        )

    proposed_snapshot = copy.deepcopy(snapshot)
    proposed_by_ref = {
        str(item.get("reference", "")): item
        for item in proposed_snapshot.get("board", {}).get("footprints", [])
    }
    for placement in placements:
        footprint = proposed_by_ref[str(placement["reference"])]
        root_x, root_y = (float(value) for value in placement["to"])
        rotation = float(placement["rotation"])
        for pad in footprint.get("pads", []):
            local_x, local_y = _pad_local_offset(footprint, pad)
            pad_x, pad_y = _rotate_local_offset(local_x, local_y, rotation)
            pad["at"] = [round(root_x + pad_x, 4), round(root_y + pad_y, 4)]
        footprint["x_mm"] = root_x
        footprint["y_mm"] = root_y
        footprint["rotation"] = rotation
    after = critical_placement_report(proposed_snapshot, constraint_list)
    status = "planned" if not unresolved and after["status"] == "pass" else "blocked"
    return {
        "schema_version": "1.0",
        "status": status,
        "reason": (
            "all critical member footprints have collision-free named-net transforms"
            if status == "planned"
            else "one or more critical placement constraints remain unresolved"
        ),
        "grid_mm": grid_mm,
        "courtyard_margin_mm": courtyard_margin_mm,
        "yielded_references": sorted(yielded_refs),
        "placements": placements,
        "unresolved": unresolved,
        "before": before,
        "after": after,
    }


def _route_collision_score(
    points: list[list[float]] | tuple[list[float], ...],
    board: JsonRecord,
    clearance: float,
    *,
    width_mm: float = 0.0,
    excluded_refs: set[str] | None = None,
    net_name: str = "",
) -> int:
    if len(points) < 2:
        return 0
    score = 0
    for first, second in zip(points, points[1:], strict=False):
        sx1, sx2 = sorted((first[0], second[0]))
        sy1, sy2 = sorted((first[1], second[1]))
        for footprint in board["footprints"]:
            reference = str(footprint["reference"])
            if reference in (excluded_refs or set()):
                # The route must escape its endpoint footprints, but every
                # neighboring pad on those packages remains a copper obstacle.
                # Skipping the whole footprint can short a wide trace into an
                # adjacent exposed/ground pad even when a centerline looks clear.
                for pad in footprint.get("pads", []):
                    if str(pad.get("net", "")) == net_name:
                        continue
                    pad_at = pad.get("at", [])
                    pad_size = pad.get("size", [])
                    if len(pad_at) != 2 or len(pad_size) != 2:
                        continue
                    half_trace = width_mm / 2.0
                    half_width = float(pad_size[0]) / 2.0 + clearance + half_trace
                    half_height = float(pad_size[1]) / 2.0 + clearance + half_trace
                    if _segment_intersects_box(
                        first,
                        second,
                        left=float(pad_at[0]) - half_width,
                        top=float(pad_at[1]) - half_height,
                        right=float(pad_at[0]) + half_width,
                        bottom=float(pad_at[1]) + half_height,
                    ):
                        score += 1
                continue
            cx = float(footprint["x_mm"])
            cy = float(footprint["y_mm"])
            half_width = float(footprint["width_mm"]) / 2 + clearance + width_mm / 2.0
            half_height = float(footprint["height_mm"]) / 2 + clearance + width_mm / 2.0
            overlaps = (
                sx2 >= cx - half_width
                and sx1 <= cx + half_width
                and sy2 >= cy - half_height
                and sy1 <= cy + half_height
            )
            if overlaps:
                score += 1
        for track in board.get("tracks", []):
            if track.get("net") == net_name:
                continue
            if track.get("layer") not in {None, "F.Cu", "B.Cu"}:
                continue
            if _segments_intersect(first, second, track["start"], track["end"], clearance):
                score += 1
    return score


def _segment_intersects_box(
    first: list[float],
    second: list[float],
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    """Conservatively test one orthogonal segment against an axis-aligned box."""
    segment_left, segment_right = sorted((float(first[0]), float(second[0])))
    segment_top, segment_bottom = sorted((float(first[1]), float(second[1])))
    return not (
        segment_right < left
        or right < segment_left
        or segment_bottom < top
        or bottom < segment_top
    )


def _segments_intersect(
    first: list[float],
    second: list[float],
    other_first: list[float],
    other_second: list[float],
    clearance: float,
) -> bool:
    left, right = sorted((first[0], second[0]))
    top, bottom = sorted((first[1], second[1]))
    other_left, other_right = sorted((other_first[0], other_second[0]))
    other_top, other_bottom = sorted((other_first[1], other_second[1]))
    return not (
        right + clearance < other_left
        or other_right + clearance < left
        or bottom + clearance < other_top
        or other_bottom + clearance < top
    )


def _compress_grid_path(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 3:
        return points
    compressed = [points[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(points)):
        dx = points[index][0] - points[index - 1][0]
        dy = points[index][1] - points[index - 1][1]
        direction = (
            0 if dx == 0 else (1 if dx > 0 else -1),
            0 if dy == 0 else (1 if dy > 0 else -1),
        )
        if previous_direction is not None and direction != previous_direction:
            compressed.append(points[index - 1])
        previous_direction = direction
    compressed.append(points[-1])
    return compressed


def _orthogonal_route(
    start: list[float],
    end: list[float],
    board: JsonRecord,
    *,
    clearance: float,
    width_mm: float = 0.0,
    excluded_refs: set[str],
    net_name: str,
    grid_mm: float = 0.5,
) -> tuple[list[list[float]], str]:
    """Return a collision-free orthogonal path when the board geometry permits it."""
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)
    candidates = [([x1, y1], [x2, y1], [x2, y2]), ([x1, y1], [x1, y2], [x2, y2])]
    candidate_scores = [
        _route_collision_score(
            candidate,
            board,
            clearance,
            width_mm=width_mm,
            excluded_refs=excluded_refs,
            net_name=net_name,
        )
        for candidate in candidates
    ]
    best_index = min(range(len(candidates)), key=candidate_scores.__getitem__)
    if candidate_scores[best_index] == 0:
        return list(candidates[best_index]), "direct-manhattan"

    bounds = board.get("bounds_mm")
    if not bounds:
        return list(candidates[best_index]), "fallback-manhattan"
    left, top, right, bottom = map(float, bounds)
    columns = max(1, round((right - left) / grid_mm))
    rows = max(1, round((bottom - top) / grid_mm))

    def grid(point: list[float]) -> tuple[int, int]:
        return (
            max(0, min(columns, round((float(point[0]) - left) / grid_mm))),
            max(0, min(rows, round((float(point[1]) - top) / grid_mm))),
        )

    def world(node: tuple[int, int]) -> list[float]:
        return [round(left + node[0] * grid_mm, 4), round(top + node[1] * grid_mm, 4)]

    start_node, end_node = grid(start), grid(end)
    frontier: list[tuple[float, float, tuple[int, int]]] = []
    heappush(frontier, (0.0, 0.0, start_node))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost: dict[tuple[int, int], float] = {start_node: 0.0}
    while frontier:
        _priority, current_cost, current = heappop(frontier)
        if current == end_node:
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] <= columns and 0 <= neighbor[1] <= rows):
                continue
            segment = [world(current), world(neighbor)]
            if neighbor != end_node and _route_collision_score(
                segment,
                board,
                clearance,
                width_mm=width_mm,
                excluded_refs=excluded_refs,
                net_name=net_name,
            ):
                continue
            next_cost = current_cost + 1.0
            if next_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = next_cost
            heuristic = abs(neighbor[0] - end_node[0]) + abs(neighbor[1] - end_node[1])
            heappush(frontier, (next_cost + heuristic, next_cost, neighbor))
            came_from[neighbor] = current

    if end_node not in came_from:
        return list(candidates[best_index]), "fallback-manhattan"
    nodes = [end_node]
    while came_from[nodes[-1]] is not None:
        nodes.append(came_from[nodes[-1]])  # type: ignore[arg-type]
    nodes.reverse()
    path = [list(start)] + [world(node) for node in nodes[1:-1]] + [list(end)]
    return _compress_grid_path(path), "astar-grid"


def placement_plan(
    snapshot: JsonRecord,
    *,
    fixed_references: Iterable[str] = (),
    anchors: Iterable[JsonRecord] = (),
    cluster_regions: Iterable[JsonRecord] = (),
    keepout_regions: Iterable[list[float]] = (),
    proximity_pairs: Iterable[JsonRecord] = (),
    margin_mm: float = 3.0,
    iterations: int = 300,
    grid_mm: float = 0.5,
    seed: int = 42,
) -> JsonRecord:
    """Generate a deterministic connectivity-aware placement proposal without editing."""
    board = snapshot["board"]
    live = board.get("live_ipc", {})
    if live.get("status") == "connected" and live.get("semantic_match") is not True:
        return {
            "status": "blocked",
            "reason": "live KiCad board differs from the saved board; save/reload before planning",
            "authority": live,
            "placements": [],
        }
    bounds = board.get("bounds_mm")
    footprints = list(board.get("footprints", []))
    if not bounds:
        return {"status": "blocked", "reason": "board has no Edge.Cuts outline", "placements": []}
    if not footprints:
        return {"status": "blocked", "reason": "board has no footprints", "placements": []}
    left, top, right, bottom = map(float, bounds)
    width, height = right - left, bottom - top
    anchor_list = list(anchors)
    anchor_by_ref = {str(anchor["reference"]): anchor for anchor in anchor_list}
    fixed = set(fixed_references) | set(anchor_by_ref)
    rotation_by_ref = {
        str(item["reference"]): float(item.get("rotation", 0.0) or 0.0) for item in footprints
    }
    for reference, anchor in anchor_by_ref.items():
        if reference not in rotation_by_ref:
            return {
                "status": "blocked",
                "reason": f"anchor reference '{reference}' is not present on the board",
                "placements": [],
            }
        if anchor.get("rotation") is not None:
            rotation_by_ref[reference] = float(anchor["rotation"])

    def rotated_geometry(
        item: JsonRecord, rotation_deg: float
    ) -> tuple[float, float, float, float]:
        min_x = float(
            item.get(
                "bbox_min_x_mm",
                -float(item["width_mm"]) / 2.0,
            )
        )
        min_y = float(
            item.get(
                "bbox_min_y_mm",
                -float(item["height_mm"]) / 2.0,
            )
        )
        max_x = float(
            item.get(
                "bbox_max_x_mm",
                float(item["width_mm"]) / 2.0,
            )
        )
        max_y = float(
            item.get(
                "bbox_max_y_mm",
                float(item["height_mm"]) / 2.0,
            )
        )
        angle = math.radians(rotation_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        corners = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        ]
        rotated = [(x * cosine + y * sine, -x * sine + y * cosine) for x, y in corners]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        rotated_min_x, rotated_max_x = min(xs), max(xs)
        rotated_min_y, rotated_max_y = min(ys), max(ys)
        return (
            (rotated_min_x + rotated_max_x) / 2.0,
            (rotated_min_y + rotated_max_y) / 2.0,
            rotated_max_x - rotated_min_x,
            rotated_max_y - rotated_min_y,
        )

    components: list[PlacementComponent] = []
    for item in footprints:
        if item.get("x_mm") is None or item.get("y_mm") is None:
            continue
        reference = str(item["reference"])
        origin_dx, origin_dy, geometry_width, geometry_height = rotated_geometry(
            item, rotation_by_ref[reference]
        )
        component_margin_mm = float(
            anchor_by_ref.get(reference, {}).get("margin_mm", margin_mm)
        )
        components.append(
            PlacementComponent(
                ref=reference,
                x=float(item["x_mm"]) - left + origin_dx,
                y=float(item["y_mm"]) - top + origin_dy,
                w=geometry_width + component_margin_mm,
                h=geometry_height + component_margin_mm,
                fixed=reference in fixed,
                origin_dx=origin_dx,
                origin_dy=origin_dy,
            )
        )
    saved_components = [PlacementComponent(**component.__dict__) for component in components]
    components_by_ref = {component.ref: component for component in components}
    resolved_anchors: list[JsonRecord] = []
    for anchor in anchor_list:
        reference = str(anchor["reference"])
        component = components_by_ref.get(reference)
        if component is None:
            return {
                "status": "blocked",
                "reason": f"anchor reference '{reference}' has no resolved placement",
                "placements": [],
            }
        if anchor.get("x_mm") is not None and anchor.get("y_mm") is not None:
            edge = "absolute"
            offset: float | None = None
            component.x = float(anchor["x_mm"]) - left + component.origin_dx
            component.y = float(anchor["y_mm"]) - top + component.origin_dy
            if (
                component.x - (component.w / 2.0) < 0.0
                or component.x + (component.w / 2.0) > width
                or component.y - (component.h / 2.0) < 0.0
                or component.y + (component.h / 2.0) > height
            ):
                return {
                    "status": "blocked",
                    "reason": f"anchor '{reference}' absolute position is outside the board",
                    "placements": [],
                }
        else:
            edge = str(anchor.get("edge", "")).lower()
            offset = float(anchor["offset_mm"])
        if edge in {"top", "bottom"}:
            edge_offset = 0.0 if offset is None else offset
            if not 0.0 <= edge_offset <= width:
                return {
                    "status": "blocked",
                    "reason": f"anchor '{reference}' offset is outside board width",
                    "placements": [],
                }
            component.x = edge_offset
            component.y = component.h / 2.0 if edge == "top" else height - component.h / 2.0
        elif edge in {"left", "right"}:
            edge_offset = 0.0 if offset is None else offset
            if not 0.0 <= edge_offset <= height:
                return {
                    "status": "blocked",
                    "reason": f"anchor '{reference}' offset is outside board height",
                    "placements": [],
                }
            component.x = component.w / 2.0 if edge == "left" else width - component.w / 2.0
            component.y = edge_offset
        elif edge != "absolute":
            return {
                "status": "blocked",
                "reason": f"anchor '{reference}' edge must be top, right, bottom, or left",
                "placements": [],
            }
        resolved_anchors.append(
            {
                "reference": reference,
                "edge": edge,
                "offset_mm": offset,
                "rotation": rotation_by_ref[reference],
                "margin_mm": float(anchor.get("margin_mm", margin_mm)),
                "position_mm": [
                    round(component.x - component.origin_dx + left, 4),
                    round(component.y - component.origin_dy + top, 4),
                ],
            }
        )
    anchored_components = [PlacementComponent(**component.__dict__) for component in components]
    sheet_by_ref = {
        str(item["reference"]): str(item.get("sheet", ""))
        for item in snapshot["schematic"].get("components", [])
    }
    component_regions: dict[str, tuple[float, float, float, float]] = {}
    resolved_clusters: list[JsonRecord] = []
    for cluster in cluster_regions:
        sheet = str(cluster["sheet"])
        x1 = float(cluster["x1_mm"])
        y1 = float(cluster["y1_mm"])
        x2 = float(cluster["x2_mm"])
        y2 = float(cluster["y2_mm"])
        if not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
            return {
                "status": "blocked",
                "reason": f"cluster region for '{sheet}' is outside the board",
                "placements": [],
            }
        local_region = (x1 - left, y1 - top, x2 - left, y2 - top)
        references = sorted(
            reference
            for reference, reference_sheet in sheet_by_ref.items()
            if reference_sheet == sheet and reference in components_by_ref
        )
        for reference in references:
            if reference not in fixed:
                component_regions[reference] = local_region
        resolved_clusters.append(
            {
                "sheet": sheet,
                "bounds_mm": [x1, y1, x2, y2],
                "references": references,
                "movable_references": [
                    reference for reference in references if reference not in fixed
                ],
            }
        )

    # Seed movable cluster members on a deterministic region grid. The force
    # solver and legalizer refine this seed but cannot leak members into another
    # subsystem's physical region.
    members_by_region: dict[tuple[float, float, float, float], list[PlacementComponent]] = {}
    for component in components:
        region = component_regions.get(component.ref)
        if region is not None:
            members_by_region.setdefault(region, []).append(component)
    for region, members in members_by_region.items():
        region_left, region_top, region_right, region_bottom = region
        region_width = region_right - region_left
        region_height = region_bottom - region_top
        ordered_members = sorted(members, key=lambda item: (-(item.w * item.h), item.ref))
        columns = max(
            1,
            math.ceil(math.sqrt(len(ordered_members) * region_width / region_height)),
        )
        rows = max(1, math.ceil(len(ordered_members) / columns))
        for index, component in enumerate(ordered_members):
            column = index % columns
            row = index // columns
            component.x = region_left + ((column + 0.5) * region_width / columns)
            component.y = region_top + ((row + 0.5) * region_height / rows)
    known_refs = {component.ref for component in components}
    nets: list[PlacementNet] = []
    for net in snapshot["schematic"]["nets"]:
        refs = sorted({str(node["reference"]) for node in net["nodes"]} & known_refs)
        if len(refs) < 2 or net.get("unconnected"):
            continue
        weight = round(2.0 / math.sqrt(len(refs)), 4)
        if any(pattern.search(str(net["name"])) for pattern in CRITICAL_NET_PATTERNS):
            weight *= 0.35
        nets.append(PlacementNet(name=str(net["name"]), refs=refs, weight=weight))
    resolved_proximity_pairs: list[JsonRecord] = []
    for pair in proximity_pairs:
        host_ref = str(pair.get("ic_ref", ""))
        max_distance_mm = float(pair.get("max_distance_mm", 3.0))
        for cap_ref_value in pair.get("cap_refs", []):
            cap_ref = str(cap_ref_value)
            if host_ref not in known_refs or cap_ref not in known_refs:
                continue
            # A dedicated two-node spring prevents a global rail such as GND,
            # +3V3, or LTE_3V8 from pulling a decoupler toward the rail centroid.
            # Repulsion and the legalizer still enforce physical separation.
            weight = max(80.0, min(200.0, 800.0 / max(max_distance_mm, 1.0)))
            nets.append(
                PlacementNet(
                    name=f"@proximity:{host_ref}:{cap_ref}",
                    refs=[host_ref, cap_ref],
                    weight=weight,
                )
            )
            resolved_proximity_pairs.append(
                {
                    "host_reference": host_ref,
                    "member_reference": cap_ref,
                    "max_distance_mm": max_distance_mm,
                    "spring_weight": round(weight, 4),
                }
            )
    local_keepouts = [
        (
            float(region[0]) - left,
            float(region[1]) - top,
            float(region[2]) - left,
            float(region[3]) - top,
        )
        for region in keepout_regions
    ]
    stats: dict[str, object] = {}
    placement_config = ForceDirectedConfig(
        iterations=iterations,
        board_w=width,
        board_h=height,
        seed=seed,
        grid_mm=grid_mm,
        keepout_regions=local_keepouts,
        component_regions=component_regions,
    )
    proposed = force_directed_placement(
        components,
        nets,
        placement_config,
        stats=stats,
    )
    proposed = legalize_placement(
        proposed,
        placement_config,
        stats=stats,
    )
    original = {component.ref: component for component in saved_components}
    placements = [
        {
            "reference": component.ref,
            "from": [
                round(
                    original[component.ref].x - original[component.ref].origin_dx + left,
                    4,
                ),
                round(
                    original[component.ref].y - original[component.ref].origin_dy + top,
                    4,
                ),
            ],
            "to": [
                round(component.x - component.origin_dx + left, 4),
                round(component.y - component.origin_dy + top, 4),
            ],
            "fixed": component.fixed,
            "anchored": component.ref in anchor_by_ref,
            "from_rotation": next(
                float(item.get("rotation", 0.0) or 0.0)
                for item in footprints
                if str(item["reference"]) == component.ref
            ),
            "rotation": rotation_by_ref.get(component.ref, 0.0),
        }
        for component in proposed
    ]

    def weighted_hpwl(layout: Iterable[PlacementComponent]) -> float:
        positions = {component.ref: (component.x, component.y) for component in layout}
        total = 0.0
        for net in nets:
            points = [positions[reference] for reference in net.refs if reference in positions]
            if len(points) < 2:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            total += net.weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
        return total

    hpwl_before = weighted_hpwl(saved_components)
    baseline_stats: dict[str, object] = {}
    legalized_baseline = legalize_placement(
        anchored_components,
        ForceDirectedConfig(
            board_w=width,
            board_h=height,
            seed=seed,
            grid_mm=grid_mm,
            keepout_regions=local_keepouts,
        ),
        stats=baseline_stats,
    )
    hpwl_baseline = weighted_hpwl(legalized_baseline)
    hpwl_after = weighted_hpwl(proposed)
    hpwl_original_delta_pct = (
        ((hpwl_after - hpwl_before) / hpwl_before) * 100.0 if hpwl_before > 0.0 else 0.0
    )
    hpwl_delta_pct = (
        ((hpwl_after - hpwl_baseline) / hpwl_baseline) * 100.0 if hpwl_baseline > 0.0 else 0.0
    )
    unresolved = list(stats.get("legalized_unresolved", []))
    baseline_unresolved = list(baseline_stats.get("legalized_unresolved", []))
    new_unresolved = sorted(set(unresolved) - set(baseline_unresolved))
    quality_pass = hpwl_delta_pct <= 25.0 and not new_unresolved
    return {
        "status": "planned",
        "board_bounds_mm": list(bounds),
        "margin_mm": margin_mm,
        "grid_mm": grid_mm,
        "seed": seed,
        "iterations": iterations,
        "iterations_run": stats.get("iterations_run"),
        "converged": stats.get("converged"),
        "nets_considered": len(nets),
        "proximity_pairs": resolved_proximity_pairs,
        "legalized_moved": stats.get("legalized_moved"),
        "legalized_unresolved": unresolved,
        "anchors": resolved_anchors,
        "clusters": resolved_clusters,
        "clustered_references": len(component_regions),
        "weighted_hpwl_before_mm": round(hpwl_before, 3),
        "weighted_hpwl_legalized_baseline_mm": round(hpwl_baseline, 3),
        "weighted_hpwl_after_mm": round(hpwl_after, 3),
        "wirelength_delta_pct": round(hpwl_delta_pct, 2),
        "wirelength_vs_original_delta_pct": round(hpwl_original_delta_pct, 2),
        "baseline_legalized_unresolved": baseline_unresolved,
        "new_legalized_unresolved": new_unresolved,
        "quality_gate": {
            "status": "pass" if quality_pass else "fail",
            "max_wirelength_increase_pct": 25.0,
            "reason": (
                "weighted HPWL is within the allowed regression threshold"
                if quality_pass
                else (
                    "placement legalization introduced unresolved footprints: "
                    + ", ".join(new_unresolved)
                    if new_unresolved
                    else (
                        "weighted HPWL would regress by more than 25% versus "
                        "the legalized baseline"
                    )
                )
            ),
        },
        "placements": placements,
        "notes": [
            "Connectivity-aware proposal only; connector, RF, thermal, and mechanical "
            "constraints remain authoritative.",
            "Review and lock mechanical anchors before applying any coordinates.",
        ],
    }
