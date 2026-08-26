"""Headless project inspection, ASCII maps, and conservative route planning.

The normal MCP backend remains the authority for KiCad mutations.  This module
adds a fast, file-oriented planning surface that works with KiCad closed and
emits ordinary Python/JSON values for shell pipelines.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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
        {key: item[key] for key in ("start", "end", "width_mm", "layer", "net_code")}
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
    nets = sorted(
        (int(code), name)
        for code, name in re.findall(rf"\(net\s+(\d+)\s+{STRING_PATTERN}\)", normalized)
    )
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
    live_board = _live_board_probe(project, board_content)
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
            "authority": "file-backed-kicad-pcb",
            "live_ipc": live_board,
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
    keys = sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in findings)
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
            excluded_refs=excluded_refs,
            net_name=net_name,
        )
        routing_methods.append(method)
        collisions += _route_collision_score(
            points,
            board,
            clearance_mm,
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


def _route_collision_score(
    points: list[list[float]] | tuple[list[float], ...],
    board: JsonRecord,
    clearance: float,
    *,
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
            if str(footprint["reference"]) in (excluded_refs or set()):
                continue
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
        for track in board.get("tracks", []):
            if track.get("net") == net_name:
                continue
            if track.get("layer") not in {None, "F.Cu", "B.Cu"}:
                continue
            if _segments_intersect(first, second, track["start"], track["end"], clearance):
                score += 1
    return score


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
    keepout_regions: Iterable[list[float]] = (),
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
    fixed = set(fixed_references)
    components = [
        PlacementComponent(
            ref=str(item["reference"]),
            x=float(item["x_mm"]) - left,
            y=float(item["y_mm"]) - top,
            w=float(item["width_mm"]) + margin_mm,
            h=float(item["height_mm"]) + margin_mm,
            fixed=str(item["reference"]) in fixed,
        )
        for item in footprints
        if item.get("x_mm") is not None and item.get("y_mm") is not None
    ]
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
    proposed = force_directed_placement(
        components,
        nets,
        ForceDirectedConfig(
            iterations=iterations,
            board_w=width,
            board_h=height,
            seed=seed,
            grid_mm=grid_mm,
            keepout_regions=local_keepouts,
        ),
        stats=stats,
    )
    proposed = legalize_placement(
        proposed,
        ForceDirectedConfig(
            board_w=width,
            board_h=height,
            seed=seed,
            grid_mm=grid_mm,
            keepout_regions=local_keepouts,
        ),
        stats=stats,
    )
    original = {component.ref: component for component in components}
    rotation_by_ref = {
        str(item["reference"]): float(item.get("rotation", 0.0) or 0.0)
        for item in footprints
    }
    placements = [
        {
            "reference": component.ref,
            "from": [
                round(original[component.ref].x + left, 4),
                round(original[component.ref].y + top, 4),
            ],
            "to": [round(component.x + left, 4), round(component.y + top, 4)],
            "fixed": component.fixed,
            "rotation": rotation_by_ref.get(component.ref, 0.0),
        }
        for component in proposed
    ]
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
        "legalized_moved": stats.get("legalized_moved"),
        "legalized_unresolved": stats.get("legalized_unresolved"),
        "placements": placements,
        "notes": [
            "Connectivity-aware proposal only; connector, RF, thermal, and mechanical "
            "constraints remain authoritative.",
            "Review and lock mechanical anchors before applying any coordinates.",
        ],
    }
