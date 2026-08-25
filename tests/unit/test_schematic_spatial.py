"""Focused tests for the bounded schematic spatial map."""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_mcp.schematic_spatial import parse_schematic_geometry, schematic_spatial_map
from kicad_mcp.shell_cli import build_parser

SHEET = """(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 2.54 0))
      (symbol "R_0_1" (rectangle (start -1 -1) (end 1 1)))
    )
  )
  (symbol
    (lib_id "MCU:Demo")
    (at 40 30 0)
    (unit 1)
    (property "Reference" "U1" (at 40 25 0))
    (property "Value" "Controller" (at 40 35 0))
  )
  (symbol
    (lib_id "Device:R")
    (at 70 30 90)
    (unit 1)
    (property "Reference" "R1" (at 70 27 90))
    (property "Value" "10k" (at 70 33 90))
  )
  (wire (pts (xy 40 30) (xy 55 30) (xy 70 30)))
  (global_label "GPIO_OUT" (shape output) (at 55 30 0))
  (label "ENABLE" (at 42.5 35 0))
)
"""


def _snapshot(tmp_path: Path) -> dict[str, object]:
    return {
        "project": {"name": "demo", "directory": str(tmp_path)},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Power/", "file": "power.kicad_sch"}],
            "components": [
                {
                    "reference": "U1",
                    "value": "Controller",
                    "footprint": "QFN",
                    "sheet": "/Power/",
                },
                {
                    "reference": "R1",
                    "value": "10k",
                    "footprint": "R_0603",
                    "sheet": "/Power/",
                },
            ],
            "nets": [
                {
                    "name": "GPIO_OUT",
                    "nodes": [
                        {"reference": "U1", "pin": "1"},
                        {"reference": "R1", "pin": "1"},
                    ],
                }
            ],
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }


def test_parser_reads_only_placed_symbols_and_root_geometry() -> None:
    geometry = parse_schematic_geometry(SHEET)

    assert [(item.reference, item.x_mm, item.y_mm) for item in geometry.symbols] == [
        ("R1", 70.0, 30.0),
        ("U1", 40.0, 30.0),
    ]
    assert [(item.name, item.kind) for item in geometry.labels] == [
        ("ENABLE", "label"),
        ("GPIO_OUT", "global_label"),
    ]
    assert geometry.wires[0].points == ((40.0, 30.0), (55.0, 30.0), (70.0, 30.0))


def test_spatial_map_shows_positions_zones_and_relationships(tmp_path: Path) -> None:
    (tmp_path / "power.kicad_sch").write_text(SHEET, encoding="utf-8")

    rendered = schematic_spatial_map(
        _snapshot(tmp_path),
        tmp_path,
        sheet="Power",
        width=72,
        height=16,
    )

    assert "SCHEMATIC SPATIAL MAP" in rendered
    assert "[U1]" in rendered
    assert "GPIO_OUT" in rendered
    assert "near=[ENABLE@5.6mm" in rendered
    assert "(  40.00,  30.00)" in rendered
    assert len(max(rendered.splitlines(), key=len)) < 190


def test_spatial_zoom_centers_on_reference_and_bounds_canvas(tmp_path: Path) -> None:
    (tmp_path / "power.kicad_sch").write_text(SHEET, encoding="utf-8")

    rendered = schematic_spatial_map(
        _snapshot(tmp_path),
        tmp_path,
        zoom=3,
        width=10,
        height=500,
        center_reference="U1",
    )

    assert "zoom=3" in rendered
    border_lines = [line for line in rendered.splitlines() if line.startswith(("┌", "│", "└"))]
    assert len(border_lines) == 80
    assert all(len(line) == 48 for line in border_lines)


def test_spatial_center_rejects_unknown_reference(tmp_path: Path) -> None:
    (tmp_path / "power.kicad_sch").write_text(SHEET, encoding="utf-8")

    with pytest.raises(ValueError, match="center reference was not found"):
        schematic_spatial_map(
            _snapshot(tmp_path),
            tmp_path,
            zoom=1,
            center_reference="U404",
        )


def test_map_cli_keeps_semantic_default_and_accepts_spatial_controls() -> None:
    parser = build_parser()

    original = parser.parse_args(["map", "--zoom", "2"])
    spatial = parser.parse_args(
        ["map", "--view", "spatial", "--zoom", "3", "--height", "40", "--center-ref", "U1"]
    )

    assert original.view == "semantic"
    assert original.width == 100
    assert spatial.view == "spatial"
    assert spatial.height == 40
    assert spatial.center_ref == "U1"
