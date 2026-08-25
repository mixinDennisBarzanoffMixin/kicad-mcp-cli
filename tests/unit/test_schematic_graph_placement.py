"""Tests for generic read-only schematic graph placement."""

from __future__ import annotations

from pathlib import Path

from kicad_mcp.schematic_graph_placement import (
    format_schematic_graph_placement,
    plan_fresh_schematic_layout,
    plan_schematic_graph_placement,
)
from kicad_mcp.shell_cli import build_parser

REFERENCES = ("J1", "U1", "R1", "U2", "C1")


def _sheet() -> str:
    symbols = "\n".join(
        f'''  (symbol
    (lib_id "Device:Generic")
    (at {40 + index * 0.5} {40 + index * 0.5} 0)
    (unit 1)
    (property "Reference" "{reference}" (at 40 36 0))
    (property "Value" "Part-{reference}" (at 40 44 0))
  )'''
        for index, reference in enumerate(REFERENCES)
    )
    return f"""(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "Device:Generic"
      (rectangle (start -3 -2) (end 3 2))
    )
  )
{symbols}
  (wire (pts (xy 20 20) (xy 60 20)))
  (wire (pts (xy 40 10) (xy 40 30)))
  (global_label "GND" (shape input) (at 20 20 0) (uuid "gnd-label"))
  (label "SIG" (at 60 20 0) (uuid "sig-label"))
)
"""


def _node(reference: str, pin_type: str) -> dict[str, str]:
    return {"reference": reference, "pin": "1", "function": "P", "type": pin_type}


def _snapshot(tmp_path: Path, *, feedback: bool = False) -> dict[str, object]:
    nets = [
        {"name": "VIN", "nodes": [_node("J1", "passive"), _node("U1", "input")]},
        {"name": "SIG", "nodes": [_node("U1", "output"), _node("R1", "passive")]},
        {"name": "OUT", "nodes": [_node("R1", "passive"), _node("U2", "input")]},
        {
            "name": "GND",
            "nodes": [_node(reference, "passive") for reference in REFERENCES],
        },
    ]
    if feedback:
        nets.append(
            {
                "name": "FEEDBACK",
                "nodes": [_node("U2", "output"), _node("U1", "input")],
            }
        )
    return {
        "project": {"name": "demo", "directory": str(tmp_path)},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Logic/", "file": "logic.kicad_sch"}],
            "components": [
                {
                    "reference": reference,
                    "value": f"Part-{reference}",
                    "footprint": "Package:Generic",
                    "sheet": "/Logic/",
                }
                for reference in REFERENCES
            ],
            "nets": nets,
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }


def test_graph_planner_infers_ranks_rail_clusters_and_costs(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    plan = plan_schematic_graph_placement(_snapshot(tmp_path), tmp_path, sheet="Logic")

    ranks = {item["reference"]: item["rank"] for item in plan["placements"]}
    assert ranks["J1"] < ranks["U1"] < ranks["R1"] < ranks["U2"]
    assert plan["graph"]["rail_nets"] == [
        {"name": "GND", "fanout": 5, "reason": "power-name or high-fanout shared-net rail"}
    ]
    assert any(
        anchor["reference"] == "J1" and "connector" in anchor["roles"] for anchor in plan["anchors"]
    )
    assert plan["functional_clusters"]
    assert plan["cost"]["new"]["symbol_overlaps"] <= plan["cost"]["old"]["symbol_overlaps"]
    assert plan["cost"]["old"]["label_count"] == plan["cost"]["new"]["label_count"] == 2
    assert all(not topology["applicable"] for topology in plan["wire_topology"])
    assert plan["read_only"] is True
    assert plan["acceptance"]["accepted"] is True
    assert plan["ranked_candidates"][0]["eligible"] is True


def test_feedback_cycle_is_collapsed_and_reported(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    plan = plan_schematic_graph_placement(
        _snapshot(tmp_path, feedback=True), tmp_path, sheet="Logic"
    )

    assert plan["graph"]["feedback_groups"]
    constraint = next(
        item for item in plan["constraints"] if item["code"] == "directed_feedback_cycles_collapsed"
    )
    assert set(constraint["groups"][0]) >= {"U1", "R1", "U2"}


def test_fixed_reference_keeps_exact_old_center(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    plan = plan_schematic_graph_placement(
        _snapshot(tmp_path),
        tmp_path,
        sheet="Logic",
        fixed_references=["U1"],
    )

    u1 = next(item for item in plan["placements"] if item["reference"] == "U1")
    assert u1["fixed"] is True
    assert u1["proposed_mm"] == u1["old_mm"]


def test_text_output_and_cli_are_explicitly_dry_run(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")
    rendered = format_schematic_graph_placement(
        plan_schematic_graph_placement(_snapshot(tmp_path), tmp_path, sheet="Logic")
    )
    args = build_parser().parse_args(
        [
            "plan-schematic",
            "--sheet",
            "Logic",
            "--fix",
            "J1",
            "--candidates",
            "4",
            "--format",
            "json",
        ]
    )

    assert "SCHEMATIC RAILWAY PLAN" in rendered
    assert "COST overlap" in rendered
    assert "cannot be applied" in rendered
    assert args.fixed_references == ["J1"]
    assert args.candidate_count == 4
    assert not hasattr(args, "apply")


def test_power_backbone_is_monotonic_or_hard_blocked_without_usable_moves(
    tmp_path: Path,
) -> None:
    references = ("U4", "U13", "RSH_LTE", "JP_LTE_PWR", "CIN", "COUT")
    symbols = "\n".join(
        f'''  (symbol
    (lib_id "Device:Generic")
    (at {x} 60 0)
    (unit 1)
    (property "Reference" "{reference}" (at {x} 56 0))
    (property "Value" "{reference}" (at {x} 64 0))
  )'''
        for reference, x in zip(references, (60, 110, 160, 200, 75, 220), strict=True)
    )
    text = f"""(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "Device:Generic" (rectangle (start -3 -2) (end 3 2)))
  )
{symbols}
  (wire (pts (xy 50 60) (xy 230 60)))
  (global_label "GND" (shape input) (at 50 60 0) (uuid "gnd"))
)
"""
    (tmp_path / "power.kicad_sch").write_text(text, encoding="utf-8")
    nodes = {
        "SYS_RAW": [_node("CIN", "passive"), _node("U4", "power_in")],
        "LTE_RAW": [
            _node("U4", "power_out"),
            _node("U13", "power_in"),
            _node("COUT", "passive"),
        ],
        "SHUNT_P": [_node("U13", "power_out"), _node("RSH_LTE", "passive")],
        "SHUNT_N": [_node("RSH_LTE", "passive"), _node("JP_LTE_PWR", "passive")],
        "LTE_3V8": [_node("JP_LTE_PWR", "passive"), _node("COUT", "passive")],
        "GND": [_node("CIN", "passive"), _node("COUT", "passive")],
    }
    snapshot = {
        "project": {"name": "power", "directory": str(tmp_path)},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Power/", "file": "power.kicad_sch"}],
            "components": [
                {
                    "reference": reference,
                    "value": reference,
                    "footprint": "Package:Generic",
                    "sheet": "/Power/",
                }
                for reference in references
            ],
            "nets": [{"name": name, "nodes": net_nodes} for name, net_nodes in nodes.items()],
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }

    plan = plan_schematic_graph_placement(snapshot, tmp_path, sheet="Power")
    ranks = {item["reference"]: item["rank"] for item in plan["placements"]}

    assert ranks["U4"] < ranks["U13"] < ranks["RSH_LTE"] < ranks["JP_LTE_PWR"]
    for metric in (
        "symbol_overlaps",
        "wire_crossings",
        "total_manhattan_wire_length_mm",
        "compactness_area_mm2",
    ):
        assert plan["cost"]["new"][metric] <= plan["cost"]["old"][metric]


def test_optimizer_is_deterministic_and_identity_is_always_available(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    first = plan_schematic_graph_placement(
        _snapshot(tmp_path), tmp_path, sheet="Logic", candidate_count=6
    )
    second = plan_schematic_graph_placement(
        _snapshot(tmp_path), tmp_path, sheet="Logic", candidate_count=6
    )

    assert first == second
    identity = next(item for item in first["ranked_candidates"] if item["name"] == "identity")
    assert identity["eligible"] is True
    assert first["acceptance"]["regressions"] == []


def test_fresh_layout_api_needs_no_saved_coordinates_and_is_deterministic() -> None:
    symbols = [
        {"reference": "U1", "value": "driver", "width_mm": 12, "height_mm": 10},
        {"reference": "R1", "value": "1k"},
        {"reference": "U2", "value": "sink", "width_mm": 12, "height_mm": 10},
    ]
    nets = [
        {"name": "SIG", "nodes": [_node("U1", "output"), _node("R1", "passive")]},
        {"name": "OUT", "nodes": [_node("R1", "passive"), _node("U2", "input")]},
    ]

    first = plan_fresh_schematic_layout(symbols, nets)
    second = plan_fresh_schematic_layout(symbols, nets)

    assert first == second
    assert first["coordinate_authority"] == "generated_without_saved_coordinates"
    ranks = {item["reference"]: item["rank"] for item in first["placements"]}
    assert ranks["U1"] < ranks["R1"] < ranks["U2"]
    assert all("old_mm" not in item for item in first["placements"])
