"""Tests for the planning-only schematic label compactor."""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_mcp.schematic_rewire_plan import (
    format_label_compaction_plan,
    plan_label_compaction,
)
from kicad_mcp.shell_cli import build_parser


def _sheet(*, crossing: bool = False, hierarchical: bool = False) -> str:
    label_kind = "hierarchical_label" if hierarchical else "global_label"
    shape = "(shape bidirectional) "
    crossing_wire = "  (wire (pts (xy 36 10) (xy 36 30)))\n" if crossing else ""
    return f"""(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "MCU:Demo"
      (rectangle (start -2 -2) (end 2 2))
    )
  )
  (symbol
    (lib_id "MCU:Demo")
    (at 36 40 0)
    (unit 1)
    (property "Reference" "U1" (at 36 36 0))
    (property "Value" "Controller" (at 36 44 0))
  )
  (wire (pts (xy 25 20) (xy 30 20)))
  (wire (pts (xy 42 20) (xy 47 20)))
{crossing_wire}  ({label_kind} "SIG" {shape}(at 30 20 0) (uuid "label-left"))
  ({label_kind} "SIG" {shape}(at 42 20 0) (uuid "label-right"))
)
"""


def _snapshot(tmp_path: Path) -> dict[str, object]:
    return {
        "project": {"name": "demo", "directory": str(tmp_path)},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Logic/", "file": "logic.kicad_sch"}],
            "components": [
                {
                    "reference": "U1",
                    "value": "Controller",
                    "footprint": "QFN",
                    "sheet": "/Logic/",
                }
            ],
            "nets": [
                {
                    "name": "/Logic/SIG",
                    "nodes": [{"reference": "U1", "pin": "1"}],
                }
            ],
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }


def test_planner_emits_exact_removal_and_manhattan_wire(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    plan = plan_label_compaction(_snapshot(tmp_path), tmp_path, reference="U1")

    assert plan["status"] == "planned"
    assert plan["read_only"] is True
    cluster = plan["clusters"][0]
    assert cluster["cluster_index"] == 1
    assert cluster["retain"]["id"] == "label-left"
    assert cluster["proposals"][0]["remove"]["id"] == "label-right"
    assert cluster["proposals"][0]["selected_route"] == {
        "points_mm": [[42.0, 20.0], [30.0, 20.0]],
        "segments": 1,
        "length_mm": 12.0,
    }
    assert plan["prediction"]["label_count_delta"] == -1
    assert "future apply must preserve" in " ".join(plan["invariants"])


def test_planner_refuses_ambiguous_existing_wire_crossing(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(crossing=True), encoding="utf-8")

    plan = plan_label_compaction(_snapshot(tmp_path), tmp_path, reference="U1")

    assert plan["status"] == "blocked"
    proposal = plan["clusters"][0]["proposals"][0]
    assert proposal["status"] == "refused"
    assert {reason["code"] for reason in proposal["refusals"]} == {"unknown_wire_intersection"}
    assert proposal["candidate_routes"][0]["refusals"][0]["at_mm"] == [36.0, 20.0]


def test_planner_never_compacts_hierarchical_interface_labels(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(
        _sheet(hierarchical=True),
        encoding="utf-8",
    )

    plan = plan_label_compaction(_snapshot(tmp_path), tmp_path, reference="U1")

    assert plan["status"] == "blocked"
    assert plan["clusters"][0]["refusals"][0]["code"] == "hierarchical_interface_label"
    assert plan["prediction"]["label_count_delta"] == 0


def test_text_plan_is_concise_and_explicitly_read_only(tmp_path: Path) -> None:
    (tmp_path / "logic.kicad_sch").write_text(_sheet(), encoding="utf-8")

    rendered = format_label_compaction_plan(
        plan_label_compaction(_snapshot(tmp_path), tmp_path, reference="U1")
    )

    assert "LABEL COMPACTION PLAN" in rendered
    assert "REMOVE label-right" in rendered
    assert "WIRE [[42.0, 20.0], [30.0, 20.0]]" in rendered
    assert "APPLY disabled" in rendered


def test_planner_validates_radius_and_cli_is_read_only() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["plan-labels", "--ref", "U4", "--sheet", "LTE", "--radius", "22", "--format", "json"]
    )

    assert args.command == "plan-labels"
    assert args.reference == "U4"
    assert args.radius_mm == 22.0
    assert not hasattr(args, "apply")
    with pytest.raises(ValueError, match="between 5 and 100"):
        plan_label_compaction({}, ".", reference="U1", radius_mm=101)
