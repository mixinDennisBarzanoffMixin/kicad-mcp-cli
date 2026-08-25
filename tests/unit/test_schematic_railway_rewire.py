"""Tests for the read-only exact-pin railway rewiring planner."""

from __future__ import annotations

import json
from pathlib import Path

from kicad_mcp.schematic_railway_rewire import (
    format_railway_rewire_plan,
    plan_railway_rewire,
)
from kicad_mcp.shell_cli import build_parser, main


def _sheet(*, unrelated_crossing: bool = False, local_labels: bool = True) -> str:
    crossing = (
        """
  (wire (pts (xy 15 8) (xy 15 12)))
  (label "OTHER" (at 15 8 0))
"""
        if unrelated_crossing
        else ""
    )
    sig_labels = (
        """
  (label "SIG" (at 12 10 0))
  (label "SIG" (at 18 10 0))
"""
        if local_labels
        else ""
    )
    return f"""(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "Demo:Node"
      (symbol "Node_1_1"
        (rectangle (start -1 -1) (end 1 1))
        (pin passive line
          (at 2 0 180)
          (length 1)
          (name "IO")
          (number "1")
        )
      )
    )
  )
  (symbol
    (lib_id "Demo:Node")
    (at 10 10 0)
    (unit 1)
    (property "Reference" "A" (at 10 7 0))
    (property "Value" "Node" (at 10 13 0))
  )
  (symbol
    (lib_id "Demo:Node")
    (at 20 10 180)
    (unit 1)
    (property "Reference" "B" (at 20 7 0))
    (property "Value" "Node" (at 20 13 0))
  )
  (symbol
    (lib_id "Demo:Node")
    (at 13 8 0)
    (unit 1)
    (property "Reference" "X" (at 13 5 0))
    (property "Value" "Other" (at 13 11 0))
  )
{sig_labels}
{crossing})
"""


def _snapshot(tmp_path: Path, *, shared: bool = False, unresolved: bool = False) -> dict:
    sig_nodes = [
        {"reference": "A", "pin": "1"},
        {"reference": "B", "pin": "99" if unresolved else "1"},
    ]
    if shared:
        sig_nodes.append({"reference": "X", "pin": "1"})
    return {
        "project": {"name": "rail-demo", "directory": str(tmp_path)},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Logic/", "file": "logic.kicad_sch"}],
            "components": [
                {"reference": reference, "value": "Node", "sheet": "/Logic/"}
                for reference in ("A", "B", "X")
            ],
            "nets": [
                {"name": "/Logic/SIG", "nodes": sig_nodes},
                {
                    "name": "/Logic/OTHER",
                    "nodes": [{"reference": "X", "pin": "1"}],
                },
            ],
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }


def test_planner_resolves_exact_tips_and_emits_direct_wire_without_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(), encoding="utf-8")
    before = source.read_bytes()

    plan = plan_railway_rewire(
        source,
        _snapshot(tmp_path),
        sheet="Logic",
        cluster_refs=["B", "A"],
    )

    assert source.read_bytes() == before
    assert plan["read_only"] is True
    assert plan["apply_supported"] is False
    assert plan["source"]["unchanged"] is True
    assert plan["cluster_refs"] == ["A", "B"]
    sig = next(net for net in plan["nets"] if net["net"] == "/Logic/SIG")
    assert sig["classification"] == "local_intra_cluster"
    assert sig["status"] == "planned"
    assert [(pin["reference"], pin["anchor_mm"]) for pin in sig["cluster_nodes"]] == [
        ("A", [12.0, 10.0]),
        ("B", [18.0, 10.0]),
    ]
    wires = [item for item in sig["selected_operations"] if item["op"] == "add_wire"]
    assert wires == [
        {
            "op": "add_wire",
            "net": "/Logic/SIG",
            "start_mm": [12.0, 10.0],
            "end_mm": [18.0, 10.0],
        }
    ]
    assert len([item for item in sig["labels"] if item["op"] == "retain_label"]) == 1
    assert len([item for item in sig["labels"] if item["op"] == "remove_label"]) == 1
    retained_name_ops = [
        item for item in sig["labels"] if item["op"] in {"retain_label", "add_label"}
    ]
    assert len(retained_name_ops) == 1
    assert retained_name_ops[0]["net"] == "/Logic/SIG"
    fingerprint = plan["connectivity_fingerprint_expectation"]
    assert fingerprint["before"] == fingerprint["expected_after"]
    assert plan["position_fingerprint_expectation"]["must_match"] is True


def test_planner_accepts_project_directory_as_schematic_location(tmp_path: Path) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(), encoding="utf-8")

    plan = plan_railway_rewire(
        tmp_path,
        _snapshot(tmp_path),
        sheet="Logic",
        cluster_refs=["A", "B"],
    )

    assert plan["source"]["path"] == str(source.resolve())
    assert plan["source"]["unchanged"] is True


def test_planner_adds_one_safe_name_label_when_no_existing_label_is_attached(
    tmp_path: Path,
) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(local_labels=False), encoding="utf-8")

    plan = plan_railway_rewire(
        source,
        _snapshot(tmp_path),
        cluster_refs=["A", "B"],
    )

    sig = next(net for net in plan["nets"] if net["net"] == "/Logic/SIG")
    name_ops = [item for item in sig["labels"] if item["op"] == "add_label"]
    assert len(name_ops) == 1
    assert name_ops[0]["net"] == "/Logic/SIG"
    assert name_ops[0]["name"] == "SIG"
    assert name_ops[0]["kind"] == "local"
    anchor = name_ops[0]["anchor_mm"]
    assert any(
        min(wire["start_mm"][0], wire["end_mm"][0])
        <= anchor[0]
        <= max(wire["start_mm"][0], wire["end_mm"][0])
        and min(wire["start_mm"][1], wire["end_mm"][1])
        <= anchor[1]
        <= max(wire["start_mm"][1], wire["end_mm"][1])
        for wire in sig["selected_operations"]
        if wire["op"] == "add_wire"
    )


def test_planner_retains_labels_for_shared_sheet_rail(tmp_path: Path) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(), encoding="utf-8")

    plan = plan_railway_rewire(
        source,
        _snapshot(tmp_path, shared=True),
        cluster_refs=["A", "B"],
    )

    sig = next(net for net in plan["nets"] if net["net"] == "/Logic/SIG")
    assert sig["classification"] == "shared_sheet_rail"
    assert sig["status"] == "retained"
    assert {item["op"] for item in sig["labels"]} == {"retain_label"}
    assert not any(
        item["op"] in {"add_wire", "remove_label"} and item["net"] == "/Logic/SIG"
        for item in plan["operations"]
    )


def test_planner_refuses_net_with_any_unresolved_cluster_pin(tmp_path: Path) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(), encoding="utf-8")

    plan = plan_railway_rewire(
        source,
        _snapshot(tmp_path, unresolved=True),
        cluster_refs=["A", "B"],
    )

    sig = next(net for net in plan["nets"] if net["net"] == "/Logic/SIG")
    assert sig["status"] == "refused"
    assert sig["refusals"] == [
        {
            "code": "unresolved_exact_pin",
            "reference": "B",
            "pin": "99",
            "reason": "pin_number_absent_from_cached_symbol",
        }
    ]
    assert not any(item.get("net") == "/Logic/SIG" for item in plan["operations"])


def test_crossing_candidate_is_refused_and_only_clear_detour_can_be_selected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(unrelated_crossing=True), encoding="utf-8")

    plan = plan_railway_rewire(
        source,
        _snapshot(tmp_path),
        cluster_refs=["A", "B"],
    )

    sig = next(net for net in plan["nets"] if net["net"] == "/Logic/SIG")
    assert sig["status"] == "planned"
    refusal_codes = {
        refusal["code"]
        for candidate in sig["candidate_routes"]
        for refusal in candidate["refusals"]
    }
    assert "unrelated_net_crossing" in refusal_codes
    assert all(
        operation.get("start_mm") != [12.0, 10.0] or operation.get("end_mm") != [18.0, 10.0]
        for operation in sig["selected_operations"]
        if operation["op"] == "add_wire"
    )


def test_terminal_format_is_explicitly_non_applying(tmp_path: Path) -> None:
    source = tmp_path / "logic.kicad_sch"
    source.write_text(_sheet(), encoding="utf-8")

    rendered = format_railway_rewire_plan(
        plan_railway_rewire(source, _snapshot(tmp_path), cluster_refs=["A", "B"])
    )

    assert "RAILWAY REWIRE" in rendered
    assert "WIRE [12.0, 10.0] -> [18.0, 10.0]" in rendered
    assert "NO APPLY" in rendered


def test_plan_rewire_cli_accepts_repeatable_refs_and_has_no_apply_mode() -> None:
    parser = build_parser()
    selected = parser.parse_args(
        [
            "plan-rewire",
            "--sheet",
            "LTE",
            "--ref",
            "C15",
            "--ref",
            "R11",
            "--format",
            "json",
        ]
    )
    whole_sheet = parser.parse_args(["plan-rewire", "--sheet", "LTE"])

    assert selected.command == "plan-rewire"
    assert selected.references == ["C15", "R11"]
    assert selected.format == "json"
    assert whole_sheet.references == []
    assert not hasattr(selected, "apply")


def test_plan_rewire_cli_json_is_pipe_friendly_and_uses_project_directory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import kicad_mcp.shell_cli as shell_cli

    calls: dict[str, object] = {}
    snapshot = {"project": {"name": "demo"}}
    report = {
        "schema_version": "1.0",
        "status": "planned",
        "read_only": True,
        "operations": [{"op": "retain_label", "net": "SIG"}],
    }

    def fake_snapshot(project_root):
        calls["snapshot_root"] = project_root
        return snapshot

    def fake_plan(project_root, received_snapshot, *, sheet, cluster_refs):
        calls.update(
            {
                "planner_root": project_root,
                "snapshot": received_snapshot,
                "sheet": sheet,
                "cluster_refs": cluster_refs,
            }
        )
        return report

    monkeypatch.setattr(shell_cli, "project_snapshot", fake_snapshot)
    monkeypatch.setattr(shell_cli, "plan_railway_rewire", fake_plan)

    main(
        [
            "-C",
            str(tmp_path),
            "plan-rewire",
            "--sheet",
            "Logic",
            "--ref",
            "A",
            "--ref",
            "B",
            "--format",
            "json",
        ]
    )

    assert json.loads(capsys.readouterr().out) == report
    assert calls == {
        "snapshot_root": tmp_path.resolve(),
        "planner_root": tmp_path.resolve(),
        "snapshot": snapshot,
        "sheet": "Logic",
        "cluster_refs": ["A", "B"],
    }


def test_plan_rewire_cli_omitted_refs_passes_whole_sheet_scope(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import kicad_mcp.shell_cli as shell_cli

    received: list[object] = []
    monkeypatch.setattr(shell_cli, "project_snapshot", lambda _root: {"schematic": {}})

    def fake_plan(_root, _snapshot, *, sheet, cluster_refs):
        received.extend((sheet, cluster_refs))
        return {"status": "noop", "nets": []}

    monkeypatch.setattr(shell_cli, "plan_railway_rewire", fake_plan)

    main(["-C", str(tmp_path), "plan-rewire", "--sheet", "Logic", "--format", "json"])

    assert received == ["Logic", None]
    assert json.loads(capsys.readouterr().out) == {"status": "noop", "nets": []}
