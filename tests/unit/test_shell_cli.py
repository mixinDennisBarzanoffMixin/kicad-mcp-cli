"""Tests for the shell-first KiCad interface."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mcp import types as mcp_types

from kicad_mcp.compact_server import server as compact_server
from kicad_mcp.deep_inspection import (
    _source_integrity_evidence,
    ascii_map,
    connectivity_proof,
    filter_snapshot,
    placement_plan,
    route_plan,
)
from kicad_mcp.shell_cli import (
    _erc_finding_keys,
    _resolve_edit_schematic,
    _rewire_tool_calls,
    _schematic_manifest,
    parse_call_arguments,
    result_envelope,
    run_native_board_transaction,
)


def test_rewire_tool_calls_translate_only_physical_operations() -> None:
    plan = {
        "sheet": {"file": "folder/power.kicad_sch"},
        "nets": [
            {
                "net": "/Power/SIG",
                "status": "planned",
                "selected_operations": [
                    {
                        "op": "add_wire",
                        "start_mm": [10.16, 20.32],
                        "end_mm": [12.7, 20.32],
                    },
                    {"op": "ensure_junction", "at_mm": [12.7, 20.32]},
                    {"op": "retain_label", "anchor_mm": [10.16, 20.32]},
                    {
                        "op": "remove_label",
                        "net": "/Power/SIG",
                        "anchor_mm": [15.24, 20.32],
                    },
                ],
            }
        ],
    }

    calls, selected = _rewire_tool_calls(plan, ["SIG"])

    assert selected == ["/Power/SIG"]
    assert calls == [
        (
            "sch_add_wire",
            {
                "x1_mm": 10.16,
                "y1_mm": 20.32,
                "x2_mm": 12.7,
                "y2_mm": 20.32,
                "snap_to_grid": False,
                "sheet_file": "power.kicad_sch",
            },
        ),
        (
            "sch_delete_label",
            {
                "name": "SIG",
                "x_mm": 15.24,
                "y_mm": 20.32,
                "sheet_file": "power.kicad_sch",
            },
        ),
    ]


def test_rewire_tool_calls_refuse_ambiguous_local_net_name() -> None:
    plan = {
        "sheet": {"file": "power.kicad_sch"},
        "nets": [
            {"net": "/A/SIG", "status": "planned", "selected_operations": []},
            {"net": "/B/SIG", "status": "planned", "selected_operations": []},
        ],
    }

    try:
        _rewire_tool_calls(plan, ["SIG"])
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous local net selector should fail")


def test_parse_call_arguments_merges_json_and_set_values() -> None:
    args = argparse.Namespace(
        args_json='{"layer":"F.Cu","count":1}',
        set_values=["count=2", "enabled=true", "label=USB"],
    )

    assert parse_call_arguments(args) == {
        "layer": "F.Cu",
        "count": 2,
        "enabled": True,
        "label": "USB",
    }


def test_schematic_manifest_ignores_generated_history(tmp_path: Path) -> None:
    source = tmp_path / "power.kicad_sch"
    source.write_text("source", encoding="utf-8")
    history = tmp_path / ".history"
    history.mkdir()
    (history / "power.kicad_sch").write_text("backup", encoding="utf-8")

    assert list(_schematic_manifest(tmp_path)) == [Path("power.kicad_sch")]


def test_resolve_edit_schematic_targets_one_child_sheet(tmp_path: Path) -> None:
    (tmp_path / "demo.kicad_sch").write_text("top", encoding="utf-8")
    child = tmp_path / "03_LTE_3V8_Power.kicad_sch"
    child.write_text("child", encoding="utf-8")

    assert _resolve_edit_schematic(tmp_path, "LTE_3V8_Power") == child
    assert _resolve_edit_schematic(tmp_path, "") is None


def test_resolve_edit_schematic_rejects_ambiguous_filter(tmp_path: Path) -> None:
    (tmp_path / "Power_A.kicad_sch").write_text("a", encoding="utf-8")
    (tmp_path / "Power_B.kicad_sch").write_text("b", encoding="utf-8")

    try:
        _resolve_edit_schematic(tmp_path, "Power")
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous sheet filter should fail")


def test_erc_finding_keys_only_tracks_errors() -> None:
    report = {
        "checks": {
            "erc": {
                "findings": [
                    {"severity": "warning", "type": "warn"},
                    {"severity": "error", "type": "broken"},
                ]
            }
        }
    }

    assert _erc_finding_keys(report) == {'{"severity":"error","type":"broken"}'}


async def test_native_board_transaction_drops_commit_on_new_drc_finding(
    tmp_path: Path, monkeypatch
) -> None:
    import kicad_mcp.shell_cli as shell_cli

    effects: list[str] = []

    class FakeBoard:
        contents = iter(["before", "staged"])

        def get_as_string(self) -> str:
            return next(self.contents)

        def begin_commit(self) -> None:
            effects.append("begin")

        def drop_commit(self) -> None:
            effects.append("drop")

        def push_commit(self) -> None:
            effects.append("push")

        def save(self) -> None:
            effects.append("save")

    monkeypatch.setattr(
        shell_cli,
        "authority_report",
        lambda _root: {"policy": {"board_mutation_allowed": True}},
    )
    monkeypatch.setattr(shell_cli, "get_board", FakeBoard)

    async def invoke(_args, _tool, _arguments):
        return {"ok": True, "tool": "pcb_move_footprint"}

    monkeypatch.setattr(shell_cli, "invoke_backend_tool", invoke)
    drc_results = iter(
        [
            {"finding_keys": [], "summary": {}},
            {
                "finding_keys": ['{"kind":"violation","type":"clearance"}'],
                "summary": {},
            },
        ]
    )
    monkeypatch.setattr(
        shell_cli,
        "board_drc_evidence",
        lambda _root, board_content: next(drc_results),
    )
    args = argparse.Namespace(
        project_dir=str(tmp_path),
        artifacts=str(tmp_path / "evidence"),
        profile="full",
        mode="write",
    )

    report = await run_native_board_transaction(
        args,
        [("pcb_move_footprint", {"reference": "U1", "x_mm": 1, "y_mm": 2})],
        label="placement",
    )

    assert report["status"] == "rejected"
    assert report["committed"] is False
    assert effects == ["begin", "drop"]


def test_result_envelope_preserves_structured_and_text_content() -> None:
    result = mcp_types.CallToolResult(
        isError=False,
        structuredContent={"result": {"version": "10.0"}},
        content=[mcp_types.TextContent(type="text", text=json.dumps({"version": "10.0"}))],
    )

    payload = result_envelope("kicad_get_version", result)

    assert payload["ok"] is True
    assert payload["tool"] == "kicad_get_version"
    assert payload["result"] == {"result": {"version": "10.0"}}
    assert payload["content"] == [{"type": "text", "text": '{"version": "10.0"}'}]


async def test_compact_server_exposes_only_discover_run_and_grep() -> None:
    tool_names = {tool.name for tool in await compact_server.list_tools()}

    assert tool_names == {"kicad_catalog", "kicad_run", "kicad_grep"}


def _snapshot() -> dict[str, object]:
    return {
        "project": {"name": "demo", "directory": "demo"},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Power/", "file": "power.kicad_sch"}],
            "components": [
                {"reference": "U1", "value": "MCU", "footprint": "QFN", "sheet": "/Power/"},
                {"reference": "R1", "value": "10k", "footprint": "R_0603", "sheet": "/Power/"},
            ],
            "nets": [
                {
                    "name": "GPIO",
                    "unconnected": False,
                    "nodes": [
                        {"reference": "U1", "pin": "1", "function": "IO", "type": "bidirectional"},
                        {"reference": "R1", "pin": "1", "function": "", "type": "passive"},
                    ],
                }
            ],
        },
        "board": {
            "bounds_mm": [0, 0, 20, 10],
            "footprints": [
                {
                    "reference": "U1",
                    "x_mm": 5,
                    "y_mm": 5,
                    "width_mm": 2,
                    "height_mm": 2,
                    "pads": [{"number": "1", "at": [5, 5], "net": "GPIO"}],
                },
                {
                    "reference": "R1",
                    "x_mm": 15,
                    "y_mm": 5,
                    "width_mm": 2,
                    "height_mm": 1,
                    "pads": [{"number": "1", "at": [15, 5], "net": "GPIO"}],
                },
            ],
            "tracks": [],
            "vias": [],
        },
    }


def test_deep_filter_and_ascii_zoom() -> None:
    snapshot = _snapshot()
    filtered = filter_snapshot(snapshot, reference="U1")

    assert [item["reference"] for item in filtered["schematic"]["components"]] == ["U1"]
    assert "COMPONENT MAP" in ascii_map(snapshot, zoom=1)
    assert "GPIO" in ascii_map(snapshot, zoom=2)
    assert "PCB MAP" in ascii_map(snapshot, zoom=3)


def test_connectivity_proof_expands_pin_net_and_peer_evidence() -> None:
    proof = connectivity_proof(_snapshot(), reference="U1")

    assert proof["status"] == "pass"
    assert proof["summary"] == {
        "components": 1,
        "pins": 1,
        "nets": 1,
        "unconnected_pins": 0,
        "intentional_no_connects": 0,
        "singleton_nets": 0,
        "missing_footprints": 0,
        "duplicate_pin_assignments": 0,
        "board_net_mismatches": 0,
    }
    assert proof["pins"][0]["net"] == "GPIO"
    assert proof["pins"][0]["peers"] == [
        {"reference": "R1", "pin": "1", "function": "", "type": "passive"}
    ]
    assert proof["pins"][0]["board_pad_net"] == "GPIO"


def test_connectivity_proof_does_not_leak_findings_from_filtered_components() -> None:
    snapshot = _snapshot()
    snapshot["schematic"]["nets"].append(
        {
            "name": "unconnected-(U2-Pad1)",
            "unconnected": True,
            "nodes": [{"reference": "U2", "pin": "1", "function": "NC", "type": "no_connect"}],
        }
    )

    proof = connectivity_proof(snapshot, reference="U1")

    assert proof["status"] == "pass"
    assert proof["findings"]["singleton_nets"] == []


def test_source_integrity_evidence_is_sheet_scoped(tmp_path: Path) -> None:
    top = tmp_path / "demo.kicad_pro"
    top.write_text("{}", encoding="utf-8")
    power = tmp_path / "power.kicad_sch"
    power.write_text(
        '(kicad_sch (uuid "00000000-0000-0000-0000-000000000001"))\n',
        encoding="utf-8",
    )
    ignored = tmp_path / "ignored.kicad_sch"
    ignored.write_text("(kicad_sch", encoding="utf-8")
    snapshot = _snapshot()
    snapshot["schematic"]["sheets"].append(
        {"number": 2, "name": "/Ignored/", "file": "ignored.kicad_sch"}
    )

    evidence = _source_integrity_evidence(top, snapshot, sheet="Power")

    assert evidence["status"] == "pass"
    assert evidence["files"] == [str(power)]


def test_route_plan_is_dry_and_refuses_critical_nets() -> None:
    snapshot = _snapshot()

    plan = route_plan(snapshot, "GPIO")
    refused = route_plan(snapshot, "+3V3")

    assert plan["status"] == "planned"
    assert len(plan["segments"]) == 1
    assert refused["status"] == "refused"


def test_route_and_placement_block_when_live_board_differs_from_disk() -> None:
    snapshot = _snapshot()
    snapshot["board"]["live_ipc"] = {"status": "connected", "semantic_match": False}

    assert route_plan(snapshot, "GPIO")["status"] == "blocked"
    assert placement_plan(snapshot)["status"] == "blocked"


def test_route_plan_ignores_its_endpoint_footprints() -> None:
    plan = route_plan(_snapshot(), "GPIO")

    assert plan["collision_score"] == 0
    assert plan["routing_methods"] == ["direct-manhattan"]


def test_route_plan_uses_astar_around_footprint_obstacle() -> None:
    snapshot = copy.deepcopy(_snapshot())
    snapshot["board"]["footprints"].append(
        {
            "reference": "U2",
            "x_mm": 10,
            "y_mm": 5,
            "width_mm": 3,
            "height_mm": 3,
            "pads": [],
        }
    )

    plan = route_plan(snapshot, "GPIO", clearance_mm=0.5)

    assert plan["collision_score"] == 0
    assert plan["routing_methods"] == ["astar-grid"]
    assert len(plan["segments"]) >= 3


def test_placement_plan_is_deterministic_and_holds_fixed_references() -> None:
    first = placement_plan(_snapshot(), fixed_references=["U1"], iterations=20)
    second = placement_plan(_snapshot(), fixed_references=["U1"], iterations=20)

    assert first == second
    u1 = next(item for item in first["placements"] if item["reference"] == "U1")
    assert u1["fixed"] is True
    assert u1["from"] == u1["to"]


def test_shell_source_does_not_use_shell_execution() -> None:
    source = Path(__file__).parents[2] / "src" / "kicad_mcp" / "deep_inspection.py"
    text = source.read_text()
    assert "shell=True" not in text
