"""Tests for the shell-first KiCad interface."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest
from kipy.geometry import Angle, Vector2
from mcp import types as mcp_types

from kicad_mcp.compact_server import server as compact_server
from kicad_mcp.deep_inspection import (
    _drc_finding_key,
    _polygons_overlap,
    _source_integrity_evidence,
    ascii_map,
    connectivity_proof,
    critical_placement_plan,
    critical_placement_report,
    filter_snapshot,
    placement_plan,
    power_loop_report,
    route_plan,
)
from kicad_mcp.shell_cli import (
    _apply_footprint_batch_to_board_content,
    _demote_reference_fields_to_fab,
    _drc_regression_details,
    _emit_records,
    _erc_finding_keys,
    _native_move_footprints_batch,
    _placement_batch_requires_guarded_file,
    _render_candidate_track,
    _resolve_edit_schematic,
    _rewire_tool_calls,
    _schematic_manifest,
    _stackup_layers_from_spec,
    _verify_rigid_footprint_children,
    _verify_staged_footprint_batch,
    file_records,
    parse_call_arguments,
    result_envelope,
    run_native_board_transaction,
)
from kicad_mcp.tools.board_file import _courtyard_polygons_from_block


def test_silk_cleanup_demotes_reference_without_moving_footprint_root() -> None:
    board = """(kicad_pcb
      (footprint "C_0603"
        (layer "F.Cu")
        (at 12.5 34.5 90)
        (property "Reference" "C1"
          (at 0 -1.5 0)
          (layer "F.SilkS")
          (effects (font (size 1 1))))
        (property "Value" "100nF"
          (at 0 1.5 0)
          (layer "F.Fab")
          (effects (font (size 1 1))))))"""

    cleaned = _demote_reference_fields_to_fab(board, ["C1"])

    assert cleaned.count('(layer "F.SilkS")') == 0
    assert cleaned.count('(layer "F.Fab")') == 2
    assert "(at 12.5 34.5 90)" in cleaned


def test_concave_courtyard_is_preserved_as_one_polygon() -> None:
    block = """(footprint "RF_Module"
      (fp_line (start -24 -27.75) (end -24 -6.75) (layer "F.CrtYd"))
      (fp_line (start -24 -6.75) (end -9.75 -6.75) (layer "F.CrtYd"))
      (fp_line (start -9.75 13.45) (end -9.75 -6.75) (layer "F.CrtYd"))
      (fp_line (start -9.75 13.45) (end 9.75 13.45) (layer "F.CrtYd"))
      (fp_line (start 9.75 -6.75) (end 9.75 13.45) (layer "F.CrtYd"))
      (fp_line (start 9.75 -6.75) (end 24 -6.75) (layer "F.CrtYd"))
      (fp_line (start 24 -27.75) (end -24 -27.75) (layer "F.CrtYd"))
      (fp_line (start 24 -6.75) (end 24 -27.75) (layer "F.CrtYd")))"""

    polygons = _courtyard_polygons_from_block(block)

    assert len(polygons) == 1
    assert len(polygons[0]) == 8
    assert [-24.0, -27.75] in polygons[0]
    assert [-9.75, 13.45] in polygons[0]


def test_concave_polygon_collision_does_not_use_one_full_bounding_box() -> None:
    tee = [
        (-24.0, -27.75),
        (-24.0, -6.75),
        (-9.75, -6.75),
        (-9.75, 13.45),
        (9.75, 13.45),
        (9.75, -6.75),
        (24.0, -6.75),
        (24.0, -27.75),
    ]
    open_side = [(-11.5, -5.5), (-10.0, -5.5), (-10.0, -3.0), (-11.5, -3.0)]
    body_overlap = [(-10.0, -5.5), (-8.0, -5.5), (-8.0, -3.0), (-10.0, -3.0)]

    assert _polygons_overlap(tee, open_side) is False
    assert _polygons_overlap(tee, body_overlap) is True


def test_drc_regression_ignores_item_order_and_unconnected_pair_churn() -> None:
    physical = {
        "kind": "violation",
        "type": "clearance",
        "items": [{"uuid": "b"}, {"uuid": "a"}],
    }
    reordered = {
        **physical,
        "items": [
            {**child, "pos": {"x": 100.0, "y": 200.0}}
            for child in reversed(physical["items"])
        ],
    }
    old_unconnected = {
        "kind": "unconnected",
        "type": "unconnected_items",
        "items": [{"uuid": "old-a"}, {"uuid": "old-b"}],
    }
    new_unconnected = {
        "kind": "unconnected",
        "type": "unconnected_items",
        "items": [{"uuid": "new-a"}, {"uuid": "new-b"}],
    }
    before = {
        "finding_keys": [_drc_finding_key(physical), _drc_finding_key(old_unconnected)],
        "findings": [physical, old_unconnected],
        "summary": {"violations": 1, "unconnected_items": 1},
    }
    staged = {
        "finding_keys": [_drc_finding_key(reordered), _drc_finding_key(new_unconnected)],
        "findings": [reordered, new_unconnected],
        "summary": {"violations": 1, "unconnected_items": 1},
    }

    details = _drc_regression_details(before, staged)

    assert details["new_physical_findings"] == []
    assert details["unconnected_increase"] == 0
    assert details["regressed"] is False


def test_native_move_footprints_batch_updates_once_and_preserves_rotation() -> None:
    class Text:
        value = "U1"

    class Reference:
        text = Text()

    class Footprint:
        reference_field = Reference()
        position = Vector2.from_xy_mm(1.0, 2.0)
        angle = Angle.from_degrees(90.0)

    footprint = Footprint()

    class Board:
        updates = 0

        def get_footprints(self) -> list[object]:
            return [footprint]

        def update_items(self, items: list[object]) -> None:
            assert items == [footprint]
            self.updates += 1

    board = Board()
    result = _native_move_footprints_batch(
        board,
        {"placements": [{"reference": "U1", "x_mm": 10.0, "y_mm": 12.0, "rotation_deg": 90.0}]},
    )

    assert result["ok"] is True
    assert result["result"] == {"moved": 1}
    assert board.updates == 1
    assert footprint.position == Vector2.from_xy_mm(10.0, 12.0)


def test_native_move_footprints_batch_rejects_non_rigid_rotation_change() -> None:
    class Text:
        value = "U1"

    class Reference:
        text = Text()

    class Footprint:
        reference_field = Reference()
        position = Vector2.from_xy_mm(1.0, 2.0)
        orientation = Angle.from_degrees(0.0)

    class Board:
        def get_footprints(self) -> list[object]:
            return [Footprint()]

        def update_items(self, _items: list[object]) -> None:
            raise AssertionError("rotation-changing batch must not update the board")

    result = _native_move_footprints_batch(
        Board(),
        {"placements": [{"reference": "U1", "x_mm": 10, "y_mm": 12, "rotation_deg": 90}]},
    )

    assert result["ok"] is False
    assert "not a rigid child transform" in str(result["error"])


def test_staged_footprint_batch_verifies_serialized_position_and_rotation() -> None:
    content = """(kicad_pcb
      (footprint "Test"
        (layer "F.Cu")
        (at 10 12 90)
        (property "Reference" "U1" (at 0 0 0) (layer "F.SilkS"))
        (fp_rect (start -1 -1) (end 1 1) (stroke (width 0.05) (type solid))
          (fill no) (layer "F.CrtYd"))
      )
    )"""

    _verify_staged_footprint_batch(
        content,
        {"placements": [{"reference": "U1", "x_mm": 10, "y_mm": 12, "rotation_deg": 90}]},
    )


def test_offline_footprint_batch_changes_only_requested_root_transform() -> None:
    content = """(kicad_pcb
      (footprint "Test"
        (layer "F.Cu")
        (at 1 2 0)
        (property "Reference" "U1" (at 0 0 0) (layer "F.SilkS"))
        (fp_rect (start -1 -1) (end 1 1) (stroke (width 0.05) (type solid))
          (fill no) (layer "F.CrtYd"))
      )
    )"""

    candidate = _apply_footprint_batch_to_board_content(
        content,
        {"placements": [{"reference": "U1", "x_mm": 10, "y_mm": 12, "rotation_deg": 90}]},
    )

    assert "(at 10.0000 12.0000 90.0000)" in candidate
    assert '(property "Reference" "U1" (at 0 0 0)' in candidate


def test_rigid_footprint_verification_rejects_embedded_zone_drift() -> None:
    expected = """(kicad_pcb
      (footprint "RF_Module"
        (layer "F.Cu")
        (at 10 12 0)
        (property "Reference" "U1" (at 0 0 0) (layer "F.SilkS"))
        (zone (layers "F.Cu") (polygon (pts (xy -2 -4) (xy 2 -4))))
      )
    )"""
    observed = expected.replace("(at 10 12 0)", "(at 20 22 0)").replace("(xy -2 -4)", "(xy 8 6)")

    with pytest.raises(RuntimeError, match="changed child geometry"):
        _verify_rigid_footprint_children(
            expected,
            observed,
            {"placements": [{"reference": "U1", "x_mm": 20, "y_mm": 22}]},
        )


def test_embedded_zone_placement_selects_guarded_file_transaction() -> None:
    board = """(kicad_pcb
      (footprint "RF_Module"
        (layer "F.Cu")
        (at 10 12)
        (property "Reference" "U1")
        (zone (layers "F.Cu") (polygon (pts (xy -2 -4) (xy 2 -4))))
      )
    )"""
    operations = [
        (
            "_native_move_footprints_batch",
            {"placements": [{"reference": "U1", "x_mm": 20, "y_mm": 22}]},
        )
    ]

    assert _placement_batch_requires_guarded_file(board, operations) is True


def test_names_output_uses_paths_for_file_manifest_records(capsys) -> None:
    _emit_records(
        [
            {"path": "board.kicad_pcb", "kind": "kicad_pcb", "bytes": 42},
            {"path": "sheet.kicad_sch", "kind": "kicad_sch", "bytes": 84},
        ],
        "names",
    )

    assert capsys.readouterr().out == "board.kicad_pcb\nsheet.kicad_sch\n"


def test_file_manifest_excludes_generated_and_history_trees(tmp_path: Path) -> None:
    (tmp_path / "live.kicad_sch").write_text("live", encoding="utf-8")
    for generated in (".history", "build", "output", "tmp"):
        directory = tmp_path / generated
        directory.mkdir()
        (directory / "stale.kicad_sch").write_text("stale", encoding="utf-8")

    records = list(file_records(argparse.Namespace(paths=[str(tmp_path)])))

    assert [Path(str(record["path"])).name for record in records] == ["live.kicad_sch"]


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
        commit = object()

        def get_as_string(self) -> str:
            return next(self.contents)

        def begin_commit(self) -> object:
            effects.append("begin")
            return self.commit

        def drop_commit(self, commit: object) -> None:
            assert commit is self.commit
            effects.append("drop")

        def push_commit(self, commit: object, message: str) -> None:
            assert commit is self.commit
            assert message == "placement"
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


async def test_native_placement_transaction_rejects_new_findings_despite_lower_total(
    tmp_path: Path, monkeypatch
) -> None:
    import kicad_mcp.shell_cli as shell_cli

    effects: list[str] = []
    commit = object()

    class FakeBoard:
        contents = iter(["before", "committed"])

        def get_as_string(self) -> str:
            return next(self.contents)

        def begin_commit(self) -> object:
            effects.append("begin")
            return commit

        def drop_commit(self, active: object) -> None:
            assert active is commit
            effects.append("drop")

        def push_commit(self, active: object, message: str) -> None:
            assert active is commit
            assert message == "placement"
            effects.append("push")

        def save(self) -> None:
            effects.append("save")

    monkeypatch.setattr(
        shell_cli,
        "authority_report",
        lambda _root: {"policy": {"board_mutation_allowed": True}},
    )
    monkeypatch.setattr(shell_cli, "get_board", FakeBoard)
    monkeypatch.setattr(
        shell_cli,
        "_native_move_footprints_batch",
        lambda _board, _arguments: {
            "ok": True,
            "tool": "_native_move_footprints_batch",
            "result": {"moved": 1},
        },
    )
    monkeypatch.setattr(
        shell_cli,
        "_apply_footprint_batch_to_board_content",
        lambda _content, _arguments: "candidate",
    )
    monkeypatch.setattr(shell_cli, "_verify_staged_footprint_batch", lambda *_args: None)
    monkeypatch.setattr(shell_cli, "_verify_rigid_footprint_children", lambda *_args: None)
    drc_results = iter(
        [
            {
                "finding_keys": ['{"kind":"old-a"}', '{"kind":"old-b"}'],
                "summary": {"violations": 2, "unconnected_items": 1},
            },
            {
                "finding_keys": ['{"kind":"new-a"}'],
                "summary": {"violations": 1, "unconnected_items": 2},
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
        [
            (
                "_native_move_footprints_batch",
                {"placements": [{"reference": "U1", "x_mm": 1, "y_mm": 2, "rotation_deg": 0}]},
            )
        ],
        label="placement",
    )

    assert report["status"] == "rejected"
    assert report["regressions"]["regressed"] is True
    assert report["regressions"]["unconnected_increase"] == 1
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
            "counts": {"footprints": 2, "tracks": 0, "vias": 0},
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
    focused_board = ascii_map(filtered, zoom=3)
    assert "FP=1/2" in focused_board
    assert "view=" in focused_board


def test_pad_rotation_uses_kicad_board_coordinate_direction() -> None:
    from kicad_mcp.deep_inspection import _pad_positions

    footprint = {
        "x_mm": 60.0,
        "y_mm": 116.5,
        "rotation": 90.0,
        "layer_name": "F.Cu",
        "block": """(footprint "Header"
          (at 60 116.5 90)
          (property "Reference" "J10")
          (pad "18" thru_hole circle (at 2.54 20.32) (size 1.7 1.7)
            (drill 1) (layers "*.Cu") (net 1 "AMP_SD")))""",
    }

    pad = _pad_positions(footprint)[0]

    assert pad["at"] == [80.32, 113.96]


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


def test_connectivity_proof_accepts_kicad_slash_token_in_unconnected_net() -> None:
    snapshot = _snapshot()
    snapshot["schematic"]["nets"] = [
        {
            "name": "unconnected-(U1-A/B-Pad1)",
            "unconnected": True,
            "nodes": [
                {
                    "reference": "U1",
                    "pin": "1",
                    "function": "A/B",
                    "type": "bidirectional+no_connect",
                }
            ],
        }
    ]
    snapshot["board"]["footprints"][0]["pads"][0]["net"] = "unconnected-(U1-A{slash}B-Pad1)"

    proof = connectivity_proof(snapshot, reference="U1")

    assert proof["findings"]["board_net_mismatches"] == []


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


def test_route_plan_keeps_adjacent_endpoint_pads_as_obstacles() -> None:
    snapshot = copy.deepcopy(_snapshot())
    snapshot["board"]["footprints"][0]["pads"].append(
        {"number": "2", "at": [5.0, 5.7], "size": [0.5, 0.5], "net": "GND"}
    )
    snapshot["board"]["footprints"][0]["pads"][0]["size"] = [0.5, 0.5]
    snapshot["board"]["footprints"][1]["pads"][0]["size"] = [0.5, 0.5]

    narrow = route_plan(snapshot, "GPIO", width_mm=0.2, clearance_mm=0.1)
    wide = route_plan(snapshot, "GPIO", width_mm=1.0, clearance_mm=0.2)

    assert narrow["collision_score"] == 0
    assert wide["collision_score"] > 0


def test_critical_placement_report_measures_same_net_pad_distance() -> None:
    snapshot = _snapshot()

    report = critical_placement_report(
        snapshot,
        [
            {
                "reference_a": "U1",
                "reference_b": "R1",
                "nets": ["GPIO"],
                "max_pad_distance_mm": 8.0,
                "reason": "test loop",
            }
        ],
    )

    assert report["status"] == "fail"
    assert report["findings"][0]["measurements"][0]["distance_mm"] == 10.0
    assert report["findings"][0]["measurements"][0]["status"] == "fail"


def test_critical_placement_plan_respects_allowed_rotations() -> None:
    snapshot = _snapshot()

    plan = critical_placement_plan(
        snapshot,
        [
            {
                "reference_a": "U1",
                "reference_b": "R1",
                "nets": ["GPIO"],
                "max_pad_distance_mm": 4.0,
                "allowed_rotations": [180.0],
            }
        ],
        grid_mm=0.25,
        courtyard_margin_mm=0.0,
    )

    assert plan["status"] == "planned"
    assert plan["placements"][0]["rotation"] == 180.0
    assert plan["after"]["status"] == "pass"


def test_candidate_stackup_spec_and_named_net_track_rendering() -> None:
    layers = _stackup_layers_from_spec(
        {
            "stackup": [
                {"layer": "F.Cu", "material": "copper", "thickness_mm": 0.035},
                {
                    "layer": "dielectric_1",
                    "material": "FR-4",
                    "thickness_mm": 0.2,
                    "epsilon_r": 4.2,
                },
                {"layer": "B.Cu", "material": "copper", "thickness_mm": 0.035},
            ]
        }
    )
    rendered = _render_candidate_track(
        {
            "x1": 1.0,
            "y1": 2.0,
            "x2": 3.0,
            "y2": 4.0,
            "width": 0.25,
            "layer": "F_Cu",
            "net": "/USB/D+",
        }
    )

    assert [layer.type for layer in layers] == ["signal", "prepreg", "signal"]
    assert '(layer "F.Cu")' in rendered
    assert '(net "/USB/D+")' in rendered
    assert "(uuid " in rendered


def test_placement_plan_is_deterministic_and_holds_fixed_references() -> None:
    snapshot = _snapshot()
    snapshot["board"]["footprints"][0]["rotation"] = 90
    first = placement_plan(snapshot, fixed_references=["U1"], iterations=20)
    second = placement_plan(snapshot, fixed_references=["U1"], iterations=20)

    assert first == second
    u1 = next(item for item in first["placements"] if item["reference"] == "U1")
    assert u1["fixed"] is True
    assert u1["from"] == u1["to"]
    assert u1["rotation"] == 90.0
    assert first["weighted_hpwl_before_mm"] >= 0.0
    assert first["weighted_hpwl_after_mm"] >= 0.0
    assert first["quality_gate"]["status"] in {"pass", "fail"}


def test_placement_plan_resolves_edge_anchor_and_rotation() -> None:
    snapshot = _snapshot()

    plan = placement_plan(
        snapshot,
        anchors=[{"reference": "U1", "edge": "left", "offset_mm": 4, "rotation": 90}],
        iterations=5,
    )

    anchor = plan["anchors"][0]
    u1 = next(item for item in plan["placements"] if item["reference"] == "U1")
    assert anchor["edge"] == "left"
    assert anchor["rotation"] == 90.0
    assert anchor["position_mm"][1] == 4.0
    assert u1["fixed"] is True
    assert u1["anchored"] is True
    assert u1["to"] == anchor["position_mm"]
    assert u1["rotation"] == 90.0


def test_placement_plan_resolves_absolute_anchor() -> None:
    plan = placement_plan(
        _snapshot(),
        anchors=[{"reference": "U1", "x_mm": 8, "y_mm": 4}],
        iterations=5,
        margin_mm=0,
    )

    anchor = plan["anchors"][0]
    u1 = next(item for item in plan["placements"] if item["reference"] == "U1")
    assert anchor["edge"] == "absolute"
    assert anchor["position_mm"] == [8.0, 4.0]
    assert u1["to"] == [8.0, 4.0]


def test_placement_plan_edge_anchor_accounts_for_asymmetric_origin() -> None:
    snapshot = _snapshot()
    footprint = snapshot["board"]["footprints"][0]
    footprint.update(
        {
            "bbox_min_x_mm": -1.0,
            "bbox_max_x_mm": 5.0,
            "bbox_min_y_mm": -2.0,
            "bbox_max_y_mm": 2.0,
            "width_mm": 6.0,
            "height_mm": 4.0,
        }
    )

    plan = placement_plan(
        snapshot,
        anchors=[{"reference": "U1", "edge": "right", "offset_mm": 5}],
        iterations=5,
        margin_mm=0,
    )

    anchor = plan["anchors"][0]
    assert anchor["position_mm"] == [15.0, 5.0]
    assert next(item for item in plan["placements"] if item["reference"] == "U1")["to"] == [
        15.0,
        5.0,
    ]


def test_placement_plan_constrains_hierarchical_sheet_to_cluster_region() -> None:
    plan = placement_plan(
        _snapshot(),
        cluster_regions=[{"sheet": "/Power/", "x1_mm": 0, "y1_mm": 0, "x2_mm": 10, "y2_mm": 10}],
        iterations=10,
        margin_mm=0,
    )

    assert plan["clustered_references"] == 2
    assert plan["clusters"][0]["sheet"] == "/Power/"
    assert all(
        0 <= coordinate <= 10 for placement in plan["placements"] for coordinate in placement["to"]
    )
    assert plan["weighted_hpwl_legalized_baseline_mm"] >= 0.0


def test_shell_source_does_not_use_shell_execution() -> None:
    source = Path(__file__).parents[2] / "src" / "kicad_mcp" / "deep_inspection.py"
    text = source.read_text()
    assert "shell=True" not in text


def test_power_loop_report_uses_same_net_pad_geometry_and_ground_return() -> None:
    snapshot = {
        "board": {
            "footprints": [
                {
                    "reference": "U1",
                    "value": "MCU",
                    "x_mm": 10.0,
                    "y_mm": 10.0,
                    "pads": [
                        {"number": "2", "net": "+3V3", "at": [9.0, 10.0]},
                        {"number": "1", "net": "GND", "at": [9.0, 11.0]},
                    ],
                },
                {
                    "reference": "C1",
                    "value": "100nF",
                    "x_mm": 7.0,
                    "y_mm": 10.0,
                    "pads": [
                        {"number": "1", "net": "+3V3", "at": [8.0, 10.0]},
                        {"number": "2", "net": "GND", "at": [8.0, 11.0]},
                    ],
                },
            ]
        }
    }

    report = power_loop_report(
        snapshot,
        [{"ic_ref": "U1", "cap_refs": ["C1"], "max_distance_mm": 1.1}],
    )

    member = report["groups"][0]["members"][0]
    assert report["status"] == "pass"
    assert member["forward"]["net"] == "+3V3"
    assert member["forward"]["distance_mm"] == pytest.approx(1.0)
    assert member["return"]["distance_mm"] == pytest.approx(1.0)
    assert member["loop_proxy_mm"] == pytest.approx(2.0)
    assert member["origin_distance_mm"] == pytest.approx(3.0)


def test_power_loop_report_rejects_declared_cap_without_shared_rail() -> None:
    snapshot = {
        "board": {
            "footprints": [
                {
                    "reference": "U1",
                    "value": "MCU",
                    "x_mm": 0.0,
                    "y_mm": 0.0,
                    "pads": [
                        {"number": "1", "net": "+3V3", "at": [0.0, 0.0]},
                        {"number": "2", "net": "GND", "at": [0.0, 1.0]},
                    ],
                },
                {
                    "reference": "C1",
                    "value": "100nF",
                    "x_mm": 1.0,
                    "y_mm": 0.0,
                    "pads": [
                        {"number": "1", "net": "+5V", "at": [1.0, 0.0]},
                        {"number": "2", "net": "GND", "at": [1.0, 1.0]},
                    ],
                },
            ]
        }
    }

    report = power_loop_report(
        snapshot,
        [{"ic_ref": "U1", "cap_refs": ["C1"], "max_distance_mm": 2.0}],
    )

    member = report["groups"][0]["members"][0]
    assert report["status"] == "fail"
    assert member["forward"] is None
    assert member["reason"] == "no shared non-ground pad net"
