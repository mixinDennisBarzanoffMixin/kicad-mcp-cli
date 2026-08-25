"""Regression tests for explicit verification filters disappearing after edits."""

from __future__ import annotations

import argparse
from pathlib import Path

import kicad_mcp.deep_inspection as deep_inspection
import kicad_mcp.shell_cli as shell_cli
from kicad_mcp.deep_inspection import connectivity_proof, verification_report


def _snapshot(*, include_net: bool = True) -> dict[str, object]:
    nets = []
    if include_net:
        nets.append(
            {
                "name": "/Power/LTE_FF_MID",
                "unconnected": False,
                "nodes": [
                    {
                        "reference": "R1",
                        "pin": "1",
                        "function": "",
                        "type": "passive",
                    },
                    {
                        "reference": "C1",
                        "pin": "1",
                        "function": "",
                        "type": "passive",
                    },
                ],
            }
        )
    return {
        "project": {"name": "demo", "directory": "."},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Power/", "file": "power.kicad_sch"}],
            "components": [
                {
                    "reference": "R1",
                    "value": "10k",
                    "footprint": "R_0603",
                    "sheet": "/Power/",
                },
                {
                    "reference": "C1",
                    "value": "1nF",
                    "footprint": "C_0603",
                    "sheet": "/Power/",
                },
            ],
            "nets": nets,
        },
        "board": {"footprints": [], "tracks": [], "vias": []},
    }


def test_explicit_net_filter_missing_is_hard_connectivity_failure() -> None:
    proof = connectivity_proof(_snapshot(include_net=False), sheet="Power", net="LTE_FF_MID")

    assert proof["status"] == "fail"
    assert proof["summary"]["nets"] == 0
    assert proof["summary"]["pins"] == 0
    assert proof["findings"]["requested_net_not_found"] == [
        {
            "code": "requested_net_not_found",
            "requested": "LTE_FF_MID",
            "match_semantics": "case-insensitive substring",
            "selected_components": 2,
            "reason": "explicit net filter matched no net pins in the selected scope",
        }
    ]


def test_net_filter_keeps_documented_substring_semantics() -> None:
    proof = connectivity_proof(_snapshot(), sheet="Power", net="ff_mid")

    assert proof["status"] == "pass"
    assert proof["summary"]["nets"] == 1
    assert proof["findings"]["requested_net_not_found"] == []


def test_exact_reference_filter_missing_is_hard_failure() -> None:
    proof = connectivity_proof(_snapshot(), sheet="Power", reference="R404")

    assert proof["status"] == "fail"
    assert proof["findings"]["requested_reference_not_found"][0]["requested"] == "R404"
    assert proof["filter_semantics"]["reference"] == "case-insensitive exact reference"


def test_verification_report_propagates_missing_requested_net(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "demo.kicad_pro"
    schematic = tmp_path / "demo.kicad_sch"
    board = tmp_path / "demo.kicad_pcb"
    project.write_text("{}", encoding="utf-8")
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    board.write_text("(kicad_pcb)", encoding="utf-8")
    monkeypatch.setattr(
        deep_inspection,
        "project_snapshot",
        lambda _root: _snapshot(include_net=False),
    )
    monkeypatch.setattr(
        deep_inspection,
        "_source_integrity_evidence",
        lambda *_args, **_kwargs: {"status": "pass", "findings": []},
    )
    monkeypatch.setattr(
        deep_inspection,
        "_erc_evidence",
        lambda *_args, **_kwargs: {"status": "pass", "findings": []},
    )

    report = verification_report(tmp_path, sheet="Power", net="LTE_FF_MID")

    assert report["status"] == "fail"
    assert report["checks"]["connectivity"]["status"] == "fail"
    assert report["checks"]["connectivity"]["findings"]["requested_net_not_found"]


def _verification(connectivity_status: str) -> dict[str, object]:
    return {
        "status": "fail" if connectivity_status == "fail" else "pass",
        "checks": {
            "source_integrity": {"status": "pass", "findings": []},
            "connectivity": {
                "status": connectivity_status,
                "findings": {
                    "requested_net_not_found": (
                        [{"requested": "LTE_FF_MID"}] if connectivity_status == "fail" else []
                    )
                },
            },
            "erc": {"status": "pass", "findings": []},
        },
    }


async def test_staged_edit_rejects_when_requested_net_disappears(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "demo.kicad_pro").write_text("{}", encoding="utf-8")
    child = tmp_path / "power.kicad_sch"
    child.write_text("before\n", encoding="utf-8")
    reports = iter([_verification("pass"), _verification("fail")])
    monkeypatch.setattr(shell_cli, "verification_report", lambda *_args, **_kwargs: next(reports))

    async def mutate(stage_args, _tool, _arguments):
        target = Path(stage_args.project_dir) / "power.kicad_sch"
        target.write_text("after without LTE_FF_MID label\n", encoding="utf-8")
        return {"ok": True, "tool": "sch_delete_label"}

    monkeypatch.setattr(shell_cli, "invoke_backend_tool", mutate)
    args = argparse.Namespace(
        mode="write",
        yes=True,
        tool="sch_delete_label",
        project_dir=str(tmp_path),
        artifacts=str(tmp_path / "evidence"),
        sheet="power",
        reference="",
        net="LTE_FF_MID",
        args_json="{}",
        set_values=[],
        profile="full",
    )

    report = await shell_cli.run_staged_schematic_edit(args)

    assert report["status"] == "rejected"
    assert report["promoted"] is False
    assert report["after"]["checks"]["connectivity"]["status"] == "fail"
    assert child.read_text(encoding="utf-8") == "before\n"
