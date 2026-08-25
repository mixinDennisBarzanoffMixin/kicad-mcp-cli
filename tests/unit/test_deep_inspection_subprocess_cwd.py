"""Regression tests for hierarchical KiCad subprocess path resolution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kicad_mcp import deep_inspection


def test_netlist_export_runs_from_selected_schematic_directory(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "staged-project"
    project.mkdir()
    schematic = project / "demo.kicad_sch"
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["cwd"] = kwargs.get("cwd")
        output = Path(command[command.index("-o") + 1])
        output.write_text("<export/>", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(deep_inspection, "_kicad_cli", lambda: "kicad-cli")
    monkeypatch.setattr(deep_inspection.subprocess, "run", fake_run)

    deep_inspection._export_netlist(schematic)

    assert observed["cwd"] == project


def test_erc_runs_from_selected_schematic_directory(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "staged-project"
    project.mkdir()
    schematic = project / "demo.kicad_sch"
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["cwd"] = kwargs.get("cwd")
        output = Path(command[command.index("-o") + 1])
        output.write_text(
            json.dumps({"kicad_version": "test", "sheets": []}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(deep_inspection, "_kicad_cli", lambda: "kicad-cli")
    monkeypatch.setattr(deep_inspection.subprocess, "run", fake_run)

    deep_inspection._erc_evidence(schematic)

    assert observed["cwd"] == project


def test_svg_export_runs_from_selected_schematic_directory(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "staged-project"
    project.mkdir()
    schematic = project / "demo.kicad_sch"
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["cwd"] = kwargs.get("cwd")
        observed["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "power.svg").write_text("<svg/>", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(deep_inspection, "_kicad_cli", lambda: "kicad-cli")
    monkeypatch.setattr(deep_inspection.subprocess, "run", fake_run)

    deep_inspection._render_verification_svgs(
        schematic,
        {
            "schematic": {
                "sheets": [
                    {"name": "Power", "number": 1, "file": "power.kicad_sch"}
                ]
            }
        },
        output_dir,
        sheet="Power",
    )

    assert observed["cwd"] == project
    assert str(project / "power.kicad_sch") in observed["command"]
