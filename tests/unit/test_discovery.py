from __future__ import annotations

from pathlib import Path

from kicad_mcp.discovery import scan_project_dir


def test_scan_finds_kicad_files(tmp_path: Path) -> None:
    (tmp_path / "board.kicad_pcb").touch()
    (tmp_path / "schematic.kicad_sch").touch()
    (tmp_path / "demo.kicad_pro").touch()
    result = scan_project_dir(tmp_path)
    assert result["pcb"] is not None
    assert result["schematic"] is not None
    assert result["project"] is not None


def test_scan_prefers_canonical_project_over_numbered_duplicate(tmp_path: Path) -> None:
    project_dir = tmp_path / "light-noise-detektor"
    project_dir.mkdir()
    canonical = project_dir / "light-noise-detektor.kicad_pro"
    duplicate = project_dir / "light-noise-detektor 2.kicad_pro"
    duplicate.touch()
    canonical.touch()

    result = scan_project_dir(project_dir)

    assert result["project"] == canonical


def test_scan_binds_hierarchical_companions_to_project_basename(tmp_path: Path) -> None:
    project_dir = tmp_path / "offline-candidate-random-name"
    project_dir.mkdir()
    project = project_dir / "Flux-RevA.kicad_pro"
    board = project_dir / "Flux-RevA.kicad_pcb"
    root_schematic = project_dir / "Flux-RevA.kicad_sch"
    child_schematic = project_dir / "01_Solar_Charger_Battery.kicad_sch"
    for path in (project, board, root_schematic, child_schematic):
        path.touch()

    result = scan_project_dir(project_dir)

    assert result == {
        "project": project,
        "pcb": board,
        "schematic": root_schematic,
    }
