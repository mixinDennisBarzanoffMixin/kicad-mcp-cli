from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP

from kicad_mcp.tools import footprint as fp


def _tool(tmp_path: Path, monkeypatch):
    server = FastMCP("footprint-export-test")
    monkeypatch.setattr(fp, "get_config", lambda: SimpleNamespace(project_dir=tmp_path))
    monkeypatch.setattr(fp, "_ensure_output_dir", lambda _name: tmp_path / "default-output")
    fp.register(server)
    return {tool.name: tool for tool in server._tool_manager.list_tools()}["fp_export_svg"]


def test_single_footprint_file_is_normalized_and_output_directory_is_created(
    tmp_path: Path, monkeypatch
) -> None:
    library = tmp_path / "Demo.pretty"
    library.mkdir()
    source = library / "ExactPart.kicad_mod"
    source.write_text('(footprint "ExactPart")\n', encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run_cli(*args: str) -> tuple[int, str, str]:
        calls.append(args)
        output_dir = Path(args[args.index("--output") + 1])
        (output_dir / "ExactPart.svg").write_text("<svg/>", encoding="utf-8")
        return 0, "Done.", ""

    monkeypatch.setattr(fp, "_run_cli", fake_run_cli)
    result = _tool(tmp_path, monkeypatch).fn(
        input_path="Demo.pretty/ExactPart.kicad_mod",
        output_dir="nested/svg-output",
    )

    assert "exported successfully" in result
    assert (tmp_path / "nested/svg-output/ExactPart.svg").is_file()
    assert calls[0][-1] == str(library)
    assert calls[0][calls[0].index("--footprint") + 1] == "ExactPart"


def test_zero_exit_without_svg_is_reported_as_failure(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "Demo.pretty"
    library.mkdir()
    monkeypatch.setattr(fp, "_run_cli", lambda *_args: (0, "Done.", ""))

    result = _tool(tmp_path, monkeypatch).fn(
        input_path="Demo.pretty",
        footprint="MissingPart",
        output_dir="svg-output",
    )

    assert result.startswith("Footprint SVG export failed:")


def test_unchanged_stale_svg_does_not_count_as_new_export(
    tmp_path: Path, monkeypatch
) -> None:
    library = tmp_path / "Demo.pretty"
    library.mkdir()
    output = tmp_path / "svg-output"
    output.mkdir()
    (output / "MissingPart.svg").write_text("<svg>stale</svg>", encoding="utf-8")
    monkeypatch.setattr(fp, "_run_cli", lambda *_args: (0, "Done.", ""))

    result = _tool(tmp_path, monkeypatch).fn(
        input_path="Demo.pretty",
        footprint="MissingPart",
        output_dir="svg-output",
    )

    assert result.startswith("Footprint SVG export failed:")
