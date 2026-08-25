"""Tests for the CLI-first circuit-spec railway transformer."""

from __future__ import annotations

import io
import json

import pytest

from kicad_mcp.circuit_spec_arrangement import (
    arrange_circuit_spec,
    format_circuit_spec_arrangement,
)
from kicad_mcp.shell_cli import build_parser, main


def _spec() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "sheet": "Demo",
        "auto_layout": True,
        "max_paper": "A4",
        "symbols": [
            {
                "library": "Connector_Generic",
                "symbol_name": "Conn_01x02",
                "reference": "J1",
                "value": "INPUT",
                "x_mm": 50.0,
                "y_mm": 60.0,
            },
            {
                "library": "Device",
                "symbol_name": "R",
                "reference": "R1",
                "value": "1k",
            },
            {
                "library": "Device",
                "symbol_name": "C",
                "reference": "C1",
                "value": "100n",
                "x_mm": 80.0,
            },
            {
                "library": "Connector_Generic",
                "symbol_name": "Conn_01x02",
                "reference": "J2",
                "value": "OUTPUT",
            },
        ],
        "nets": [
            {"name": "SIG_IN", "scope": "local", "endpoints": ["J1.1", "R1.1"]},
            {"name": "SIG_OUT", "scope": "local", "endpoints": ["R1.2", "J2.1"]},
            {
                "name": "GND",
                "scope": "global",
                "endpoints": ["J1.2", "R1.3", "C1.2", "J2.2"],
            },
        ],
    }


def test_arrangement_preserves_anchors_and_only_fills_absent_coordinates() -> None:
    spec = _spec()

    first = arrange_circuit_spec(spec, source="demo.json")
    second = arrange_circuit_spec(spec, source="demo.json")

    assert first == second
    arranged = first["arranged_spec"]
    arranged_by_ref = {symbol["reference"]: symbol for symbol in arranged["symbols"]}
    assert arranged_by_ref["J1"]["x_mm"] == 50.0
    assert arranged_by_ref["J1"]["y_mm"] == 60.0
    assert arranged_by_ref["C1"]["x_mm"] == 80.0
    assert isinstance(arranged_by_ref["C1"]["y_mm"], float)
    assert all("x_mm" in symbol and "y_mm" in symbol for symbol in arranged["symbols"])
    assert arranged["auto_layout"] is False
    assert spec["auto_layout"] is True
    assert first["summary"]["explicit_anchors"] == 1
    assert first["summary"]["partial_anchors"] == 1
    assert first["summary"]["generated_symbols"] == 3


def test_spec_payload_contains_no_planning_diagnostics() -> None:
    result = arrange_circuit_spec(_spec())
    payload = result["arranged_spec"]

    assert set(payload) == {
        "schema_version",
        "sheet",
        "auto_layout",
        "max_paper",
        "symbols",
        "nets",
    }
    assert not ({"layout", "status", "selected_candidate", "cost"} & set(payload))
    assert payload["nets"] == _spec()["nets"]


def test_json_diagnostics_expose_candidates_ranks_and_rails() -> None:
    result = arrange_circuit_spec(_spec(), candidate_count=2)

    assert len(result["layout"]["ranked_candidates"]) == 2
    assert all("cost" in candidate for candidate in result["layout"]["ranked_candidates"])
    assert all("rank" in placement for placement in result["layout"]["placements"])
    assert result["layout"]["graph"]["rail_nets"] == ["GND"]
    report = format_circuit_spec_arrangement(result)
    assert "CANDIDATES" in report
    assert "RANKS" in report
    assert "RAILS GND" in report


def test_cli_spec_mode_reads_stdin_and_emits_only_build_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_spec())))

    main(["arrange-spec", "-", "--format", "spec", "--candidates", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["auto_layout"] is False
    assert all("x_mm" in symbol and "y_mm" in symbol for symbol in payload["symbols"])
    assert "layout" not in payload


def test_cli_parser_exposes_pipe_and_candidate_controls() -> None:
    args = build_parser().parse_args(
        ["arrange-spec", "-", "--format", "json", "--candidates", "2"]
    )

    assert args.path == "-"
    assert args.format == "json"
    assert args.candidate_count == 2
    assert not hasattr(args, "apply")


def test_invalid_endpoint_fails_before_generating_coordinates() -> None:
    spec = _spec()
    spec["nets"] = [{"name": "BAD", "endpoints": ["R1"]}]

    with pytest.raises(ValueError, match="REF.PIN"):
        arrange_circuit_spec(spec)
