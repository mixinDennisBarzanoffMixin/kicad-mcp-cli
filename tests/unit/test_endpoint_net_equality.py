"""Regression tests for post-build circuit endpoint equality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kicad_mcp.shell_cli as shell_cli
from kicad_mcp.endpoint_net_equality import compare_compiled_endpoint_nets


def _pin(reference: str, pin: str, net: str, *, function: str = "") -> dict[str, object]:
    return {
        "reference": reference,
        "pin": pin,
        "function": function,
        "net": net,
        "type": "passive",
    }


def _build(nets: list[dict[str, object]]) -> list[tuple[str, dict[str, object]]]:
    return [("sch_build_circuit", {"nets": nets})]


def test_usb_duplicate_contacts_must_all_export_on_requested_net() -> None:
    calls = _build(
        [
            {
                "name": "USB_CONN_D+",
                "endpoints": ["J3.A6", "J3.B6", "U14.1"],
            },
            {
                "name": "USB_CONN_D-",
                "endpoints": ["J3.A7", "J3.B7", "U14.2"],
            },
        ]
    )
    connectivity = {
        "pins": [
            _pin("J3", "A6", "/USB/USB_CONN_D+", function="D+_A6"),
            _pin("J3", "B6", "/USB/USB_CONN_D+", function="D+_B6"),
            _pin("U14", "1", "/USB/USB_CONN_D+", function="D1+_1"),
            _pin("J3", "A7", "/USB/USB_CONN_D-", function="D-_A7"),
            _pin("J3", "B7", "/USB/USB_CONN_D-", function="D-_B7"),
            _pin("U14", "2", "/USB/USB_CONN_D-", function="D1-_2"),
        ]
    }

    report = compare_compiled_endpoint_nets(calls, connectivity)

    assert report["status"] == "pass"
    assert report["requested_endpoints"] == report["compared_endpoints"] == 6
    assert report["mismatches"] == []


def test_esp_module_pin_short_is_reported_with_exact_expected_and_actual_net() -> None:
    calls = _build(
        [
            {"name": "USB_D+", "endpoints": ["U1.14"]},
            {"name": "LTE_REG_CTL", "endpoints": ["U1.25"]},
        ]
    )
    connectivity = {
        "pins": [
            _pin("U1", "14", "/ESP/ESP_3V3", function="USB_D+_14"),
            _pin("U1", "25", "I2C_SCL", function="IO48_25"),
        ]
    }

    report = compare_compiled_endpoint_nets(calls, connectivity)

    assert report["status"] == "fail"
    assert report["mismatches"] == [
        {
            "endpoint": "U1.14",
            "reference": "U1",
            "pin": "14",
            "expected_net": "USB_D+",
            "actual_net": "ESP_3V3",
            "reason": "wrong_net",
        },
        {
            "endpoint": "U1.25",
            "reference": "U1",
            "pin": "25",
            "expected_net": "LTE_REG_CTL",
            "actual_net": "I2C_SCL",
            "reason": "wrong_net",
        },
    ]


def test_rotated_two_pin_orientation_swap_is_a_hard_failure() -> None:
    calls = _build(
        [
            {"name": "SOURCE", "endpoints": ["R21.1"]},
            {"name": "SINK", "endpoints": ["R21.2"]},
        ]
    )
    connectivity = {
        "pins": [
            _pin("R21", "1", "/USB/SINK"),
            _pin("R21", "2", "/USB/SOURCE"),
        ]
    }

    report = compare_compiled_endpoint_nets(calls, connectivity)

    assert report["status"] == "fail"
    mismatches = [
        (item["endpoint"], item["expected_net"], item["actual_net"])
        for item in report["mismatches"]
    ]
    assert mismatches == [
        ("R21.1", "SOURCE", "SINK"),
        ("R21.2", "SINK", "SOURCE"),
    ]


def _verification(pins: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "pass",
        "checks": {
            "source_integrity": {"status": "pass", "findings": []},
            "connectivity": {"status": "pass", "pins": pins, "findings": {}},
            "erc": {"status": "pass", "findings": []},
        },
    }


async def test_staged_build_rejects_wrong_exported_endpoint_net(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "demo.kicad_pro").write_text("{}", encoding="utf-8")
    child = tmp_path / "usb.kicad_sch"
    child.write_text("before\n", encoding="utf-8")
    reports = iter(
        [
            _verification([]),
            _verification(
                [
                    _pin("J3", "A6", "/USB/USB_CONN_D+"),
                    _pin("J3", "B6", "EPD_CS_N"),
                ]
            ),
        ]
    )
    monkeypatch.setattr(shell_cli, "verification_report", lambda *_args, **_kwargs: next(reports))

    async def mutate(stage_args, _tool, _arguments):
        target = Path(stage_args.project_dir) / "usb.kicad_sch"
        target.write_text("compiled\n", encoding="utf-8")
        return {"ok": True, "tool": "sch_build_circuit"}

    monkeypatch.setattr(shell_cli, "invoke_backend_tool", mutate)
    arguments = {
        "nets": [
            {
                "name": "USB_CONN_D+",
                "endpoints": ["J3.A6", "J3.B6"],
            }
        ]
    }
    args = argparse.Namespace(
        mode="write",
        yes=True,
        tool="sch_build_circuit",
        project_dir=str(tmp_path),
        artifacts=str(tmp_path / "evidence"),
        sheet="usb",
        reference="",
        net="",
        args_json=json.dumps(arguments),
        set_values=[],
        profile="full",
    )

    report = await shell_cli.run_staged_schematic_edit(args)

    assert report["status"] == "rejected"
    assert report["promoted"] is False
    assert report["reason"] == "staged build endpoint-to-net equality failed"
    assert report["endpoint_net_equality"]["mismatches"][0] == {
        "endpoint": "J3.B6",
        "reference": "J3",
        "pin": "B6",
        "expected_net": "USB_CONN_D+",
        "actual_net": "EPD_CS_N",
        "reason": "wrong_net",
    }
    assert child.read_text(encoding="utf-8") == "before\n"
