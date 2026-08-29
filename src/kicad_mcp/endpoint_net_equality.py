"""Post-build endpoint-to-net equality checks for schematic compilation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

type JsonRecord = dict[str, Any]
type ToolCall = tuple[str, dict[str, Any]]


def _canonical_net(name: object) -> str:
    return str(name or "").rstrip("/").rsplit("/", 1)[-1]


def _normalize_endpoint(endpoint: object) -> tuple[str, str] | None:
    if isinstance(endpoint, str):
        for separator in (".", ":"):
            if separator in endpoint:
                reference, pin = endpoint.split(separator, 1)
                if reference and pin:
                    return reference, pin
        return None
    if not isinstance(endpoint, dict):
        return None
    raw_reference = endpoint.get("reference", endpoint.get("ref", endpoint.get("symbol")))
    raw_pin = endpoint.get(
        "pin",
        endpoint.get(
            "pin_number",
            endpoint.get("number", endpoint.get("pin_name", endpoint.get("pad"))),
        ),
    )
    if raw_reference is None or raw_pin is None:
        return None
    return str(raw_reference), str(raw_pin)


def _endpoint_lists(net: JsonRecord) -> list[object]:
    for key in ("endpoints", "connections", "pins", "nodes"):
        value = net.get(key)
        if isinstance(value, list):
            return value
    from_ref = net.get("from_ref", net.get("from_reference"))
    to_ref = net.get("to_ref", net.get("to_reference"))
    if from_ref is not None and to_ref is not None:
        return [
            {"reference": from_ref, "pin": net.get("from_pin")},
            {"reference": to_ref, "pin": net.get("to_pin")},
        ]
    return []


def _alias(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _pin_function_alias(pin: JsonRecord) -> str:
    function = str(pin.get("function", ""))
    number = str(pin.get("pin", ""))
    suffix = f"_{number}"
    if function.endswith(suffix):
        function = function[: -len(suffix)]
    return _alias(function)


def _expected_records(tool_calls: Sequence[ToolCall]) -> tuple[list[JsonRecord], list[JsonRecord]]:
    records: list[JsonRecord] = []
    uncomparable: list[JsonRecord] = []
    for tool_name, arguments in tool_calls:
        if tool_name != "sch_build_circuit":
            continue
        nets = arguments.get("nets", [])
        if not isinstance(nets, list):
            continue
        for net_index, net in enumerate(nets):
            if not isinstance(net, dict):
                continue
            net_name = _canonical_net(net.get("name", net.get("net", net.get("label", ""))))
            for endpoint_index, endpoint in enumerate(_endpoint_lists(net)):
                identity = _normalize_endpoint(endpoint)
                if identity is None:
                    # Power/label-only endpoints do not identify a component pin
                    # and therefore are outside this exact endpoint invariant.
                    if isinstance(endpoint, str) or (
                        isinstance(endpoint, dict)
                        and (
                            any(
                                key in endpoint
                                for key in (
                                    "power",
                                    "power_symbol",
                                    "rail",
                                    "label",
                                    "net_label",
                                )
                            )
                            or endpoint.get("type") in {"power", "label"}
                        )
                    ):
                        continue
                    uncomparable.append(
                        {
                            "net": net_name,
                            "net_index": net_index,
                            "endpoint_index": endpoint_index,
                            "endpoint": endpoint,
                            "reason": "component reference and pin are required",
                        }
                    )
                    continue
                reference, pin = identity
                records.append(
                    {
                        "endpoint": f"{reference}.{pin}",
                        "reference": reference,
                        "pin": pin,
                        "expected_net": net_name,
                    }
                )
    return records, uncomparable


def compare_compiled_endpoint_nets(
    tool_calls: Sequence[ToolCall],
    connectivity: JsonRecord,
) -> JsonRecord:
    """Compare requested build endpoints with KiCad-exported pin connectivity."""

    expected, uncomparable = _expected_records(tool_calls)
    if not expected and not uncomparable:
        return {
            "status": "not_applicable",
            "requested_endpoints": 0,
            "compared_endpoints": 0,
            "mismatches": [],
            "duplicate_assignments": [],
            "uncomparable_endpoints": [],
        }

    assignments: dict[tuple[str, str], set[str]] = {}
    for record in expected:
        key = (str(record["reference"]), str(record["pin"]))
        assignments.setdefault(key, set()).add(str(record["expected_net"]))
    duplicate_assignments = [
        {"endpoint": f"{reference}.{pin}", "expected_nets": sorted(nets)}
        for (reference, pin), nets in sorted(assignments.items())
        if len(nets) > 1
    ]

    actual_pins = [pin for pin in connectivity.get("pins", []) if isinstance(pin, dict)]
    by_reference: dict[str, list[JsonRecord]] = {}
    for pin in actual_pins:
        by_reference.setdefault(str(pin.get("reference", "")), []).append(pin)

    mismatches: list[JsonRecord] = []
    compared = 0
    for record in expected:
        reference = str(record["reference"])
        selector = str(record["pin"])
        candidates = [
            pin for pin in by_reference.get(reference, []) if str(pin.get("pin", "")) == selector
        ]
        if not candidates:
            normalized_selector = _alias(selector)
            candidates = [
                pin
                for pin in by_reference.get(reference, [])
                if normalized_selector and _pin_function_alias(pin) == normalized_selector
            ]
        if len(candidates) != 1:
            mismatches.append(
                {
                    **record,
                    "actual_net": None,
                    "reason": "endpoint_missing" if not candidates else "endpoint_alias_ambiguous",
                    "candidate_count": len(candidates),
                }
            )
            continue
        compared += 1
        actual_net = _canonical_net(candidates[0].get("net"))
        if actual_net != record["expected_net"]:
            mismatches.append(
                {
                    **record,
                    "actual_net": actual_net,
                    "reason": "wrong_net",
                }
            )

    status = "pass"
    if mismatches or duplicate_assignments or uncomparable:
        status = "fail"
    return {
        "status": status,
        "requested_endpoints": len(expected),
        "compared_endpoints": compared,
        "mismatches": mismatches,
        "duplicate_assignments": duplicate_assignments,
        "uncomparable_endpoints": uncomparable,
    }
