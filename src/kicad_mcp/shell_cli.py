"""Shell-native interface for the KiCad MCP backend.

The MCP server deliberately has a broad capability surface.  This module turns
that surface into a conventional Unix command: discover with text tools, pass
arguments as JSON, and receive stable JSON or JSONL on stdout.  Diagnostics and
logs remain on stderr so pipelines do not need cleanup filters.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from kipy.geometry import Vector2
from mcp import types as mcp_types
from pydantic import BaseModel

from .circuit_spec_arrangement import arrange_circuit_spec, format_circuit_spec_arrangement
from .config import reset_config
from .connection import get_board
from .deep_inspection import (
    _project_files,
    ascii_map,
    authority_report,
    board_drc_evidence,
    connectivity_proof,
    filter_snapshot,
    format_power_loop_report,
    placement_plan,
    power_loop_placement_plan,
    power_loop_report,
    project_snapshot,
    route_plan,
    verification_report,
)
from .endpoint_net_equality import compare_compiled_endpoint_nets
from .pcb.board_access import board_footprints
from .schematic_graph_placement import (
    format_schematic_graph_placement,
    plan_schematic_graph_placement,
)
from .schematic_railway_rewire import format_railway_rewire_plan, plan_railway_rewire
from .schematic_rewire_plan import format_label_compaction_plan, plan_label_compaction
from .schematic_spatial import schematic_spatial_map
from .server import build_server
from .tools.board_file import FLOAT_PATTERN, _parse_board_footprint_blocks
from .tools.router import TOOL_CATEGORIES, available_profiles

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
KICAD_GLOBS = (
    "*.kicad_pro",
    "*.kicad_sch",
    "*.kicad_pcb",
    "*.kicad_sym",
    "*.kicad_mod",
    "*.kicad_wks",
    "*.kicad_dru",
    "*.kicad_jobset",
)
GENERATED_TREE_GLOBS = (
    "**/.git/**",
    "**/.history/**",
    "**/build/**",
    "**/output/**",
    "**/tmp/**",
)


class BackendServer(Protocol):
    """Subset of the upstream server used by the shell facade."""

    def list_tools_sync(self) -> list[mcp_types.Tool]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object: ...


def _category_index() -> dict[str, str]:
    return {
        tool_name: category
        for category, category_info in TOOL_CATEGORIES.items()
        for tool_name in category_info["tools"]
    }


def _jsonable(value: object) -> JSONValue:
    """Convert MCP/Pydantic values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Iterable):
        return [_jsonable(item) for item in value]
    return str(value)


def _content_payload(content: object) -> list[JSONValue]:
    if isinstance(content, Iterable) and not isinstance(content, str | bytes | dict):
        return [_jsonable(item) for item in content]
    return [_jsonable(content)]


def result_envelope(tool_name: str, result: object) -> dict[str, JSONValue]:
    """Normalize every FastMCP result shape into one stable shell envelope."""
    ok = True
    structured: JSONValue = None
    content: list[JSONValue] = []

    if isinstance(result, mcp_types.CallToolResult):
        ok = not bool(result.isError)
        structured = _jsonable(result.structuredContent)
        content = _content_payload(result.content)
    elif isinstance(result, tuple) and len(result) == 2:
        raw_content, raw_structured = result
        structured = _jsonable(raw_structured)
        content = _content_payload(raw_content)
    else:
        structured = _jsonable(result)

    payload: dict[str, JSONValue] = {
        "ok": ok,
        "tool": tool_name,
        "result": structured,
    }
    if content:
        payload["content"] = content
    return payload


def _tool_payload(tool: mcp_types.Tool) -> dict[str, JSONValue]:
    dumped = tool.model_dump(mode="json", exclude_none=True)
    name = str(dumped.get("name", ""))
    annotations = dumped.get("annotations", {})
    return {
        "name": name,
        "category": _category_index().get(name, "uncategorized"),
        "description": str(dumped.get("description", "")),
        "input_schema": _jsonable(dumped.get("inputSchema", {})),
        "annotations": _jsonable(annotations),
    }


def _apply_runtime_options(args: argparse.Namespace) -> None:
    if args.project_dir:
        os.environ["KICAD_MCP_PROJECT_DIR"] = str(Path(args.project_dir).expanduser().resolve())
        os.environ.setdefault("KICAD_MCP_WORKSPACE_ROOT", os.environ["KICAD_MCP_PROJECT_DIR"])
    if args.profile:
        os.environ["KICAD_MCP_PROFILE"] = args.profile
    if args.mode:
        os.environ["KICAD_MCP_OPERATING_MODE"] = args.mode
    # Catalog discovery should be stable even when the KiCad GUI is closed.
    os.environ["KICAD_MCP_FILTER_RUNTIME_TOOLS"] = "false"
    os.environ["KICAD_MCP_LOG_LEVEL"] = os.environ.get("KICAD_MCP_LOG_LEVEL", "WARNING")
    reset_config()


def _backend(args: argparse.Namespace) -> BackendServer:
    _apply_runtime_options(args)
    with contextlib.redirect_stdout(io.StringIO()):
        return cast(BackendServer, build_server(args.profile, defer_registration=False))


def catalog_records(args: argparse.Namespace) -> list[dict[str, JSONValue]]:
    server = _backend(args)
    list_sync = server.list_tools_sync
    with contextlib.redirect_stdout(io.StringIO()):
        records = [_tool_payload(tool) for tool in list_sync()]
    terms = [term.casefold() for term in (args.query or "").split() if term]
    if args.category:
        records = [record for record in records if record["category"] == args.category]
    if terms:
        records = [
            record
            for record in records
            if all(
                term
                in " ".join(
                    (
                        str(record["name"]),
                        str(record["category"]),
                        str(record["description"]),
                    )
                ).casefold()
                for term in terms
            )
        ]
    records.sort(key=lambda record: str(record["name"]))
    return records[: args.limit] if args.limit else records


def _emit_records(records: Sequence[dict[str, JSONValue]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(records, indent=2, sort_keys=True))
        return
    if output_format == "jsonl":
        for record in records:
            print(json.dumps(record, separators=(",", ":"), sort_keys=True))
        return
    if output_format == "tsv":
        for record in records:
            fields = (
                str(record.get("name", "")),
                str(record.get("category", "")),
                str(record.get("description", "")).replace("\t", " ").replace("\n", " "),
            )
            print("\t".join(fields))
        return
    for record in records:
        print(record.get("name", record.get("path", "")))


def _parse_json_object(raw: str, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def parse_call_arguments(args: argparse.Namespace) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if args.args_json:
        if args.args_json == "-":
            merged.update(_parse_json_object(sys.stdin.read(), source="stdin"))
        elif args.args_json.startswith("@"):
            path = Path(args.args_json[1:]).expanduser()
            merged.update(_parse_json_object(path.read_text(), source=str(path)))
        else:
            merged.update(_parse_json_object(args.args_json, source="--args"))
    for assignment in args.set_values:
        if "=" not in assignment:
            raise ValueError(f"--set requires KEY=VALUE, got: {assignment}")
        key, raw_value = assignment.split("=", 1)
        if not key:
            raise ValueError("--set key cannot be empty")
        try:
            merged[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            merged[key] = raw_value
    return merged


async def call_backend_tool(args: argparse.Namespace) -> dict[str, JSONValue]:
    arguments = parse_call_arguments(args)
    return await invoke_backend_tool(args, args.tool, arguments)


async def invoke_backend_tool(
    args: argparse.Namespace, tool_name: str, arguments: dict[str, Any]
) -> dict[str, JSONValue]:
    """Invoke a backend operation with protocol/data stdout protected."""
    server = _backend(args)
    with contextlib.redirect_stdout(sys.stderr):
        result = await server.call_tool(tool_name, arguments)
    return result_envelope(tool_name, result)


def _unwrap_result(payload: dict[str, JSONValue]) -> JSONValue:
    result = payload.get("result")
    if isinstance(result, dict) and set(result) == {"result"}:
        return result["result"]
    if result is not None:
        return result
    content = payload.get("content")
    if isinstance(content, list) and len(content) == 1:
        item = content[0]
        text_value = item.get("text") if isinstance(item, dict) else None
        if isinstance(text_value, str):
            text = text_value
            try:
                return _jsonable(json.loads(text))
            except json.JSONDecodeError:
                return text
        return item
    return content


def _emit_call(payload: dict[str, JSONValue], output_format: str) -> None:
    if output_format == "raw":
        value = _unwrap_result(payload)
        if isinstance(value, str):
            print(value)
        else:
            print(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return
    indent = 2 if output_format == "json" else None
    separators = None if indent else (",", ":")
    print(json.dumps(payload, indent=indent, separators=separators, sort_keys=True))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schematic_manifest(root: Path) -> dict[Path, str]:
    ignored_parts = {
        ".git",
        ".history",
        ".kicad-mcp",
        "build",
        "output",
        "tmp",
        "__pycache__",
    }
    return {
        path.relative_to(root): _hash_file(path)
        for path in sorted(root.rglob("*.kicad_sch"))
        if path.is_file() and not (set(path.relative_to(root).parts) & ignored_parts)
    }


def _resolve_edit_schematic(root: Path, sheet: str) -> Path | None:
    """Resolve a human sheet filter to one concrete child schematic file."""
    query = sheet.strip().strip("/").casefold()
    if not query:
        return None
    candidates = [root / relative for relative in _schematic_manifest(root)]
    exact = [path for path in candidates if query in {path.name.casefold(), path.stem.casefold()}]
    matches = exact or [
        path
        for path in candidates
        if query in path.name.casefold() or query in str(path.relative_to(root)).casefold()
    ]
    if not matches:
        raise ValueError(f"sheet filter matched no schematic file: {sheet}")
    if len(matches) != 1:
        relative = [str(path.relative_to(root)) for path in matches]
        raise ValueError(f"sheet filter is ambiguous: {sheet} -> {relative}")
    return matches[0]


def _erc_finding_keys(report: dict[str, Any]) -> set[str]:
    findings = report["checks"]["erc"]["findings"]
    return {
        json.dumps(finding, sort_keys=True, separators=(",", ":"))
        for finding in findings
        if finding.get("severity") == "error"
    }


def _transaction_artifacts(root: Path, requested: str, tool_name: str) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    transaction = f"{tool_name}-{uuid.uuid4().hex[:10]}"
    return root / "build" / "kicadq-transactions" / transaction


def _canonical_rewire_net(name: str) -> str:
    return name.rstrip("/").rsplit("/", 1)[-1]


def _rewire_fingerprints(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "connectivity": str(plan["connectivity_fingerprint_expectation"]["before"]),
        "positions": str(plan["position_fingerprint_expectation"]["before"]),
    }


def _rewire_tool_calls(
    plan: dict[str, Any], requested_nets: Sequence[str]
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Translate reviewed railway operations to ordinary schematic tool calls."""

    requested = {name.casefold() for name in requested_nets}
    for requested_name in requested:
        matches = {
            str(item["net"])
            for item in plan["nets"]
            if str(item["net"]).casefold() == requested_name
            or _canonical_rewire_net(str(item["net"])).casefold() == requested_name
        }
        if len(matches) > 1:
            raise ValueError(
                f"railway net selector {requested_name!r} is ambiguous: "
                + ", ".join(sorted(matches))
            )
    selected = [
        item
        for item in plan["nets"]
        if str(item["net"]).casefold() in requested
        or _canonical_rewire_net(str(item["net"])).casefold() in requested
    ]
    found = {
        value
        for item in selected
        for value in (
            str(item["net"]).casefold(),
            _canonical_rewire_net(str(item["net"])).casefold(),
        )
    }
    missing = sorted(name for name in requested if name not in found)
    if missing:
        raise ValueError(f"requested railway nets were not found: {', '.join(missing)}")
    unplanned = [str(item["net"]) for item in selected if item.get("status") != "planned"]
    if unplanned:
        raise ValueError("requested railway nets are not safely planned: " + ", ".join(unplanned))
    sheet_file = Path(str(plan["sheet"]["file"])).name
    calls: list[tuple[str, dict[str, Any]]] = []
    selected_names: list[str] = []
    seen: set[str] = set()
    for net_plan in selected:
        selected_names.append(str(net_plan["net"]))
        for operation in net_plan.get("selected_operations", []):
            op = str(operation.get("op", ""))
            call: tuple[str, dict[str, Any]] | None = None
            if op == "add_wire":
                start = operation["start_mm"]
                end = operation["end_mm"]
                call = (
                    "sch_add_wire",
                    {
                        "x1_mm": start[0],
                        "y1_mm": start[1],
                        "x2_mm": end[0],
                        "y2_mm": end[1],
                        "snap_to_grid": False,
                        "sheet_file": sheet_file,
                    },
                )
            elif op == "add_label":
                anchor = operation["anchor_mm"]
                call = (
                    "sch_add_label",
                    {
                        "name": operation["name"],
                        "x_mm": anchor[0],
                        "y_mm": anchor[1],
                        "snap_to_grid": False,
                        "sheet_file": sheet_file,
                    },
                )
            elif op == "remove_label":
                anchor = operation["anchor_mm"]
                call = (
                    "sch_delete_label",
                    {
                        "name": _canonical_rewire_net(str(operation["net"])),
                        "x_mm": anchor[0],
                        "y_mm": anchor[1],
                        "sheet_file": sheet_file,
                    },
                )
            elif op not in {"retain_label", "ensure_junction"}:
                raise ValueError(f"unsupported railway operation: {op}")
            if call is None:
                continue
            key = json.dumps(call, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                calls.append(call)
    if not calls:
        raise ValueError("selected railway nets contain no unapplied physical edits")
    return calls, sorted(set(selected_names))


async def run_staged_schematic_edit(args: argparse.Namespace) -> dict[str, Any]:
    """Apply one or more schematic tools to a clone, then promote atomically."""
    if args.mode not in {"write", "experimental"}:
        raise ValueError("edit requires --mode write or --mode experimental")
    if not args.yes:
        raise ValueError("edit requires --yes after reviewing the tool schema")
    tool_calls = getattr(args, "tool_calls", None)
    if tool_calls is None:
        tool_calls = [(args.tool, parse_call_arguments(args))]
    if not tool_calls:
        raise ValueError("staged edit requires at least one schematic tool call")
    if not all(tool.startswith("sch_") for tool, _ in tool_calls):
        raise ValueError("staged edit currently accepts only sch_* tools")
    transaction_label = getattr(args, "transaction_label", None) or args.tool
    root = Path(args.project_dir or ".").expanduser().resolve()
    before_manifest = _schematic_manifest(root)
    if not before_manifest:
        raise ValueError(f"no schematic files found in {root}")
    expected_source_sha = getattr(args, "expected_schematic_sha", None)
    if expected_source_sha:
        selected_source = _resolve_edit_schematic(root, args.sheet)
        if selected_source is None or _hash_file(selected_source) != expected_source_sha:
            raise RuntimeError(
                "selected schematic changed after railway planning; re-plan required"
            )
    artifacts = _transaction_artifacts(root, args.artifacts, transaction_label)
    before_report = verification_report(
        root,
        sheet=args.sheet,
        reference=args.reference,
        net=args.net,
        artifacts_dir=artifacts / "before",
    )
    payloads: list[dict[str, JSONValue]] = []
    changed: list[Path]
    diff_text = ""
    after_report: dict[str, Any]
    staged_rewire_fingerprints: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="kicadq-edit-") as temporary:
        stage = Path(temporary) / root.name
        shutil.copytree(
            root,
            stage,
            ignore=shutil.ignore_patterns(
                ".git",
                ".history",
                "build",
                "output",
                "tmp",
                "__pycache__",
                "*.bak",
            ),
        )
        stage_args = argparse.Namespace(**vars(args))
        stage_args.project_dir = str(stage)
        target_schematic = _resolve_edit_schematic(stage, args.sheet)
        previous_schematic = os.environ.get("KICAD_MCP_SCH_FILE")
        if target_schematic is not None:
            os.environ["KICAD_MCP_SCH_FILE"] = str(target_schematic)
        try:
            for tool_name, tool_arguments in tool_calls:
                payload = await invoke_backend_tool(stage_args, tool_name, tool_arguments)
                payloads.append(payload)
                if not payload["ok"]:
                    return {
                        "schema_version": "1.0",
                        "status": "rejected",
                        "promoted": False,
                        "reason": "backend rejected a staged edit",
                        "tool": transaction_label,
                        "failed_tool": tool_name,
                        "tool_results": payloads,
                        "artifacts": str(artifacts),
                    }
        finally:
            if previous_schematic is None:
                os.environ.pop("KICAD_MCP_SCH_FILE", None)
            else:
                os.environ["KICAD_MCP_SCH_FILE"] = previous_schematic
        stage_manifest = _schematic_manifest(stage)
        if set(stage_manifest) != set(before_manifest):
            added = sorted(str(item) for item in set(stage_manifest) - set(before_manifest))
            removed = sorted(str(item) for item in set(before_manifest) - set(stage_manifest))
            raise RuntimeError(
                "staged edit added or removed schematic files; refusing promotion "
                f"(added={added}, removed={removed})"
            )
        changed = [
            relative
            for relative in sorted(before_manifest)
            if before_manifest[relative] != stage_manifest[relative]
        ]
        if len(changed) != 1:
            raise RuntimeError(
                f"staged edit must change exactly one schematic file; changed {len(changed)}"
            )
        if target_schematic is not None:
            expected = target_schematic.relative_to(stage)
            if changed != [expected]:
                raise RuntimeError(
                    "staged edit changed a schematic other than the selected sheet: "
                    f"expected {expected}, changed {changed}"
                )
        relative = changed[0]
        original_text = (root / relative).read_text(encoding="utf-8")
        staged_text = (stage / relative).read_text(encoding="utf-8")
        diff_text = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                staged_text.splitlines(keepends=True),
                fromfile=f"before/{relative}",
                tofile=f"after/{relative}",
            )
        )
        artifacts.mkdir(parents=True, exist_ok=True)
        diff_path = artifacts / "edit.diff"
        diff_path.write_text(diff_text, encoding="utf-8")
        after_report = verification_report(
            stage,
            sheet=args.sheet,
            reference=args.reference,
            net=args.net,
            artifacts_dir=artifacts / "after",
        )
        endpoint_net_equality = compare_compiled_endpoint_nets(
            tool_calls,
            after_report["checks"]["connectivity"],
        )
        if endpoint_net_equality["status"] == "fail":
            return {
                "schema_version": "1.0",
                "status": "rejected",
                "promoted": False,
                "reason": "staged build endpoint-to-net equality failed",
                "tool": transaction_label,
                "tool_result": payloads[0] if len(payloads) == 1 else payloads,
                "tool_results": payloads,
                "changed_files": [str(relative)],
                "endpoint_net_equality": endpoint_net_equality,
                "before": before_report,
                "after": after_report,
                "artifacts": str(artifacts),
                "diff": str(diff_path),
            }
        expected_rewire_fingerprints = getattr(args, "expected_rewire_fingerprints", None)
        if expected_rewire_fingerprints:
            staged_rewire_plan = plan_railway_rewire(
                stage,
                project_snapshot(stage),
                sheet=args.sheet,
                cluster_refs=getattr(args, "rewire_references", None) or None,
            )
            staged_rewire_fingerprints = _rewire_fingerprints(staged_rewire_plan)
            if staged_rewire_fingerprints != expected_rewire_fingerprints:
                return {
                    "schema_version": "1.0",
                    "status": "rejected",
                    "promoted": False,
                    "reason": "staged railway edit changed connectivity or symbol positions",
                    "tool": transaction_label,
                    "expected_fingerprints": expected_rewire_fingerprints,
                    "staged_fingerprints": staged_rewire_fingerprints,
                    "tool_results": payloads,
                    "artifacts": str(artifacts),
                    "diff": str(diff_path),
                }
        source_ok = after_report["checks"]["source_integrity"]["status"] == "pass"
        connectivity_ok = after_report["checks"]["connectivity"]["status"] != "fail"
        new_erc_errors = sorted(_erc_finding_keys(after_report) - _erc_finding_keys(before_report))
        if not source_ok or not connectivity_ok or new_erc_errors:
            return {
                "schema_version": "1.0",
                "status": "rejected",
                "promoted": False,
                "reason": (
                    "staged verification found corruption, connectivity failure, or new ERC errors"
                ),
                "tool": transaction_label,
                "tool_result": payloads[0] if len(payloads) == 1 else payloads,
                "tool_results": payloads,
                "changed_files": [str(relative)],
                "new_erc_errors": [json.loads(item) for item in new_erc_errors],
                "endpoint_net_equality": endpoint_net_equality,
                "before": before_report,
                "after": after_report,
                "artifacts": str(artifacts),
                "diff": str(diff_path),
            }
        if _hash_file(root / relative) != before_manifest[relative]:
            raise RuntimeError(f"source changed during staged verification: {relative}")
        backup = (root / relative).read_bytes()
        promotion = (root / relative).with_suffix(f"{(root / relative).suffix}.kicadq-tmp")
        try:
            shutil.copy2(stage / relative, promotion)
            os.replace(promotion, root / relative)
            final_report = verification_report(
                root,
                sheet=args.sheet,
                reference=args.reference,
                net=args.net,
            )
            if _erc_finding_keys(final_report) != _erc_finding_keys(after_report):
                raise RuntimeError("promoted project does not reproduce staged ERC evidence")
            if final_report["checks"]["connectivity"]["status"] == "fail":
                raise RuntimeError("promoted project failed the final connectivity filter gate")
            if expected_rewire_fingerprints:
                final_rewire_plan = plan_railway_rewire(
                    root,
                    project_snapshot(root),
                    sheet=args.sheet,
                    cluster_refs=getattr(args, "rewire_references", None) or None,
                )
                if _rewire_fingerprints(final_rewire_plan) != expected_rewire_fingerprints:
                    raise RuntimeError(
                        "promoted project does not preserve railway connectivity/positions"
                    )
        except Exception:
            (root / relative).write_bytes(backup)
            if promotion.exists():
                promotion.unlink()
            raise

    return {
        "schema_version": "1.0",
        "status": "pass" if after_report["status"] == "pass" else "review",
        "promoted": True,
        "tool": transaction_label,
        "tool_result": payloads[0] if len(payloads) == 1 else payloads,
        "tool_results": payloads,
        "changed_files": [str(item) for item in changed],
        "before": before_report,
        "after": after_report,
        "artifacts": str(artifacts),
        "diff": str(artifacts / "edit.diff"),
        "diff_lines": len(diff_text.splitlines()),
        "endpoint_net_equality": endpoint_net_equality,
        "rewire_fingerprints": staged_rewire_fingerprints,
    }


async def run_native_board_transaction(
    args: argparse.Namespace,
    operations: list[tuple[str, dict[str, Any]]],
    *,
    label: str,
) -> dict[str, Any]:
    """Keep native diagnostics off the JSON/JSONL protocol stream."""
    with contextlib.redirect_stdout(sys.stderr):
        return await _run_native_board_transaction(args, operations, label=label)


def _drc_regression_details(
    before_drc: dict[str, Any],
    staged_drc: dict[str, Any],
) -> dict[str, Any]:
    """Compare stable physical findings and unrouted counts separately.

    KiCad chooses representative pairs for an unrouted net. Moving a footprint
    can therefore replace many ``unconnected_items`` keys even when the number
    of unrouted items is unchanged. Treating those pair identities as new DRC
    failures makes safe placement candidates impossible to verify.
    """
    new_keys = sorted(set(staged_drc["finding_keys"]) - set(before_drc["finding_keys"]))
    new_findings = [json.loads(item) for item in new_keys]
    new_physical = [
        finding
        for finding in new_findings
        if finding.get("type") != "unconnected_items" and finding.get("kind") != "unconnected"
    ]
    before_summary = cast(dict[str, int], before_drc["summary"])
    staged_summary = cast(dict[str, int], staged_drc["summary"])
    before_unconnected = before_summary.get("unconnected_items", 0)
    staged_unconnected = staged_summary.get("unconnected_items", 0)
    return {
        "new_findings": new_findings,
        "new_physical_findings": new_physical,
        "before_unconnected_items": before_unconnected,
        "staged_unconnected_items": staged_unconnected,
        "unconnected_increase": max(0, staged_unconnected - before_unconnected),
        "regressed": bool(new_physical or staged_unconnected > before_unconnected),
    }


def _run_offline_placement_candidate(
    *,
    root: Path,
    before_content: str,
    operations: list[tuple[str, dict[str, Any]]],
    artifacts: Path,
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Build and DRC a placement candidate when live KiCad cannot authorize commit."""
    staged_content = before_content
    for _tool_name, arguments in operations:
        staged_content = _apply_footprint_batch_to_board_content(staged_content, arguments)
        _verify_staged_footprint_batch(staged_content, arguments)
        _verify_rigid_footprint_children(before_content, staged_content, arguments)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "before.kicad_pcb").write_text(before_content, encoding="utf-8")
    (artifacts / "staged.kicad_pcb").write_text(staged_content, encoding="utf-8")
    diff_text = "".join(
        difflib.unified_diff(
            before_content.splitlines(keepends=True),
            staged_content.splitlines(keepends=True),
            fromfile="before.kicad_pcb",
            tofile="staged.kicad_pcb",
        )
    )
    (artifacts / "edit.diff").write_text(diff_text, encoding="utf-8")
    before_drc = board_drc_evidence(root, board_content=before_content)
    staged_drc = board_drc_evidence(root, board_content=staged_content)
    regressions = _drc_regression_details(before_drc, staged_drc)
    if regressions["regressed"]:
        return {
            "schema_version": "1.0",
            "status": "rejected",
            "committed": False,
            "candidate_verified": False,
            "reason": "offline staged board introduces new DRC findings",
            "before_drc": before_drc,
            "staged_drc": staged_drc,
            "regressions": regressions,
            "authority": authority,
            "artifacts": str(artifacts),
            "diff": str(artifacts / "edit.diff"),
        }
    return {
        "schema_version": "1.0",
        "status": "blocked",
        "committed": False,
        "candidate_verified": True,
        "reason": "offline candidate passes DRC regression gate; native authority is unavailable",
        "before_drc": before_drc,
        "staged_drc": staged_drc,
        "regressions": regressions,
        "authority": authority,
        "artifacts": str(artifacts),
        "diff": str(artifacts / "edit.diff"),
        "diff_lines": len(diff_text.splitlines()),
    }


async def _run_native_board_transaction(
    args: argparse.Namespace,
    operations: list[tuple[str, dict[str, Any]]],
    *,
    label: str,
) -> dict[str, Any]:
    """Apply live PCB mutations as one IPC commit and drop on DRC regression."""
    root = Path(args.project_dir or ".").expanduser().resolve()
    authority = authority_report(root)
    if not authority["policy"]["board_mutation_allowed"]:
        placement_only = bool(operations) and all(
            name == "_native_move_footprints_batch" for name, _arguments in operations
        )
        if placement_only:
            artifacts = _transaction_artifacts(root, args.artifacts, label)
            _project, _schematic, board_path = _project_files(root)
            return _run_offline_placement_candidate(
                root=root,
                before_content=board_path.read_text(encoding="utf-8"),
                operations=operations,
                artifacts=artifacts,
                authority=authority,
            )
        return {
            "schema_version": "1.0",
            "status": "blocked",
            "committed": False,
            "reason": "native board authority is unavailable or live/disk state differs",
            "authority": authority,
        }
    if not operations:
        return {
            "schema_version": "1.0",
            "status": "pass",
            "committed": False,
            "reason": "no board mutations were necessary",
            "authority": authority,
        }
    artifacts = _transaction_artifacts(root, args.artifacts, label)
    artifacts.mkdir(parents=True, exist_ok=True)
    board = get_board()
    before_content = board.get_as_string()
    (artifacts / "before.kicad_pcb").write_text(before_content, encoding="utf-8")
    before_drc = board_drc_evidence(root, board_content=before_content)
    if _placement_batch_requires_guarded_file(before_content, operations):
        return _run_guarded_file_placement_transaction(
            root=root,
            board=board,
            before_content=before_content,
            before_drc=before_drc,
            operations=operations,
            artifacts=artifacts,
            label=label,
        )
    results: list[dict[str, JSONValue]] = []
    commit_active = False
    commit: object | None = None

    def drop_commit() -> None:
        if commit is None:
            board.drop_commit()
        else:
            board.drop_commit(commit)

    def push_commit() -> None:
        if commit is None:
            board.push_commit()
        else:
            board.push_commit(commit, label)

    try:
        commit = board.begin_commit()
        commit_active = True
        offline_candidate = before_content
        offline_candidate_complete = True
        for tool_name, arguments in operations:
            if tool_name == "_native_move_footprints_batch":
                result = _native_move_footprints_batch(board, arguments)
                offline_candidate = _apply_footprint_batch_to_board_content(
                    offline_candidate,
                    arguments,
                )
            else:
                offline_candidate_complete = False
                result = await invoke_backend_tool(args, tool_name, arguments)
            results.append(result)
            if not result["ok"]:
                drop_commit()
                commit_active = False
                return {
                    "schema_version": "1.0",
                    "status": "rejected",
                    "committed": False,
                    "reason": f"backend rejected {tool_name}",
                    "operations": results,
                    "artifacts": str(artifacts),
                }
        staged_content = offline_candidate if offline_candidate_complete else board.get_as_string()
        for tool_name, arguments in operations:
            if tool_name == "_native_move_footprints_batch":
                _verify_staged_footprint_batch(staged_content, arguments)
        (artifacts / "staged.kicad_pcb").write_text(staged_content, encoding="utf-8")
        diff_text = "".join(
            difflib.unified_diff(
                before_content.splitlines(keepends=True),
                staged_content.splitlines(keepends=True),
                fromfile="before.kicad_pcb",
                tofile="staged.kicad_pcb",
            )
        )
        (artifacts / "edit.diff").write_text(diff_text, encoding="utf-8")
        staged_drc = board_drc_evidence(root, board_content=staged_content)
        regressions = _drc_regression_details(before_drc, staged_drc)
        before_summary = cast(dict[str, int], before_drc["summary"])
        staged_summary = cast(dict[str, int], staged_drc["summary"])
        # KiCad reports physical DRC violations and unrouted items in separate
        # arrays; never subtract one from the other.
        before_physical_violations = before_summary.get("violations", 0)
        staged_physical_violations = staged_summary.get("violations", 0)
        transform_only_placement = (
            label == "placement"
            and offline_candidate_complete
            and all(tool_name == "_native_move_footprints_batch" for tool_name, _ in operations)
        )
        placement_improved = (
            transform_only_placement and staged_physical_violations < before_physical_violations
        )
        if regressions["regressed"]:
            drop_commit()
            commit_active = False
            return {
                "schema_version": "1.0",
                "status": "rejected",
                "committed": False,
                "reason": "staged board introduces new DRC findings",
                "new_drc_findings": regressions["new_findings"],
                "regressions": regressions,
                "before_drc": before_drc,
                "staged_drc": staged_drc,
                "operations": results,
                "artifacts": str(artifacts),
                "diff": str(artifacts / "edit.diff"),
            }
        push_commit()
        commit_active = False
        placement_operations = [
            arguments
            for tool_name, arguments in operations
            if tool_name == "_native_move_footprints_batch"
        ]
        try:
            if placement_operations:
                committed_content = board.get_as_string()
            for arguments in placement_operations:
                _verify_staged_footprint_batch(committed_content, arguments)
                _verify_rigid_footprint_children(
                    staged_content,
                    committed_content,
                    arguments,
                )
        except Exception:
            board.revert()
            raise
        try:
            board.save()
        except Exception:
            board.revert()
            raise
        final_authority = authority_report(root)
        if not final_authority["policy"]["board_mutation_allowed"]:
            raise RuntimeError("saved board failed the live/disk semantic synchronization gate")
        return {
            "schema_version": "1.0",
            "status": "pass",
            "committed": True,
            "operations": results,
            "before_drc": before_drc,
            "staged_drc": staged_drc,
            "before_physical_violations": before_physical_violations,
            "staged_physical_violations": staged_physical_violations,
            "accepted_placement_improvement": placement_improved,
            "new_drc_findings_after_improvement": [],
            "regressions": regressions,
            "authority": final_authority,
            "artifacts": str(artifacts),
            "diff": str(artifacts / "edit.diff"),
            "diff_lines": len(diff_text.splitlines()),
        }
    except Exception:
        if commit_active:
            with contextlib.suppress(Exception):
                drop_commit()
        raise


def _native_move_footprints_batch(
    board: object,
    arguments: dict[str, Any],
) -> dict[str, JSONValue]:
    """Apply one placement plan through a single native IPC update call."""
    raw_placements = arguments.get("placements", [])
    if not isinstance(raw_placements, list):
        return {
            "ok": False,
            "tool": "_native_move_footprints_batch",
            "error": "placements must be a list",
        }
    footprints = board_footprints(board)
    by_reference = {
        str(footprint.reference_field.text.value): footprint for footprint in footprints
    }
    missing: list[str] = []
    resolved: list[tuple[str, object, float, float, float]] = []
    changed: list[object] = []
    for raw in raw_placements:
        if not isinstance(raw, dict):
            return {
                "ok": False,
                "tool": "_native_move_footprints_batch",
                "error": "every placement must be an object",
            }
        reference = str(raw.get("reference", ""))
        footprint = by_reference.get(reference)
        if footprint is None:
            missing.append(reference)
            continue
        x_mm = float(raw["x_mm"])
        y_mm = float(raw["y_mm"])
        rotation_deg = float(raw.get("rotation_deg", 0.0))
        resolved.append((reference, footprint, x_mm, y_mm, rotation_deg))
    if missing:
        return {
            "ok": False,
            "tool": "_native_move_footprints_batch",
            "error": "missing footprint references: " + ", ".join(sorted(missing)),
        }
    rotation_changes: list[str] = []
    for reference, footprint, _x_mm, _y_mm, rotation_deg in resolved:
        orientation = getattr(footprint, "orientation", None)
        if orientation is None:
            orientation = getattr(footprint, "angle", None)
        current_deg = float(getattr(orientation, "degrees", orientation))
        delta = (current_deg - rotation_deg + 180.0) % 360.0 - 180.0
        if abs(delta) > 1e-3:
            rotation_changes.append(f"{reference}:{current_deg:g}->{rotation_deg:g}")
    if rotation_changes:
        return {
            "ok": False,
            "tool": "_native_move_footprints_batch",
            "error": (
                "native footprint orientation is not a rigid child transform; "
                "rotation changes require the guarded file-level rigid-rotation path: "
                + ", ".join(rotation_changes)
            ),
        }
    for _reference, footprint, x_mm, y_mm, _rotation_deg in resolved:
        footprint.position = Vector2.from_xy_mm(x_mm, y_mm)
        changed.append(footprint)
    board.update_items(changed)
    return {
        "ok": True,
        "tool": "_native_move_footprints_batch",
        "result": {"moved": len(changed)},
    }


def _placement_batch_requires_guarded_file(
    board_content: str,
    operations: list[tuple[str, dict[str, Any]]],
) -> bool:
    """Use rigid file transforms for rotations or footprints owning embedded zones."""
    if not operations or any(name != "_native_move_footprints_batch" for name, _ in operations):
        return False
    footprints = _parse_board_footprint_blocks(board_content)
    for _name, arguments in operations:
        placements = arguments.get("placements", [])
        if not isinstance(placements, list):
            continue
        for raw in placements:
            if not isinstance(raw, dict):
                continue
            footprint = footprints.get(str(raw.get("reference", "")))
            if footprint is None:
                continue
            requested_rotation = float(raw.get("rotation_deg", footprint["rotation"]) or 0.0)
            current_rotation = float(footprint.get("rotation", 0.0) or 0.0)
            rotation_delta = (requested_rotation - current_rotation + 180.0) % 360.0 - 180.0
            if abs(rotation_delta) > 1e-3 or "(zone" in str(footprint["block"]):
                return True
    return False


def _run_guarded_file_placement_transaction(
    *,
    root: Path,
    board: object,
    before_content: str,
    before_drc: dict[str, Any],
    operations: list[tuple[str, dict[str, Any]]],
    artifacts: Path,
    label: str,
) -> dict[str, Any]:
    """Apply root-only transforms atomically when KiCad IPC cannot move rigidly."""
    staged_content = before_content
    moved = 0
    for _tool_name, arguments in operations:
        staged_content = _apply_footprint_batch_to_board_content(staged_content, arguments)
        placements = arguments.get("placements", [])
        moved += len(placements) if isinstance(placements, list) else 0
    for _tool_name, arguments in operations:
        _verify_staged_footprint_batch(staged_content, arguments)
        _verify_rigid_footprint_children(before_content, staged_content, arguments)

    (artifacts / "staged.kicad_pcb").write_text(staged_content, encoding="utf-8")
    diff_text = "".join(
        difflib.unified_diff(
            before_content.splitlines(keepends=True),
            staged_content.splitlines(keepends=True),
            fromfile="before.kicad_pcb",
            tofile="staged.kicad_pcb",
        )
    )
    (artifacts / "edit.diff").write_text(diff_text, encoding="utf-8")
    staged_drc = board_drc_evidence(root, board_content=staged_content)
    regressions = _drc_regression_details(before_drc, staged_drc)
    before_summary = cast(dict[str, int], before_drc["summary"])
    staged_summary = cast(dict[str, int], staged_drc["summary"])
    before_physical = before_summary.get("violations", 0)
    staged_physical = staged_summary.get("violations", 0)
    placement_improved = staged_physical < before_physical
    if regressions["regressed"]:
        return {
            "schema_version": "1.0",
            "status": "rejected",
            "committed": False,
            "reason": "staged board introduces new DRC findings",
            "new_drc_findings": regressions["new_findings"],
            "regressions": regressions,
            "before_drc": before_drc,
            "staged_drc": staged_drc,
            "artifacts": str(artifacts),
            "diff": str(artifacts / "edit.diff"),
        }

    _project, _schematic, board_path = _project_files(root)
    temporary_path = board_path.with_name(f".{board_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(staged_content, encoding="utf-8")
        temporary_path.replace(board_path)
        board.revert()
        final_authority = authority_report(root)
        if not final_authority["policy"]["board_mutation_allowed"]:
            raise RuntimeError("guarded placement failed the live/disk synchronization gate")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        board_path.write_text(before_content, encoding="utf-8")
        with contextlib.suppress(Exception):
            board.revert()
        raise

    return {
        "schema_version": "1.0",
        "status": "pass",
        "committed": True,
        "write_authority": "guarded-rigid-file",
        "reason": "embedded footprint zones require a rigid root-only transform",
        "operations": [
            {
                "ok": True,
                "tool": "_native_move_footprints_batch",
                "result": {"moved": moved},
            }
        ],
        "before_drc": before_drc,
        "staged_drc": staged_drc,
        "before_physical_violations": before_physical,
        "staged_physical_violations": staged_physical,
        "accepted_placement_improvement": placement_improved,
        "new_drc_findings_after_improvement": [],
        "regressions": regressions,
        "authority": final_authority,
        "artifacts": str(artifacts),
        "diff": str(artifacts / "edit.diff"),
        "diff_lines": len(diff_text.splitlines()),
        "label": label,
    }


def _verify_staged_footprint_batch(
    board_content: str,
    arguments: dict[str, Any],
    *,
    tolerance_mm: float = 1e-3,
    tolerance_deg: float = 1e-3,
) -> None:
    """Verify one native batch against KiCad's staged serialized board."""
    footprints = _parse_board_footprint_blocks(board_content)
    raw_placements = arguments.get("placements", [])
    if not isinstance(raw_placements, list):
        raise RuntimeError("staged placement verification requires a placements list")
    for raw in raw_placements:
        if not isinstance(raw, dict):
            raise RuntimeError("staged placement verification received a non-object placement")
        reference = str(raw.get("reference", ""))
        footprint = footprints.get(reference)
        if footprint is None:
            raise RuntimeError(
                f"staged placement verification could not reload footprint '{reference}'"
            )
        requested_x = float(raw["x_mm"])
        requested_y = float(raw["y_mm"])
        actual_x = float(footprint["x_mm"])
        actual_y = float(footprint["y_mm"])
        if abs(actual_x - requested_x) > tolerance_mm or abs(actual_y - requested_y) > tolerance_mm:
            raise RuntimeError(
                f"staged placement verification failed for '{reference}': requested "
                f"({requested_x:.4f}, {requested_y:.4f}) mm, observed "
                f"({actual_x:.4f}, {actual_y:.4f}) mm"
            )
        requested_rotation = float(raw.get("rotation_deg", 0.0))
        actual_rotation = float(footprint.get("rotation", 0.0) or 0.0)
        rotation_delta = (actual_rotation - requested_rotation + 180.0) % 360.0 - 180.0
        if abs(rotation_delta) > tolerance_deg:
            raise RuntimeError(
                f"staged placement verification failed for '{reference}': requested "
                f"{requested_rotation:.4f} degrees, observed {actual_rotation:.4f} degrees"
            )


def _verify_rigid_footprint_children(
    expected_content: str,
    observed_content: str,
    arguments: dict[str, Any],
) -> None:
    """Reject IPC moves that mutate any footprint child geometry.

    KiCad 10 IPC currently translates some embedded footprint zones when only
    the parent position is updated.  Position-only placement must be a rigid
    transform: after normalizing the requested root ``(at ...)``, every child
    token must remain byte-identical to the offline candidate.
    """
    expected = _parse_board_footprint_blocks(expected_content)
    observed = _parse_board_footprint_blocks(observed_content)
    root_at_pattern = re.compile(
        rf"(?P<indent>^[ \t]*)\(at\s+{FLOAT_PATTERN}\s+{FLOAT_PATTERN}"
        rf"(?:\s+{FLOAT_PATTERN})?\)",
        flags=re.MULTILINE,
    )

    def without_root_transform(block: str) -> str:
        return root_at_pattern.sub(r"\g<indent>(at <ROOT_TRANSFORM>)", block, count=1)

    changed: list[str] = []
    raw_placements = arguments.get("placements", [])
    if not isinstance(raw_placements, list):
        raise RuntimeError("rigid footprint verification requires a placements list")
    for raw in raw_placements:
        if not isinstance(raw, dict):
            raise RuntimeError("rigid footprint verification received a non-object placement")
        reference = str(raw.get("reference", ""))
        expected_footprint = expected.get(reference)
        observed_footprint = observed.get(reference)
        if expected_footprint is None or observed_footprint is None:
            changed.append(reference)
            continue
        if without_root_transform(str(expected_footprint["block"])) != without_root_transform(
            str(observed_footprint["block"])
        ):
            changed.append(reference)
    if changed:
        raise RuntimeError(
            "native footprint move changed child geometry; board reverted: "
            + ", ".join(sorted(changed))
        )


def _apply_footprint_batch_to_board_content(
    board_content: str,
    arguments: dict[str, Any],
) -> str:
    """Create a DRC-able placement candidate without exposing unpushed IPC state."""
    footprints = _parse_board_footprint_blocks(board_content)
    raw_placements = arguments.get("placements", [])
    if not isinstance(raw_placements, list):
        raise RuntimeError("offline placement candidate requires a placements list")
    replacements: list[tuple[int, int, str]] = []
    root_at_pattern = re.compile(
        rf"(?P<indent>^[ \t]*)\(at\s+{FLOAT_PATTERN}\s+{FLOAT_PATTERN}"
        rf"(?:\s+{FLOAT_PATTERN})?\)",
        flags=re.MULTILINE,
    )
    for raw in raw_placements:
        if not isinstance(raw, dict):
            raise RuntimeError("offline placement candidate received a non-object placement")
        reference = str(raw.get("reference", ""))
        footprint = footprints.get(reference)
        if footprint is None:
            raise RuntimeError(
                f"offline placement candidate could not reload footprint '{reference}'"
            )
        block = str(footprint["block"])
        match = root_at_pattern.search(block)
        if match is None:
            raise RuntimeError(
                f"offline placement candidate found no root transform for '{reference}'"
            )
        x_mm = float(raw["x_mm"])
        y_mm = float(raw["y_mm"])
        rotation_deg = float(raw.get("rotation_deg", 0.0))
        replacement = f"{match.group('indent')}(at {x_mm:.4f} {y_mm:.4f} {rotation_deg:.4f})"
        updated_block = block[: match.start()] + replacement + block[match.end() :]
        replacements.append((int(footprint["start"]), int(footprint["end"]), updated_block))

    candidate = board_content
    for start, end, updated_block in sorted(replacements, reverse=True):
        candidate = candidate[:start] + updated_block + candidate[end:]
    _verify_staged_footprint_batch(candidate, arguments)
    return candidate


def _rg_binary() -> str:
    binary = shutil.which("rg")
    if binary is None:
        raise RuntimeError("ripgrep (rg) is required for `kicadq files` and `kicadq grep`")
    return binary


def _project_roots(args: argparse.Namespace) -> list[str]:
    roots = list(getattr(args, "paths", []) or [])
    if not roots:
        roots = [args.project_dir or "."]
    return [str(Path(root).expanduser()) for root in roots]


def file_records(args: argparse.Namespace) -> Iterator[dict[str, JSONValue]]:
    command = [_rg_binary(), "--files", "--hidden", "--no-messages"]
    for pattern in KICAD_GLOBS:
        command.extend(("--glob", pattern))
    for pattern in GENERATED_TREE_GLOBS:
        command.extend(("--glob", f"!{pattern}"))
    command.extend(_project_roots(args))
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode not in (0, 1):
        raise RuntimeError(process.stderr.strip() or "rg file discovery failed")
    for raw_path in sorted(filter(None, process.stdout.splitlines())):
        path = Path(raw_path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        yield {"path": str(path), "kind": path.suffix.removeprefix("."), "bytes": size}


def grep_records(args: argparse.Namespace) -> Iterator[dict[str, JSONValue]]:
    command = [
        _rg_binary(),
        "--json",
        "--line-number",
        "--column",
        "--color",
        "never",
        "--no-messages",
    ]
    if args.fixed_strings:
        command.append("--fixed-strings")
    if args.ignore_case:
        command.append("--ignore-case")
    for pattern in KICAD_GLOBS:
        command.extend(("--glob", pattern))
    for pattern in GENERATED_TREE_GLOBS:
        command.extend(("--glob", f"!{pattern}"))
    command.extend(("--", args.pattern))
    command.extend(_project_roots(args))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.stdout is None:  # pragma: no cover - Popen contract guard
        process.kill()
        raise RuntimeError("rg search did not expose stdout")
    emitted = 0
    for line in process.stdout:
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        path = data["path"].get("text") or data["path"].get("bytes", "")
        line_text = data["lines"].get("text", "").rstrip("\r\n")
        submatches = data.get("submatches") or [{}]
        first = submatches[0]
        record: dict[str, JSONValue] = {
            "path": path,
            "line": data.get("line_number"),
            "column": int(first.get("start", 0)) + 1,
            "text": line_text,
            "match": first.get("match", {}).get("text", ""),
        }
        yield record
        emitted += 1
        if args.limit and emitted >= args.limit:
            process.terminate()
            break
    _, stderr = process.communicate()
    if process.returncode not in (0, 1, -15):
        raise RuntimeError(stderr.strip() or "rg search failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kicadq",
        description="Query and automate KiCad through a pipe-friendly CLI.",
    )
    parser.add_argument("--project-dir", "-C", help="active KiCad project/search directory")
    parser.add_argument(
        "--profile", choices=available_profiles(), default="full", help="backend capability profile"
    )
    parser.add_argument(
        "--mode",
        choices=("readonly", "write", "manufacturing", "experimental"),
        default="readonly",
        help="risk-oriented execution mode",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    tools = subcommands.add_parser("tools", help="list and search the backend command catalog")
    tools.add_argument("query", nargs="?", default="", help="terms matched across names and docs")
    tools.add_argument("--category", help="restrict results to one category")
    tools.add_argument("--limit", type=int, default=0, help="maximum results; zero means unlimited")
    tools.add_argument("--format", choices=("names", "json", "jsonl", "tsv"), default="names")

    schema = subcommands.add_parser("schema", help="print one backend command schema")
    schema.add_argument("tool")

    call = subcommands.add_parser("call", help="invoke one backend command")
    call.add_argument("tool")
    call.add_argument(
        "--args",
        dest="args_json",
        default="",
        help="JSON object, '-' for stdin, or @path/to/file.json",
    )
    call.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="merge one JSON value (plain strings are accepted); repeatable",
    )
    call.add_argument("--format", choices=("json", "jsonl", "raw"), default="json")

    edit = subcommands.add_parser(
        "edit", help="stage, diff, verify, and atomically promote one schematic mutation"
    )
    edit.add_argument("tool", help="one sch_* backend tool")
    edit.add_argument(
        "--args",
        dest="args_json",
        default="",
        help="JSON object, '-' for stdin, or @path/to/file.json",
    )
    edit.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="merge one JSON value; repeatable",
    )
    edit.add_argument("--sheet", required=True, help="sheet substring used for verification")
    edit.add_argument("--net", default="", help="optional net filter for connectivity proof")
    edit.add_argument("--ref", dest="reference", default="", help="optional reference filter")
    edit.add_argument("--artifacts", default="", metavar="DIR")
    edit.add_argument("--yes", action="store_true", help="confirm promotion after staged checks")
    edit.add_argument("--format", choices=("json", "jsonl"), default="json")

    files = subcommands.add_parser("files", help="emit a KiCad project file manifest")
    files.add_argument("paths", nargs="*")
    files.add_argument("--format", choices=("json", "jsonl", "names"), default="jsonl")

    grep = subcommands.add_parser("grep", help="search KiCad source files with ripgrep")
    grep.add_argument("pattern")
    grep.add_argument("paths", nargs="*")
    grep.add_argument("--fixed-strings", "-F", action="store_true")
    grep.add_argument("--ignore-case", "-i", action="store_true")
    grep.add_argument("--limit", type=int, default=0)
    grep.add_argument("--format", choices=("json", "jsonl", "lines"), default="jsonl")

    inspect = subcommands.add_parser(
        "inspect", help="emit one deep hierarchical schematic and PCB snapshot"
    )
    inspect.add_argument("--sheet", default="", help="filter by hierarchical sheet substring")
    inspect.add_argument("--net", default="", help="filter by net-name substring")
    inspect.add_argument("--ref", dest="reference", default="", help="filter by exact reference")
    inspect.add_argument("--format", choices=("json", "jsonl"), default="json")

    prove = subcommands.add_parser(
        "prove", help="emit pin-to-net-to-peer connectivity evidence from KiCad's netlist"
    )
    prove.add_argument("--sheet", default="", help="filter by hierarchical sheet substring")
    prove.add_argument("--net", default="", help="filter by net-name substring")
    prove.add_argument("--ref", dest="reference", default="", help="filter by exact reference")
    prove.add_argument("--format", choices=("json", "jsonl"), default="json")

    verify = subcommands.add_parser(
        "verify", help="bundle source integrity, connectivity proof, ERC, and SVG evidence"
    )
    verify.add_argument("--sheet", default="", help="filter by hierarchical sheet substring")
    verify.add_argument("--net", default="", help="filter by net-name substring")
    verify.add_argument("--ref", dest="reference", default="", help="filter by exact reference")
    verify.add_argument(
        "--artifacts",
        default="",
        metavar="DIR",
        help="render selected schematic pages as SVG into DIR",
    )
    verify.add_argument("--format", choices=("json", "jsonl"), default="json")

    backend = subcommands.add_parser(
        "backend", help="show native IPC, CLI, and file-fallback authority for this project"
    )
    backend.add_argument("--format", choices=("json", "jsonl"), default="json")

    map_command = subcommands.add_parser(
        "map", help="render a zoomable-in-spirit ASCII/Unicode project map"
    )
    map_command.add_argument("--zoom", type=int, choices=range(4), default=0)
    map_command.add_argument("--width", type=int, default=100)
    map_command.add_argument(
        "--view",
        choices=("semantic", "spatial"),
        default="semantic",
        help="preserve the semantic map or render saved schematic geometry",
    )
    map_command.add_argument(
        "--height",
        type=int,
        default=32,
        help="spatial canvas height (bounded to 12..80 rows)",
    )
    map_command.add_argument(
        "--center-ref",
        default="",
        help="center spatial zoom on a component reference",
    )
    map_command.add_argument("--sheet", default="")
    map_command.add_argument("--net", default="")
    map_command.add_argument("--ref", dest="reference", default="")

    compact_labels = subcommands.add_parser(
        "plan-labels",
        help="plan read-only local same-net label compaction around one component",
    )
    compact_labels.add_argument("--ref", dest="reference", required=True)
    compact_labels.add_argument("--sheet", default="")
    compact_labels.add_argument("--radius", type=float, default=35.0, dest="radius_mm")
    compact_labels.add_argument("--format", choices=("json", "text"), default="text")

    plan_rewire = subcommands.add_parser(
        "plan-rewire",
        help="plan exact-pin schematic railway wiring without modifying CAD files",
    )
    plan_rewire.add_argument(
        "--sheet",
        required=True,
        help="select exactly one hierarchical sheet by name or file substring",
    )
    plan_rewire.add_argument(
        "--ref",
        dest="references",
        action="append",
        default=[],
        help="include one cluster reference; repeatable, omitted means the whole sheet",
    )
    plan_rewire.add_argument("--format", choices=("json", "text"), default="text")

    apply_rewire = subcommands.add_parser(
        "apply-rewire",
        help="atomically apply selected exact-pin railway nets after staged verification",
    )
    apply_rewire.add_argument("--sheet", required=True)
    apply_rewire.add_argument(
        "--ref",
        dest="references",
        action="append",
        default=[],
        help="limit planning to one cluster reference; repeatable",
    )
    apply_rewire.add_argument(
        "--net",
        dest="rewire_nets",
        action="append",
        required=True,
        help="apply one safely planned net; repeatable (full or local net name)",
    )
    apply_rewire.add_argument("--artifacts", default="")
    apply_rewire.add_argument("--yes", action="store_true")
    apply_rewire.add_argument("--format", choices=("json", "text"), default="text")

    plan_schematic = subcommands.add_parser(
        "plan-schematic",
        help="dry-run graph-ranked schematic symbol placement and railway wiring",
    )
    plan_schematic.add_argument("--sheet", required=True)
    plan_schematic.add_argument(
        "--fix",
        dest="fixed_references",
        action="append",
        default=[],
        help="preserve one symbol center; repeatable",
    )
    plan_schematic.add_argument(
        "--candidates",
        type=int,
        choices=range(1, 7),
        default=6,
        dest="candidate_count",
        help="deterministic candidate budget, including saved identity",
    )
    plan_schematic.add_argument("--format", choices=("json", "text"), default="text")

    arrange_spec = subcommands.add_parser(
        "arrange-spec",
        help="fill missing circuit-spec coordinates with a read-only railway layout",
    )
    arrange_spec.add_argument("path", help="circuit-spec JSON path, or - for stdin")
    arrange_spec.add_argument(
        "--candidates",
        type=int,
        choices=range(1, 4),
        default=3,
        dest="candidate_count",
        help="deterministic fresh-layout candidate budget",
    )
    arrange_spec.add_argument(
        "--reflow",
        action="store_true",
        help="replace all saved symbol coordinates with the selected fresh layout",
    )
    arrange_spec.add_argument("--format", choices=("report", "json", "spec"), default="report")

    route = subcommands.add_parser(
        "route", help="plan or apply a conservative PCB route for one net"
    )
    route.add_argument("net")
    route.add_argument("--layer", default="F.Cu")
    route.add_argument("--width", type=float, default=0.25, dest="width_mm")
    route.add_argument("--clearance", type=float, default=0.5, dest="clearance_mm")
    route.add_argument("--allow-critical", action="store_true")
    route.add_argument("--apply", action="store_true")
    route.add_argument("--yes", action="store_true", help="confirm route mutation")
    route.add_argument("--artifacts", default="", metavar="DIR")
    route.add_argument("--format", choices=("json", "jsonl"), default="json")

    place = subcommands.add_parser(
        "place", help="plan connectivity-aware PCB placement from existing footprints"
    )
    place.add_argument(
        "--fix",
        dest="fixed_references",
        action="append",
        default=[],
        help="hold one mechanical anchor reference fixed; repeatable",
    )
    place.add_argument(
        "--anchor",
        dest="anchors",
        action="append",
        default=[],
        metavar="REF,EDGE,OFFSET[,ROTATION]",
        help=(
            "anchor a footprint at top/right/bottom/left; offset is measured from "
            "the board's left or top edge; repeatable"
        ),
    )
    place.add_argument(
        "--at",
        dest="absolute_anchors",
        action="append",
        default=[],
        metavar="REF,X,Y[,ROTATION]",
        help="anchor a footprint at an absolute board coordinate; repeatable",
    )
    place.add_argument(
        "--keepout",
        dest="keepout_regions",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="absolute board keepout rectangle in mm; repeatable",
    )
    place.add_argument(
        "--cluster",
        dest="cluster_regions",
        action="append",
        default=[],
        metavar="SHEET,X1,Y1,X2,Y2",
        help="constrain one hierarchical sheet to an absolute board region; repeatable",
    )
    place.add_argument("--margin", type=float, default=3.0, dest="margin_mm")
    place.add_argument("--iterations", type=int, default=300)
    place.add_argument("--grid", type=float, default=0.5, dest="grid_mm")
    place.add_argument("--seed", type=int, default=42)
    place.add_argument(
        "--spec",
        default=".kicad-mcp/project_spec.json",
        help="optional project-relative intent JSON supplying decoupling proximity pairs",
    )
    place.add_argument("--apply", action="store_true")
    place.add_argument("--yes", action="store_true", help="confirm placement mutation")
    place.add_argument("--artifacts", default="", metavar="DIR")
    place.add_argument("--format", choices=("json", "jsonl"), default="json")

    power_loops = subcommands.add_parser(
        "power-loops",
        help="inspect declared decouplers using actual rail-pad and ground-pad geometry",
    )
    power_loops.add_argument(
        "--spec",
        default=".kicad-mcp/project_spec.json",
        help="project-relative design-intent JSON containing decoupling_pairs",
    )
    power_loops.add_argument("--ref", dest="reference", default="", help="one host IC ref")
    power_loops.add_argument("--format", choices=("json", "jsonl", "text"), default="text")

    place_power_loops = subcommands.add_parser(
        "place-power-loops",
        help="plan or apply pad-aware capacitor placement around declared host rails",
    )
    place_power_loops.add_argument(
        "--spec",
        default=".kicad-mcp/project_spec.json",
        help="project-relative design-intent JSON containing decoupling_pairs",
    )
    place_power_loops.add_argument("--ref", dest="reference", default="", help="one host IC ref")
    place_power_loops.add_argument("--grid", type=float, default=0.25, dest="grid_mm")
    place_power_loops.add_argument(
        "--margin", type=float, default=0.0, dest="courtyard_margin_mm"
    )
    place_power_loops.add_argument("--apply", action="store_true")
    place_power_loops.add_argument("--yes", action="store_true")
    place_power_loops.add_argument("--artifacts", default="", metavar="DIR")
    place_power_loops.add_argument("--format", choices=("json", "jsonl"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "tools":
            _emit_records(catalog_records(args), args.format)
            return
        if args.command == "schema":
            args.query = args.tool
            args.category = None
            args.limit = 0
            records = [record for record in catalog_records(args) if record["name"] == args.tool]
            if not records:
                raise LookupError(f"unknown or unavailable tool: {args.tool}")
            print(json.dumps(records[0], indent=2, sort_keys=True))
            return
        if args.command == "call":
            payload = asyncio.run(call_backend_tool(args))
            _emit_call(payload, args.format)
            if not payload["ok"]:
                raise SystemExit(1)
            return
        if args.command == "edit":
            report = asyncio.run(run_staged_schematic_edit(args))
            if args.format == "jsonl":
                print(json.dumps(report, separators=(",", ":"), sort_keys=True))
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            if not report["promoted"]:
                raise SystemExit(3)
            return
        if args.command == "files":
            records = list(file_records(args))
            _emit_records(records, args.format)
            return
        if args.command == "grep":
            records = list(grep_records(args))
            if args.format == "lines":
                for record in records:
                    print(f"{record['path']}:{record['line']}:{record['column']}:{record['text']}")
            else:
                _emit_records(records, args.format)
            if not records:
                raise SystemExit(1)
            return
        if args.command == "inspect":
            snapshot = filter_snapshot(
                project_snapshot(args.project_dir or "."),
                sheet=args.sheet,
                net=args.net,
                reference=args.reference,
            )
            if args.format == "jsonl":
                for section in ("project", "schematic", "board"):
                    print(
                        json.dumps(
                            {"section": section, "data": snapshot[section]},
                            separators=(",", ":"),
                        )
                    )
            else:
                print(json.dumps(snapshot, indent=2, sort_keys=True))
            return
        if args.command == "prove":
            proof = connectivity_proof(
                project_snapshot(args.project_dir or "."),
                sheet=args.sheet,
                net=args.net,
                reference=args.reference,
            )
            if args.format == "jsonl":
                print(
                    json.dumps(
                        {
                            "section": "summary",
                            "status": proof["status"],
                            "data": proof["summary"],
                            "findings": proof["findings"],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                for pin in proof["pins"]:
                    print(
                        json.dumps(
                            {"section": "pin", **pin},
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
            else:
                print(json.dumps(proof, indent=2, sort_keys=True))
            if proof["status"] == "fail":
                raise SystemExit(3)
            return
        if args.command == "verify":
            report = verification_report(
                args.project_dir or ".",
                sheet=args.sheet,
                net=args.net,
                reference=args.reference,
                artifacts_dir=args.artifacts or None,
            )
            if args.format == "jsonl":
                print(
                    json.dumps(
                        {
                            "section": "summary",
                            "project": report["project"],
                            "filters": report["filters"],
                            "status": report["status"],
                            "artifacts": report["artifacts"],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                for name, check in report["checks"].items():
                    print(
                        json.dumps(
                            {"section": "check", "name": name, **check},
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            if report["status"] == "fail":
                raise SystemExit(3)
            return
        if args.command == "backend":
            report = authority_report(args.project_dir or ".")
            if args.format == "jsonl":
                for section in ("authorities", "live_ipc", "policy", "limitations"):
                    print(
                        json.dumps(
                            {"section": section, "data": report[section]},
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            return
        if args.command == "map":
            map_reference = args.reference or (args.center_ref if args.zoom == 3 else "")
            snapshot = filter_snapshot(
                project_snapshot(args.project_dir or "."),
                sheet=args.sheet,
                net=args.net,
                reference=map_reference,
            )
            if args.view == "spatial":
                print(
                    schematic_spatial_map(
                        snapshot,
                        args.project_dir or ".",
                        zoom=args.zoom,
                        width=args.width,
                        height=args.height,
                        sheet=args.sheet,
                        center_reference=args.center_ref or args.reference,
                    )
                )
            else:
                print(ascii_map(snapshot, zoom=args.zoom, width=args.width))
            return
        if args.command == "plan-labels":
            report = plan_label_compaction(
                project_snapshot(args.project_dir or "."),
                args.project_dir or ".",
                reference=args.reference,
                sheet=args.sheet,
                radius_mm=args.radius_mm,
            )
            if args.format == "json":
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(format_label_compaction_plan(report))
            return
        if args.command == "plan-rewire":
            project_root = Path(args.project_dir or ".").expanduser().resolve()
            report = plan_railway_rewire(
                project_root,
                project_snapshot(project_root),
                sheet=args.sheet,
                cluster_refs=args.references or None,
            )
            if args.format == "json":
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(format_railway_rewire_plan(report))
            return
        if args.command == "apply-rewire":
            project_root = Path(args.project_dir or ".").expanduser().resolve()
            plan = plan_railway_rewire(
                project_root,
                project_snapshot(project_root),
                sheet=args.sheet,
                cluster_refs=args.references or None,
            )
            calls, selected_nets = _rewire_tool_calls(plan, args.rewire_nets)
            args.tool = "apply-rewire"
            args.transaction_label = "railway-rewire"
            args.tool_calls = calls
            args.expected_schematic_sha = plan["source"]["sha256_before"]
            args.expected_rewire_fingerprints = _rewire_fingerprints(plan)
            args.rewire_references = args.references
            args.reference = ""
            args.net = ""
            transaction = asyncio.run(run_staged_schematic_edit(args))
            report = {
                "schema_version": "1.0",
                "status": transaction["status"],
                "selected_nets": selected_nets,
                "planned_source": plan["source"],
                "physical_tool_calls": [
                    {"tool": tool, "arguments": arguments} for tool, arguments in calls
                ],
                "transaction": transaction,
            }
            if args.format == "json":
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(
                    f"RAILWAY APPLY status={report['status']} "
                    f"promoted={str(transaction.get('promoted', False)).lower()}"
                )
                print("nets=" + ",".join(selected_nets))
                print(f"physical_tool_calls={len(calls)} artifacts={transaction['artifacts']}")
            if transaction["status"] == "rejected":
                raise SystemExit(3)
            return
        if args.command == "plan-schematic":
            report = plan_schematic_graph_placement(
                project_snapshot(args.project_dir or "."),
                args.project_dir or ".",
                sheet=args.sheet,
                fixed_references=args.fixed_references,
                candidate_count=args.candidate_count,
            )
            if args.format == "json":
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(format_schematic_graph_placement(report))
            return
        if args.command == "arrange-spec":
            # Arrangement resolves project-local symbol libraries to infer
            # multi-unit pin ownership before planning placement identities.
            _apply_runtime_options(args)
            if args.path == "-":
                source = "stdin"
                spec = _parse_json_object(sys.stdin.read(), source=source)
            else:
                path = Path(args.path).expanduser().resolve()
                source = str(path)
                spec = _parse_json_object(path.read_text(encoding="utf-8"), source=source)
            result = arrange_circuit_spec(
                spec,
                source=source,
                candidate_count=args.candidate_count,
                respect_anchors=not args.reflow,
            )
            if args.format == "spec":
                print(json.dumps(result["arranged_spec"], indent=2))
            elif args.format == "json":
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(format_circuit_spec_arrangement(result))
            return
        if args.command == "power-loops":
            project_root = Path(args.project_dir or ".").expanduser().resolve()
            spec_path = Path(args.spec).expanduser()
            if not spec_path.is_absolute():
                spec_path = project_root / spec_path
            spec = _parse_json_object(spec_path.read_text(encoding="utf-8"), source=str(spec_path))
            raw_pairs = spec.get("decoupling_pairs", [])
            if not isinstance(raw_pairs, list):
                raise ValueError(f"{spec_path}: decoupling_pairs must be a JSON array")
            report = power_loop_report(
                project_snapshot(project_root),
                [cast(dict[str, Any], pair) for pair in raw_pairs if isinstance(pair, dict)],
                reference=args.reference,
            )
            if args.format == "json":
                print(json.dumps(report, indent=2, sort_keys=True))
            elif args.format == "jsonl":
                print(
                    json.dumps(
                        {"section": "summary", "status": report["status"], **report["summary"]},
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                for group in report["groups"]:
                    print(
                        json.dumps(
                            {"section": "group", **group},
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
            else:
                print(format_power_loop_report(report))
            if report["status"] == "fail":
                raise SystemExit(3)
            return
        if args.command == "place-power-loops":
            project_root = Path(args.project_dir or ".").expanduser().resolve()
            spec_path = Path(args.spec).expanduser()
            if not spec_path.is_absolute():
                spec_path = project_root / spec_path
            spec = _parse_json_object(spec_path.read_text(encoding="utf-8"), source=str(spec_path))
            raw_pairs = spec.get("decoupling_pairs", [])
            if not isinstance(raw_pairs, list):
                raise ValueError(f"{spec_path}: decoupling_pairs must be a JSON array")
            plan = power_loop_placement_plan(
                project_snapshot(project_root),
                [cast(dict[str, Any], pair) for pair in raw_pairs if isinstance(pair, dict)],
                reference=args.reference,
                grid_mm=args.grid_mm,
                courtyard_margin_mm=args.courtyard_margin_mm,
            )
            if args.apply:
                if args.mode not in {"write", "experimental"}:
                    raise ValueError("--apply requires --mode write or --mode experimental")
                if not args.yes:
                    raise ValueError("--apply requires --yes after reviewing the placement plan")
                if plan["status"] != "planned":
                    raise ValueError(f"power-loop placement cannot be applied: {plan['reason']}")
                native_placements = [
                    {
                        "reference": placement["reference"],
                        "x_mm": placement["to"][0],
                        "y_mm": placement["to"][1],
                        "rotation_deg": placement["rotation"],
                    }
                    for placement in plan["placements"]
                ]
                transaction = asyncio.run(
                    run_native_board_transaction(
                        args,
                        [("_native_move_footprints_batch", {"placements": native_placements})],
                        label="placement",
                    )
                )
                plan["transaction"] = transaction
                if transaction["status"] in {"rejected", "blocked"}:
                    plan["status"] = transaction["status"]
                    plan["reason"] = transaction["reason"]
                else:
                    plan["status"] = "applied"
            indent = 2 if args.format == "json" else None
            print(
                json.dumps(
                    plan,
                    indent=indent,
                    separators=None if indent else (",", ":"),
                    sort_keys=True,
                )
            )
            if plan["status"] == "blocked":
                raise SystemExit(4)
            if plan["status"] == "rejected":
                raise SystemExit(3)
            return
        if args.command == "route":
            plan = route_plan(
                project_snapshot(args.project_dir or "."),
                args.net,
                layer=args.layer,
                width_mm=args.width_mm,
                clearance_mm=args.clearance_mm,
                allow_critical=args.allow_critical,
            )
            if args.apply:
                if args.mode not in {"write", "experimental"}:
                    raise ValueError("--apply requires --mode write or --mode experimental")
                if not args.yes:
                    raise ValueError("--apply requires --yes after reviewing the route plan")
                if plan["status"] != "planned":
                    reason = plan.get("reason", plan["status"])
                    raise ValueError(f"route cannot be applied: {reason}")
                transaction = asyncio.run(
                    run_native_board_transaction(
                        args,
                        [("pcb_add_tracks_bulk", {"tracks": plan["segments"]})],
                        label=f"route-{args.net}",
                    )
                )
                plan["transaction"] = transaction
                if transaction["status"] in {"rejected", "blocked"}:
                    plan["status"] = transaction["status"]
                    plan["reason"] = transaction["reason"]
            indent = 2 if args.format == "json" else None
            print(
                json.dumps(
                    plan,
                    indent=indent,
                    separators=None if indent else (",", ":"),
                    sort_keys=True,
                )
            )
            if plan["status"] == "refused":
                raise SystemExit(3)
            if plan["status"] == "blocked":
                raise SystemExit(4)
            if plan["status"] == "rejected":
                raise SystemExit(3)
            return
        if args.command == "place":
            keepouts: list[list[float]] = []
            for raw in args.keepout_regions:
                values = [float(value.strip()) for value in raw.split(",")]
                if len(values) != 4:
                    raise ValueError("--keepout requires X1,Y1,X2,Y2")
                keepouts.append(values)
            clusters: list[dict[str, Any]] = []
            for raw in args.cluster_regions:
                values = [value.strip() for value in raw.split(",")]
                if len(values) != 5:
                    raise ValueError("--cluster requires SHEET,X1,Y1,X2,Y2")
                clusters.append(
                    {
                        "sheet": values[0],
                        "x1_mm": float(values[1]),
                        "y1_mm": float(values[2]),
                        "x2_mm": float(values[3]),
                        "y2_mm": float(values[4]),
                    }
                )
            anchors: list[dict[str, Any]] = []
            for raw in args.anchors:
                values = [value.strip() for value in raw.split(",")]
                if len(values) not in {3, 4}:
                    raise ValueError("--anchor requires REF,EDGE,OFFSET[,ROTATION]")
                anchor: dict[str, Any] = {
                    "reference": values[0],
                    "edge": values[1],
                    "offset_mm": float(values[2]),
                }
                if len(values) == 4:
                    anchor["rotation"] = float(values[3])
                anchors.append(anchor)
            for raw in args.absolute_anchors:
                values = [value.strip() for value in raw.split(",")]
                if len(values) not in {3, 4}:
                    raise ValueError("--at requires REF,X,Y[,ROTATION]")
                anchor = {
                    "reference": values[0],
                    "x_mm": float(values[1]),
                    "y_mm": float(values[2]),
                }
                if len(values) == 4:
                    anchor["rotation"] = float(values[3])
                anchors.append(anchor)
            project_root = Path(args.project_dir or ".").expanduser().resolve()
            proximity_pairs: list[dict[str, Any]] = []
            spec_path = Path(args.spec).expanduser()
            if not spec_path.is_absolute():
                spec_path = project_root / spec_path
            if spec_path.is_file():
                spec = _parse_json_object(
                    spec_path.read_text(encoding="utf-8"), source=str(spec_path)
                )
                raw_pairs = spec.get("decoupling_pairs", [])
                if not isinstance(raw_pairs, list):
                    raise ValueError(f"{spec_path}: decoupling_pairs must be a JSON array")
                proximity_pairs = [
                    cast(dict[str, Any], pair) for pair in raw_pairs if isinstance(pair, dict)
                ]
            plan = placement_plan(
                project_snapshot(project_root),
                fixed_references=args.fixed_references,
                anchors=anchors,
                cluster_regions=clusters,
                keepout_regions=keepouts,
                proximity_pairs=proximity_pairs,
                margin_mm=args.margin_mm,
                iterations=args.iterations,
                grid_mm=args.grid_mm,
                seed=args.seed,
            )
            if args.apply:
                if args.mode not in {"write", "experimental"}:
                    raise ValueError("--apply requires --mode write or --mode experimental")
                if not args.yes:
                    raise ValueError("--apply requires --yes after reviewing the placement plan")
                if plan["status"] != "planned":
                    reason = plan.get("reason", plan["status"])
                    raise ValueError(f"placement cannot be applied: {reason}")
                quality_gate = plan.get("quality_gate", {})
                if quality_gate.get("status") != "pass":
                    plan["status"] = "rejected"
                    plan["reason"] = str(
                        quality_gate.get("reason", "placement quality gate failed")
                    )
                    indent = 2 if args.format == "json" else None
                    print(
                        json.dumps(
                            plan,
                            indent=indent,
                            separators=None if indent else (",", ":"),
                            sort_keys=True,
                        )
                    )
                    raise SystemExit(3)
                placements_to_apply: list[dict[str, Any]] = []
                for placement in plan["placements"]:
                    if placement["fixed"] and not placement.get("anchored"):
                        continue
                    if (
                        placement["from"] == placement["to"]
                        and placement.get("from_rotation") == placement["rotation"]
                    ):
                        continue
                    x_mm, y_mm = placement["to"]
                    placements_to_apply.append(
                        {
                            "reference": placement["reference"],
                            "x_mm": x_mm,
                            "y_mm": y_mm,
                            "rotation_deg": placement["rotation"],
                        }
                    )
                operations = (
                    [
                        (
                            "_native_move_footprints_batch",
                            {"placements": placements_to_apply},
                        )
                    ]
                    if placements_to_apply
                    else []
                )
                transaction = asyncio.run(
                    run_native_board_transaction(args, operations, label="placement")
                )
                plan["transaction"] = transaction
                if transaction["status"] in {"rejected", "blocked"}:
                    plan["status"] = transaction["status"]
                    plan["reason"] = transaction["reason"]
                else:
                    plan["status"] = "applied"
            indent = 2 if args.format == "json" else None
            print(
                json.dumps(
                    plan,
                    indent=indent,
                    separators=None if indent else (",", ":"),
                    sort_keys=True,
                )
            )
            if plan["status"] == "blocked":
                raise SystemExit(4)
            if plan["status"] == "rejected":
                raise SystemExit(3)
            return
    except BrokenPipeError:
        # A downstream command (commonly head/jq/rg) intentionally stopped
        # reading. Replace stdout so Python's shutdown flush stays quiet.
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        raise SystemExit(0) from None
    except (LookupError, RuntimeError, ValueError, OSError) as exc:
        print(f"kicadq: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
