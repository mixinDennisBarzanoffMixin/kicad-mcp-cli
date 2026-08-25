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
import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from mcp import types as mcp_types
from pydantic import BaseModel

from .config import reset_config
from .deep_inspection import (
    ascii_map,
    authority_report,
    connectivity_proof,
    filter_snapshot,
    placement_plan,
    project_snapshot,
    route_plan,
    verification_report,
)
from .server import build_server
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
        print(record.get("name", ""))


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
    command.extend(("--glob", "!.git/**"))
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
    command.extend(("--glob", "!.git/**", "--", args.pattern))
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
    map_command.add_argument("--sheet", default="")
    map_command.add_argument("--net", default="")
    map_command.add_argument("--ref", dest="reference", default="")

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
        "--keepout",
        dest="keepout_regions",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="absolute board keepout rectangle in mm; repeatable",
    )
    place.add_argument("--margin", type=float, default=3.0, dest="margin_mm")
    place.add_argument("--iterations", type=int, default=300)
    place.add_argument("--grid", type=float, default=0.5, dest="grid_mm")
    place.add_argument("--seed", type=int, default=42)
    place.add_argument("--apply", action="store_true")
    place.add_argument("--yes", action="store_true", help="confirm placement mutation")
    place.add_argument("--format", choices=("json", "jsonl"), default="json")
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
            snapshot = filter_snapshot(
                project_snapshot(args.project_dir or "."),
                sheet=args.sheet,
                net=args.net,
                reference=args.reference,
            )
            print(ascii_map(snapshot, zoom=args.zoom, width=args.width))
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
                payload = asyncio.run(
                    invoke_backend_tool(args, "pcb_add_tracks_bulk", {"tracks": plan["segments"]})
                )
                plan["apply_result"] = payload
                if not payload["ok"]:
                    raise RuntimeError("backend rejected route application")
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
            return
        if args.command == "place":
            keepouts: list[list[float]] = []
            for raw in args.keepout_regions:
                values = [float(value.strip()) for value in raw.split(",")]
                if len(values) != 4:
                    raise ValueError("--keepout requires X1,Y1,X2,Y2")
                keepouts.append(values)
            plan = placement_plan(
                project_snapshot(args.project_dir or "."),
                fixed_references=args.fixed_references,
                keepout_regions=keepouts,
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
                results = []
                for placement in plan["placements"]:
                    if placement["fixed"] or placement["from"] == placement["to"]:
                        continue
                    x_mm, y_mm = placement["to"]
                    result = asyncio.run(
                        invoke_backend_tool(
                            args,
                            "pcb_move_footprint",
                            {
                                "reference": placement["reference"],
                                "x_mm": x_mm,
                                "y_mm": y_mm,
                                "rotation_deg": 0.0,
                            },
                        )
                    )
                    results.append(result)
                    if not result["ok"]:
                        raise RuntimeError(
                            f"backend rejected placement for {placement['reference']}"
                        )
                plan["apply_results"] = results
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
