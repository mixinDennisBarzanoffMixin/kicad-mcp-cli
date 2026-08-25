"""Small MCP facade over the shell-first KiCad command surface.

This server intentionally publishes three tools instead of mirroring the full
backend catalog.  Agents discover with ``kicad_catalog``, execute by name with
``kicad_run``, and search project source with ``kicad_grep``.  Humans can use
the same operations directly through ``kicadq`` and normal shell pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .shell_cli import catalog_records, grep_records, invoke_backend_tool

server = FastMCP(
    name="kicad-mcp-cli",
    instructions=(
        "A compact shell-first KiCad interface. Search commands with kicad_catalog, "
        "inspect input_schema, then invoke one by name with kicad_run."
    ),
)
_backend_lock = asyncio.Lock()


def _base_args(
    *, project_dir: str, profile: str, mode: str, query: str = "", limit: int = 0
) -> argparse.Namespace:
    return argparse.Namespace(
        project_dir=project_dir or None,
        profile=profile,
        mode=mode,
        query=query,
        category=None,
        limit=limit,
    )


@server.tool()
async def kicad_catalog(
    query: str = "",
    category: str = "",
    limit: int = 40,
    profile: str = "full",
    mode: str = "readonly",
) -> list[dict[str, Any]]:
    """Search KiCad operations by words in their names, categories, and descriptions."""
    args = _base_args(project_dir="", profile=profile, mode=mode, query=query, limit=limit)
    args.category = category or None
    async with _backend_lock:
        return catalog_records(args)


@server.tool()
async def kicad_run(
    tool: str,
    arguments: dict[str, Any] | None = None,
    project_dir: str = "",
    profile: str = "full",
    mode: str = "readonly",
) -> dict[str, Any]:
    """Run one named KiCad operation with JSON arguments and return structured JSON."""
    args = _base_args(project_dir=project_dir, profile=profile, mode=mode)
    args.tool = tool
    args.args_json = ""
    args.set_values = []
    async with _backend_lock:
        return await invoke_backend_tool(args, tool, arguments or {})


@server.tool()
async def kicad_grep(
    pattern: str,
    project_dir: str,
    ignore_case: bool = False,
    fixed_strings: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Search textual KiCad project files using ripgrep and return match records."""
    root = str(Path(project_dir).expanduser().resolve())
    args = argparse.Namespace(
        pattern=pattern,
        paths=[root],
        project_dir=root,
        ignore_case=ignore_case,
        fixed_strings=fixed_strings,
        limit=limit,
    )
    async with _backend_lock:
        return list(grep_records(args))


def main() -> None:
    """Run the compact three-tool MCP server over stdio."""
    os.environ.setdefault("KICAD_MCP_LOG_LEVEL", "WARNING")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
