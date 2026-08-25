"""Compact adapters for deep project inspection and route planning."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from ..config import get_config
from ..deep_inspection import ascii_map, filter_snapshot, project_snapshot, route_plan
from .metadata import headless_compatible


def register(mcp: FastMCP) -> None:
    """Register file-backed project intelligence tools."""

    @mcp.tool()
    @headless_compatible
    def project_get_deep_snapshot(
        sheet: str = "",
        net: str = "",
        reference: str = "",
    ) -> str:
        """Inspect hierarchy, component pins, nets, footprints, pads, tracks, and vias.

        The result is one JSON snapshot produced from KiCad's own XML netlist export
        plus the PCB file. It works headlessly and is intended for jq-style queries.
        Optional filters reduce the returned schematic/footprint collections.
        """
        snapshot = filter_snapshot(
            project_snapshot(get_config().project_root),
            sheet=sheet,
            net=net,
            reference=reference,
        )
        return json.dumps(snapshot, indent=2, sort_keys=True)

    @mcp.tool()
    @headless_compatible
    def project_get_ascii_map(
        zoom: int = 0,
        width: int = 100,
        sheet: str = "",
        net: str = "",
        reference: str = "",
    ) -> str:
        """Render subsystem, component, net/pin, or physical-PCB text maps.

        Zoom 0 is the sheet architecture, 1 lists components and footprints, 2
        renders net-to-pin connectivity, and 3 renders board geometry.
        """
        if zoom not in range(4):
            raise ValueError("zoom must be between 0 and 3")
        snapshot = filter_snapshot(
            project_snapshot(get_config().project_root),
            sheet=sheet,
            net=net,
            reference=reference,
        )
        return ascii_map(snapshot, zoom=zoom, width=width)

    @mcp.tool()
    @headless_compatible
    def pcb_get_route_plan(
        net_name: str,
        layer: str = "F.Cu",
        width_mm: float = 0.25,
        clearance_mm: float = 0.5,
        allow_critical: bool = False,
    ) -> str:
        """Dry-run a conservative Manhattan route between placed pads on one net.

        This never edits the board. Critical power, ground, clock, USB differential,
        and RF nets are refused by default. Review the JSON segments, then pass them
        to ``pcb_add_tracks_bulk`` in write mode if appropriate.
        """
        plan = route_plan(
            project_snapshot(get_config().project_root),
            net_name,
            layer=layer,
            width_mm=width_mm,
            clearance_mm=clearance_mm,
            allow_critical=allow_critical,
        )
        return json.dumps(plan, indent=2, sort_keys=True)
