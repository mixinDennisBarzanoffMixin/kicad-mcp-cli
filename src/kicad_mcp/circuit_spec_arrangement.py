"""Pipe-friendly railway layout transformer for schematic circuit specs."""

from __future__ import annotations

import copy
import math
from typing import Any

from .models.visual_qa import DEFAULT_PAPER, PAPER_SIZES_MM
from .schematic_graph_placement import plan_fresh_schematic_layout

type JsonRecord = dict[str, Any]


def _require_records(value: object, *, field: str) -> list[JsonRecord]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"circuit spec {field!r} must be a list of JSON objects")
    return value


def _planner_nets(nets: list[JsonRecord]) -> tuple[list[JsonRecord], int]:
    """Translate build-spec ``REF.PIN`` endpoints to railway graph nodes."""

    translated: list[JsonRecord] = []
    endpoint_count = 0
    for index, net in enumerate(nets):
        name = net.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"circuit spec net {index} requires a non-empty string name")
        endpoints = net.get("endpoints", [])
        if not isinstance(endpoints, list) or not all(
            isinstance(endpoint, str) for endpoint in endpoints
        ):
            raise ValueError(f"circuit spec net {name!r} endpoints must be strings")
        nodes: list[JsonRecord] = []
        for endpoint in endpoints:
            if "." not in endpoint:
                raise ValueError(
                    f"circuit spec net {name!r} endpoint {endpoint!r} must be REF.PIN"
                )
            reference, pin = endpoint.rsplit(".", 1)
            if not reference or not pin:
                raise ValueError(
                    f"circuit spec net {name!r} endpoint {endpoint!r} must be REF.PIN"
                )
            nodes.append(
                {
                    "reference": reference,
                    "pin": pin,
                    "function": "",
                    # Build specs do not carry library pin electrical types. Keep
                    # direction explicitly unknown instead of inventing authority.
                    "type": "passive",
                }
            )
            endpoint_count += 1
        translated.append({"name": name, "nodes": nodes})
    return translated, endpoint_count


def _page_extent(spec: JsonRecord) -> tuple[str, tuple[float, float]]:
    paper = str(spec.get("paper") or spec.get("max_paper") or DEFAULT_PAPER)
    if paper not in PAPER_SIZES_MM:
        raise ValueError(
            f"unsupported circuit spec paper {paper!r}; expected one of "
            + ", ".join(sorted(PAPER_SIZES_MM))
        )
    return paper, PAPER_SIZES_MM[paper]


def _validate_symbols(symbols: list[JsonRecord]) -> list[str]:
    references: list[str] = []
    for index, symbol in enumerate(symbols):
        reference = symbol.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"circuit spec symbol {index} requires a non-empty reference")
        references.append(reference)
        for axis in ("x_mm", "y_mm"):
            if axis not in symbol:
                continue
            coordinate = symbol[axis]
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, int | float)
                or not math.isfinite(float(coordinate))
            ):
                raise ValueError(f"circuit spec symbol {reference} has invalid {axis}")
    if not references:
        raise ValueError("circuit spec requires at least one symbol")
    if len(set(references)) != len(references):
        raise ValueError("circuit spec symbols must have unique references")
    return references


def arrange_circuit_spec(
    spec: JsonRecord,
    *,
    source: str = "stdin",
    candidate_count: int = 3,
) -> JsonRecord:
    """Fill only absent symbol coordinates using the fresh railway planner.

    The returned ``arranged_spec`` is directly consumable by ``sch_build_circuit``
    or ``sch_analyze_net_compilation``. Planning diagnostics live beside it and
    are never injected into that build payload.
    """

    symbols = _require_records(spec.get("symbols"), field="symbols")
    nets = _require_records(spec.get("nets", []), field="nets")
    references = _validate_symbols(symbols)
    planner_nets, endpoint_count = _planner_nets(nets)
    paper, extent = _page_extent(spec)
    layout = plan_fresh_schematic_layout(
        symbols,
        planner_nets,
        page_extent_mm=extent,
        candidate_count=candidate_count,
    )
    generated_by_ref = {
        str(item["reference"]): item["proposed_mm"] for item in layout["placements"]
    }
    arranged = copy.deepcopy(spec)
    arranged_symbols = _require_records(arranged.get("symbols"), field="symbols")
    generated_axes = 0
    explicit_anchors: list[str] = []
    partially_anchored: list[str] = []
    for symbol in arranged_symbols:
        reference = str(symbol["reference"])
        has_x = "x_mm" in symbol
        has_y = "y_mm" in symbol
        if has_x and has_y:
            explicit_anchors.append(reference)
        elif has_x or has_y:
            partially_anchored.append(reference)
        proposed_x, proposed_y = generated_by_ref[reference]
        if not has_x:
            symbol["x_mm"] = proposed_x
            generated_axes += 1
        if not has_y:
            symbol["y_mm"] = proposed_y
            generated_axes += 1
    arranged["auto_layout"] = False
    return {
        "schema_version": "1.0",
        "status": "arranged",
        "read_only": True,
        "source": source,
        "summary": {
            "paper": paper,
            "page_extent_mm": list(extent),
            "symbols": len(references),
            "nets": len(nets),
            "endpoints": endpoint_count,
            "explicit_anchors": len(explicit_anchors),
            "partial_anchors": len(partially_anchored),
            "generated_coordinate_axes": generated_axes,
            "generated_symbols": sum(
                1 for symbol in symbols if "x_mm" not in symbol or "y_mm" not in symbol
            ),
        },
        "anchors": {
            "explicit": sorted(explicit_anchors),
            "partial": sorted(partially_anchored),
        },
        "layout": layout,
        "arranged_spec": arranged,
    }


def format_circuit_spec_arrangement(result: JsonRecord) -> str:
    """Render the transformer diagnostics without contaminating stdout JSON modes."""

    summary = result["summary"]
    layout = result["layout"]
    graph = layout["graph"]
    lines = [
        f"CIRCUIT SPEC ARRANGEMENT  status={result['status']} read_only=true",
        f"source={result['source']} paper={summary['paper']} "
        f"extent={summary['page_extent_mm'][0]}x{summary['page_extent_mm'][1]}mm",
        f"symbols={summary['symbols']} nets={summary['nets']} endpoints={summary['endpoints']} "
        f"explicit_anchors={summary['explicit_anchors']} "
        f"generated_symbols={summary['generated_symbols']}",
        f"SELECTED {layout['selected_candidate']}",
        "CANDIDATES",
    ]
    for candidate in layout["ranked_candidates"]:
        cost = candidate["cost"]
        lines.append(
            f"  {candidate['name']} overlap={cost['symbol_overlaps']} "
            f"crossings={cost['wire_crossings']} "
            f"wire={cost['total_manhattan_wire_length_mm']}mm "
            f"area={cost['compactness_area_mm2']}mm2"
        )
    lines.append("RANKS")
    for placement in layout["placements"]:
        lines.append(
            f"  {placement['reference']} rank={placement['rank']} "
            f"cluster={placement['cluster']} at={placement['proposed_mm']}"
        )
    rails = graph["rail_nets"]
    lines.append("RAILS " + (", ".join(rails) if rails else "none"))
    lines.append(
        "OUTPUT: use --format spec for a clean sch_build_circuit/"
        "sch_analyze_net_compilation payload"
    )
    return "\n".join(lines)
