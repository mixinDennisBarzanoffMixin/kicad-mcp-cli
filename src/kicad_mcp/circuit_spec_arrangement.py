"""Pipe-friendly railway layout transformer for schematic circuit specs."""

from __future__ import annotations

import copy
import math
from typing import Any, cast

from .models.visual_qa import DEFAULT_PAPER, PAPER_SIZES_MM
from .schematic_graph_placement import plan_fresh_schematic_layout

type JsonRecord = dict[str, Any]


def _require_records(value: object, *, field: str) -> list[JsonRecord]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"circuit spec {field!r} must be a list of JSON objects")
    return value


def _planner_symbols(
    symbols: list[JsonRecord],
) -> tuple[
    list[JsonRecord],
    dict[str, dict[str, str]],
    dict[str, tuple[str, int]],
]:
    """Enrich a build spec with installed-library size and pin-type evidence."""

    # Local import keeps the standalone graph planner lightweight and avoids a
    # module-level dependency on the schematic MCP composition root.
    from .tools.schematic import _symbol_local_extent, get_pin_metadata

    enriched = copy.deepcopy(symbols)
    pin_types: dict[str, dict[str, str]] = {}
    identities: dict[str, tuple[str, int]] = {}
    for symbol in enriched:
        reference = str(symbol["reference"])
        library = str(symbol.get("library", ""))
        symbol_name = str(symbol.get("symbol_name", ""))
        unit = int(symbol.get("unit", 1) or 1)
        placement_id = f"{reference}@@unit:{unit}"
        identities[placement_id] = (reference, unit)
        symbol["reference"] = placement_id
        aliases: dict[str, str] = {}
        if library and symbol_name:
            for number, info in get_pin_metadata(library, symbol_name, unit).items():
                electrical_type = str(info.get("etype", "passive"))
                name = str(info.get("name", ""))
                for key in (number, number.casefold(), name, name.casefold()):
                    if key:
                        aliases.setdefault(key, electrical_type)
        pin_types[placement_id] = aliases

        extent = _symbol_local_extent(symbol)
        if extent is not None:
            _min_x, _min_y, width, height = extent
            # Reserve room for body text and one short terminal lane around the
            # exact pin span.  The graph repairer then treats large MCUs and
            # connectors as large objects instead of default 7.62 mm squares.
            symbol.setdefault("width_mm", max(7.62, width + 15.24))
            symbol.setdefault("height_mm", max(7.62, height + 10.16))
    return enriched, pin_types, identities


def _planner_nets(
    nets: list[JsonRecord],
    pin_types: dict[str, dict[str, str]],
    identities: dict[str, tuple[str, int]],
) -> tuple[list[JsonRecord], int]:
    """Translate build-spec ``REF.PIN`` endpoints to railway graph nodes."""

    translated: list[JsonRecord] = []
    endpoint_count = 0
    for index, net in enumerate(nets):
        name = net.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"circuit spec net {index} requires a non-empty string name")
        endpoints = net.get("endpoints", [])
        if not isinstance(endpoints, list) or not all(
            isinstance(endpoint, str | dict) for endpoint in endpoints
        ):
            raise ValueError(
                f"circuit spec net {name!r} endpoints must be REF.PIN strings or objects"
            )
        nodes: list[JsonRecord] = []
        for endpoint in endpoints:
            unit: int | None = None
            if isinstance(endpoint, str):
                if "." not in endpoint:
                    raise ValueError(
                        f"circuit spec net {name!r} endpoint {endpoint!r} must be REF.PIN"
                    )
                reference, pin = endpoint.rsplit(".", 1)
            else:
                reference = str(
                    endpoint.get("reference", endpoint.get("ref", endpoint.get("symbol", "")))
                )
                pin = str(
                    endpoint.get(
                        "pin",
                        endpoint.get("pin_number", endpoint.get("number", "")),
                    )
                )
                raw_unit = endpoint.get("unit", endpoint.get("symbol_unit"))
                if raw_unit is not None:
                    if isinstance(raw_unit, bool):
                        raise ValueError(
                            f"circuit spec net {name!r} endpoint has invalid unit {raw_unit!r}"
                        )
                    try:
                        unit = int(raw_unit)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"circuit spec net {name!r} endpoint has invalid unit {raw_unit!r}"
                        ) from exc
                    if unit < 1:
                        raise ValueError(
                            f"circuit spec net {name!r} endpoint has invalid unit {raw_unit!r}"
                        )
            if not reference or not pin:
                raise ValueError(f"circuit spec net {name!r} endpoint {endpoint!r} must be REF.PIN")
            candidates = [
                placement_id
                for placement_id, identity in identities.items()
                if identity[0] == reference and (unit is None or identity[1] == unit)
            ]
            pin_candidates = [
                placement_id
                for placement_id in candidates
                if pin in pin_types.get(placement_id, {})
                or pin.casefold() in pin_types.get(placement_id, {})
            ]
            if len(pin_candidates) == 1:
                placement_id = pin_candidates[0]
            elif len(candidates) == 1:
                placement_id = candidates[0]
            elif not candidates:
                suffix = f" unit {unit}" if unit is not None else ""
                raise ValueError(
                    f"circuit spec net {name!r} endpoint references missing symbol "
                    f"{reference!r}{suffix}"
                )
            else:
                raise ValueError(
                    f"circuit spec net {name!r} endpoint {endpoint!r} is ambiguous across "
                    f"units of {reference!r}; use {{reference, unit, pin}}"
                )
            reference_types = pin_types.get(placement_id, {})
            nodes.append(
                {
                    "reference": placement_id,
                    "pin": pin,
                    "function": "",
                    "type": reference_types.get(
                        pin,
                        reference_types.get(pin.casefold(), "passive"),
                    ),
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
    identities: list[tuple[str, int]] = []
    for index, symbol in enumerate(symbols):
        reference = symbol.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"circuit spec symbol {index} requires a non-empty reference")
        references.append(reference)
        raw_unit = symbol.get("unit", 1)
        if isinstance(raw_unit, bool):
            raise ValueError(f"circuit spec symbol {reference} has invalid unit")
        try:
            unit = int(raw_unit or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"circuit spec symbol {reference} has invalid unit") from exc
        if unit < 1:
            raise ValueError(f"circuit spec symbol {reference} has invalid unit")
        identities.append((reference, unit))
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
    if len(set(identities)) != len(identities):
        raise ValueError("circuit spec symbols must have unique reference+unit placements")
    return references


def _rewrite_layout_identities(
    layout: JsonRecord,
    identities: dict[str, tuple[str, int]],
) -> JsonRecord:
    """Replace internal planner IDs with stable human-readable unit identities."""

    reference_counts: dict[str, int] = {}
    for reference, _unit in identities.values():
        reference_counts[reference] = reference_counts.get(reference, 0) + 1
    display = {
        placement_id: (
            reference if reference_counts[reference] == 1 else f"{reference}[unit={unit}]"
        )
        for placement_id, (reference, unit) in identities.items()
    }

    def rewrite(value: object) -> object:
        if isinstance(value, str):
            return display.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    public = cast(JsonRecord, rewrite(copy.deepcopy(layout)))
    for source_placement, placement in zip(layout["placements"], public["placements"], strict=True):
        internal = str(source_placement["reference"])
        reference, unit = identities[internal]
        placement["reference"] = reference
        placement["unit"] = unit
        placement["placement_identity"] = [reference, unit]
    return public


def arrange_circuit_spec(
    spec: JsonRecord,
    *,
    source: str = "stdin",
    candidate_count: int = 3,
    respect_anchors: bool = True,
) -> JsonRecord:
    """Fill absent coordinates, or reflow all symbols, using the railway planner.

    The returned ``arranged_spec`` is directly consumable by ``sch_build_circuit``
    or ``sch_analyze_net_compilation``. Planning diagnostics live beside it and
    are never injected into that build payload.
    """

    symbols = _require_records(spec.get("symbols"), field="symbols")
    nets = _require_records(spec.get("nets", []), field="nets")
    references = _validate_symbols(symbols)
    planner_symbols, pin_types, identities = _planner_symbols(symbols)
    planner_nets, endpoint_count = _planner_nets(nets, pin_types, identities)
    paper, extent = _page_extent(spec)
    layout = plan_fresh_schematic_layout(
        planner_symbols,
        planner_nets,
        page_extent_mm=extent,
        candidate_count=candidate_count,
    )
    generated_by_identity = {
        identities[str(item["reference"])]: item["proposed_mm"] for item in layout["placements"]
    }
    arranged = copy.deepcopy(spec)
    arranged_symbols = _require_records(arranged.get("symbols"), field="symbols")
    generated_axes = 0
    explicit_anchors: list[str] = []
    partially_anchored: list[str] = []
    for symbol in arranged_symbols:
        reference = str(symbol["reference"])
        unit = int(symbol.get("unit", 1) or 1)
        has_x = "x_mm" in symbol
        has_y = "y_mm" in symbol
        if has_x and has_y:
            explicit_anchors.append(reference)
        elif has_x or has_y:
            partially_anchored.append(reference)
        proposed_x, proposed_y = generated_by_identity[(reference, unit)]
        if not respect_anchors or not has_x:
            symbol["x_mm"] = proposed_x
            generated_axes += 1
        if not respect_anchors or not has_y:
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
                1
                for symbol in symbols
                if not respect_anchors or "x_mm" not in symbol or "y_mm" not in symbol
            ),
            "respect_anchors": respect_anchors,
        },
        "anchors": {
            "explicit": sorted(explicit_anchors),
            "partial": sorted(partially_anchored),
            "honored": respect_anchors,
        },
        "layout": _rewrite_layout_identities(layout, identities),
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
