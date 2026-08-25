"""Generic read-only railway-diagram planner for KiCad schematic pages."""

from __future__ import annotations

import math
import re
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from .models.visual_qa import parse_paper_extent, parse_placed_symbols
from .schematic_spatial import parse_schematic_geometry, select_schematic_sheet
from .utils.geometry import Box, union

type JsonRecord = dict[str, Any]
type Point = tuple[float, float]
type Segment = tuple[Point, Point]

_SOURCE_TYPES = frozenset({"output", "power_out", "open_collector", "open_emitter"})
_SINK_TYPES = frozenset({"input", "power_in"})
_POWER_NAME = re.compile(
    r"(?:^|[/_+-])(GND|VCC|VDD|VSS|VBAT|BAT|POWER|RAW|SYS|\dV\d?|\dV)(?:$|[/_+-])",
    re.IGNORECASE,
)
_CONNECTOR_REF = re.compile(r"^(?:JP|J|P|CN|CON|X)[A-Z_]*\d", re.IGNORECASE)
_GRID_MM = 2.54


def _canonical(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _is_ground(name: str) -> bool:
    return _canonical(name).upper() in {"GND", "VSS", "AGND", "DGND", "PGND"}


def _snap(value: float) -> float:
    return round(round(value / _GRID_MM) * _GRID_MM, 4)


def _local_nets(snapshot: JsonRecord, references: set[str]) -> list[JsonRecord]:
    nets: list[JsonRecord] = []
    for net in snapshot["schematic"]["nets"]:
        nodes = [node for node in net["nodes"] if str(node["reference"]) in references]
        unique_refs = sorted({str(node["reference"]) for node in nodes})
        if len(unique_refs) >= 2:
            nets.append({**net, "nodes": nodes, "references": unique_refs})
    return nets


def _rail_nets(nets: list[JsonRecord], component_count: int) -> set[str]:
    fanout_limit = max(4, math.ceil(component_count * 0.3))
    return {
        str(net["name"])
        for net in nets
        if len(net["references"]) >= fanout_limit
        or _POWER_NAME.search(_canonical(str(net["name"])))
    }


def _directed_arcs(
    references: set[str],
    nets: list[JsonRecord],
    rails: set[str],
    positions: dict[str, Point],
) -> tuple[set[tuple[str, str]], list[JsonRecord]]:
    arcs: set[tuple[str, str]] = set()
    ambiguities: list[JsonRecord] = []
    nets_by_reference: dict[str, set[str]] = {reference: set() for reference in references}
    for net in nets:
        for reference in net["references"]:
            nets_by_reference[reference].add(str(net["name"]))

    def decoupling_attachment(reference: str) -> bool:
        attached = nets_by_reference[reference]
        return reference.upper().startswith("C") and any(_is_ground(name) for name in attached)

    def series_candidate(reference: str) -> bool:
        non_ground = [name for name in nets_by_reference[reference] if not _is_ground(name)]
        return len(non_ground) == 2 and not reference.upper().startswith("C")

    for net in nets:
        name = str(net["name"])
        if _is_ground(name):
            continue
        sources = sorted(
            {
                str(node["reference"])
                for node in net["nodes"]
                if str(node.get("type", "")).casefold() in _SOURCE_TYPES
            }
        )
        sinks = sorted(
            {
                str(node["reference"])
                for node in net["nodes"]
                if str(node.get("type", "")).casefold() in _SINK_TYPES
            }
        )
        passive = sorted(set(net["references"]) - set(sources) - set(sinks))
        if sources:
            targets = sinks + [ref for ref in passive if not decoupling_attachment(ref)]
            arcs.update(
                (source, target) for source in sources for target in targets if source != target
            )
        elif sinks:
            connectors = [ref for ref in passive if _CONNECTOR_REF.match(ref)]
            inferred_sources = connectors or (
                [ref for ref in passive if not decoupling_attachment(ref)]
                if name not in rails
                else []
            )
            if inferred_sources:
                arcs.update(
                    (source, sink)
                    for source in inferred_sources
                    for sink in sinks
                    if source != sink
                )
            else:
                ambiguities.append(
                    {
                        "net": name,
                        "code": "external_or_unresolved_source_net",
                        "references": net["references"],
                        "reason": (
                            "sink net has no local typed driver; attachments do not become sources"
                        ),
                    }
                )
        else:
            series = [ref for ref in net["references"] if series_candidate(ref)]
            if len(series) >= 2:
                ordered = sorted(
                    series,
                    key=lambda ref: (positions[ref][0], positions[ref][1], ref),
                )
                arcs.update(zip(ordered, ordered[1:], strict=False))
                ambiguities.append(
                    {
                        "net": name,
                        "code": "series_direction_geometry_tiebreak",
                        "references": ordered,
                        "reason": (
                            "ambiguous two-net series elements follow saved left-to-right order"
                        ),
                    }
                )
            else:
                ambiguities.append(
                    {
                        "net": name,
                        "code": "direction_unresolved",
                        "references": net["references"],
                        "reason": "no unique driver, sink, or connector source was inferable",
                    }
                )
    return {(a, b) for a, b in arcs if a in references and b in references}, ambiguities


def _strong_components(references: set[str], arcs: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {ref: [] for ref in references}
    reverse = {ref: [] for ref in references}
    for source, target in arcs:
        adjacency[source].append(target)
        reverse[target].append(source)
    visited: set[str] = set()
    order: list[str] = []

    def visit(node: str) -> None:
        visited.add(node)
        for target in sorted(adjacency[node]):
            if target not in visited:
                visit(target)
        order.append(node)

    for reference in sorted(references):
        if reference not in visited:
            visit(reference)
    visited.clear()
    components: list[list[str]] = []

    def collect(node: str, members: list[str]) -> None:
        visited.add(node)
        members.append(node)
        for source in sorted(reverse[node]):
            if source not in visited:
                collect(source, members)

    for reference in reversed(order):
        if reference not in visited:
            members: list[str] = []
            collect(reference, members)
            components.append(sorted(members))
    return components


def _ranks(
    references: set[str], arcs: set[tuple[str, str]]
) -> tuple[dict[str, int], list[list[str]]]:
    components = _strong_components(references, arcs)
    owner = {reference: index for index, members in enumerate(components) for reference in members}
    dag_edges = {(owner[a], owner[b]) for a, b in arcs if owner[a] != owner[b]}
    incoming = {index: 0 for index in range(len(components))}
    adjacency = {index: [] for index in range(len(components))}
    for source, target in dag_edges:
        adjacency[source].append(target)
        incoming[target] += 1
    queue = deque(sorted(index for index, count in incoming.items() if count == 0))
    component_rank = {index: 0 for index in range(len(components))}
    while queue:
        source = queue.popleft()
        for target in sorted(adjacency[source]):
            component_rank[target] = max(component_rank[target], component_rank[source] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    ranks = {reference: component_rank[owner[reference]] for reference in references}
    cycles = [members for members in components if len(members) > 1]
    return ranks, cycles


def _undirected_adjacency(
    references: set[str], nets: list[JsonRecord], rails: set[str]
) -> dict[str, set[str]]:
    adjacency = {reference: set() for reference in references}
    for net in nets:
        if str(net["name"]) in rails:
            continue
        members = list(net["references"])
        for reference in members:
            adjacency[reference].update(item for item in members if item != reference)
    return adjacency


def _functional_clusters(
    references: set[str],
    nets: list[JsonRecord],
    rails: set[str],
    positions: dict[str, Point],
) -> tuple[list[JsonRecord], dict[str, str]]:
    adjacency = _undirected_adjacency(references, nets, rails)
    hubs = sorted(
        ref
        for ref in references
        if ref.upper().startswith(("U", "IC", "J", "Q")) or len(adjacency[ref]) >= 4
    )
    if not hubs:
        hubs = [min(references)] if references else []
    assignments: dict[str, str] = {}
    for reference in sorted(references):
        if reference in hubs:
            assignments[reference] = reference
            continue
        distances = {reference: 0}
        queue = deque([reference])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        reachable = [hub for hub in hubs if hub in distances]
        if not reachable:
            assignments[reference] = min(
                hubs,
                key=lambda hub: (math.dist(positions[reference], positions[hub]), hub),
            )
            continue
        assignments[reference] = min(
            reachable,
            key=lambda hub: (
                distances[hub],
                math.dist(positions[reference], positions[hub]),
                hub,
            ),
        )
    records = []
    for hub in sorted(set(assignments.values())):
        members = sorted(ref for ref, owner in assignments.items() if owner == hub)
        records.append(
            {
                "id": f"cluster-{hub}",
                "hub": hub if hub in references else None,
                "members": members,
                "reason": "nearest non-rail graph hub; geometry breaks equal graph-distance ties",
            }
        )
    return records, {reference: f"cluster-{hub}" for reference, hub in assignments.items()}


def _anchor_roles(
    references: set[str], nets: list[JsonRecord], arcs: set[tuple[str, str]]
) -> list[JsonRecord]:
    outgoing = {reference: 0 for reference in references}
    incoming = {reference: 0 for reference in references}
    for source, target in arcs:
        outgoing[source] += 1
        incoming[target] += 1
    records = []
    for reference in sorted(references):
        roles = []
        pin_types = {
            str(node.get("type", "")).casefold()
            for net in nets
            for node in net["nodes"]
            if str(node["reference"]) == reference
        }
        if _CONNECTOR_REF.match(reference):
            roles.append("connector")
        if pin_types & _SOURCE_TYPES:
            roles.append("source")
        if pin_types & _SINK_TYPES:
            roles.append("sink")
        if roles:
            records.append(
                {
                    "reference": reference,
                    "roles": roles,
                    "incoming_arcs": incoming[reference],
                    "outgoing_arcs": outgoing[reference],
                    "net_count": sum(reference in net["references"] for net in nets),
                }
            )
    return records


def _proposed_positions(
    references: set[str],
    old: dict[str, Point],
    sizes: dict[str, tuple[float, float]],
    ranks: dict[str, int],
    cluster_for: dict[str, str],
    fixed: set[str],
    page_extent: tuple[float, float],
) -> dict[str, Point]:
    if not references:
        return {}
    max_width = max((sizes.get(ref, (5.08, 5.08))[0] for ref in references), default=5.08)
    column_spacing = max(25.4, _snap(max_width + 15.24))
    page_width, _page_height = page_extent
    max_rank = max(ranks.values(), default=0)
    if max_rank:
        column_spacing = min(column_spacing, _snap((page_width - 30.48) / max_rank))
    origin_x = 15.24
    origin_y = 15.24
    proposed: dict[str, Point] = {}
    for rank in sorted(set(ranks.values())):
        members = sorted(
            (ref for ref in references if ranks[ref] == rank),
            key=lambda ref: (cluster_for.get(ref, ref), old[ref][1], ref),
        )
        cursor_y = origin_y
        previous_half = 0.0
        for reference in members:
            if reference in fixed:
                proposed[reference] = old[reference]
                continue
            height = sizes.get(reference, (5.08, 5.08))[1]
            half = height / 2
            cursor_y = max(
                cursor_y + previous_half + half + 2.54,
                origin_y + half,
            )
            proposed[reference] = (_snap(origin_x + rank * column_spacing), _snap(cursor_y))
            previous_half = half
    return proposed


def _mst_edges(references: list[str], positions: dict[str, Point]) -> list[tuple[str, str]]:
    if len(references) < 2:
        return []
    connected = {min(references)}
    edges: list[tuple[str, str]] = []
    while len(connected) < len(references):
        candidates = [
            (abs(positions[a][0] - positions[b][0]) + abs(positions[a][1] - positions[b][1]), a, b)
            for a in connected
            for b in references
            if b not in connected
        ]
        _, source, target = min(candidates)
        edges.append((source, target))
        connected.add(target)
    return edges


def _path(start: Point, end: Point) -> tuple[Point, ...]:
    if start[0] == end[0] or start[1] == end[1]:
        return (start, end)
    return (start, (end[0], start[1]), end)


def _wire_topology(
    nets: list[JsonRecord], rails: set[str], positions: dict[str, Point]
) -> tuple[list[JsonRecord], list[tuple[str, Segment]]]:
    records: list[JsonRecord] = []
    all_segments: list[tuple[str, Segment]] = []
    for net in sorted(nets, key=lambda item: str(item["name"])):
        name = str(net["name"])
        references = [ref for ref in net["references"] if ref in positions]
        if len(references) < 2:
            continue
        segments: list[Segment] = []
        if name in rails:
            rail_y = _snap(statistics.median(positions[ref][1] for ref in references))
            rail_x = _snap(statistics.median(positions[ref][0] for ref in references))
            horizontal_cost = (
                max(positions[ref][0] for ref in references)
                - min(positions[ref][0] for ref in references)
                + sum(abs(positions[ref][1] - rail_y) for ref in references)
            )
            vertical_cost = (
                max(positions[ref][1] for ref in references)
                - min(positions[ref][1] for ref in references)
                + sum(abs(positions[ref][0] - rail_x) for ref in references)
            )
            if horizontal_cost <= vertical_cost:
                left = min(positions[ref][0] for ref in references)
                right = max(positions[ref][0] for ref in references)
                if left != right:
                    segments.append(((left, rail_y), (right, rail_y)))
                for reference in references:
                    center = positions[reference]
                    if center[1] != rail_y:
                        segments.append((center, (center[0], rail_y)))
                style = "shared_horizontal_rail"
            else:
                top = min(positions[ref][1] for ref in references)
                bottom = max(positions[ref][1] for ref in references)
                if top != bottom:
                    segments.append(((rail_x, top), (rail_x, bottom)))
                for reference in references:
                    center = positions[reference]
                    if center[0] != rail_x:
                        segments.append((center, (rail_x, center[1])))
                style = "shared_vertical_rail"
        else:
            for source, target in _mst_edges(references, positions):
                points = _path(positions[source], positions[target])
                segments.extend(zip(points, points[1:], strict=False))
            style = "manhattan_minimum_spanning_tree"
        all_segments.extend((name, segment) for segment in segments)
        records.append(
            {
                "net": name,
                "style": style,
                "references": references,
                "segments": [
                    {"start_mm": list(start), "end_mm": list(end)} for start, end in segments
                ],
                "anchor_authority": "component_center_only",
                "applicable": False,
                "refusal": "pin anchors and body-edge exits must be resolved before CAD wiring",
            }
        )
    return records, all_segments


def _segment_crossing(first: Segment, second: Segment) -> bool:
    (ax1, ay1), (ax2, ay2) = first
    (bx1, by1), (bx2, by2) = second
    ah = ay1 == ay2
    bh = by1 == by2
    if ah == bh:
        return False
    horizontal, vertical = (first, second) if ah else (second, first)
    (hx1, hy), (hx2, _) = horizontal
    (vx, vy1), (_, vy2) = vertical
    point = (vx, hy)
    if point in {first[0], first[1], second[0], second[1]}:
        return False
    return min(hx1, hx2) < vx < max(hx1, hx2) and min(vy1, vy2) < hy < max(vy1, vy2)


def _crossings(segments: list[tuple[str, Segment]]) -> int:
    return sum(
        first_net != second_net and _segment_crossing(first, second)
        for index, (first_net, first) in enumerate(segments)
        for second_net, second in segments[index + 1 :]
    )


def _boxes_at(
    references: set[str],
    positions: dict[str, Point],
    old_positions: dict[str, Point],
    old_boxes: dict[str, Box],
) -> dict[str, Box]:
    result = {}
    for reference in references:
        box = old_boxes[reference]
        dx = positions[reference][0] - old_positions[reference][0]
        dy = positions[reference][1] - old_positions[reference][1]
        result[reference] = Box(box.x_min + dx, box.y_min + dy, box.x_max + dx, box.y_max + dy)
    return result


def _overlap_count(boxes: dict[str, Box]) -> int:
    items = sorted(boxes.items())
    return sum(
        first.overlaps(second)
        for index, (_, first) in enumerate(items)
        for _, second in items[index + 1 :]
    )


def _area(boxes: dict[str, Box]) -> float:
    extent = union(boxes.values())
    return round(extent.area, 4) if extent is not None else 0.0


def plan_schematic_graph_placement(
    snapshot: JsonRecord,
    project_dir: str | Path,
    *,
    sheet: str,
    fixed_references: list[str] | None = None,
) -> JsonRecord:
    """Propose symbol centers and conceptual railway wiring for one page."""

    selected_sheet = select_schematic_sheet(snapshot, sheet, "")
    source = Path(project_dir).expanduser().resolve() / str(selected_sheet["file"])
    text = source.read_text(encoding="utf-8", errors="ignore")
    geometry = parse_schematic_geometry(text)
    local_components = {
        str(item["reference"]): item
        for item in snapshot["schematic"]["components"]
        if str(item.get("sheet", "")) == str(selected_sheet["name"])
    }
    geometry_by_ref = {item.reference: item for item in geometry.symbols}
    references = set(local_components) & set(geometry_by_ref)
    missing_geometry = sorted(set(local_components) - references)
    if not references:
        raise ValueError("selected page has no netlisted symbols with parsed geometry")
    old_positions = {
        reference: (geometry_by_ref[reference].x_mm, geometry_by_ref[reference].y_mm)
        for reference in references
    }
    placed = {
        item.reference: item for item in parse_placed_symbols(text) if item.reference in references
    }
    old_boxes = {
        reference: placed[reference].extent()
        if reference in placed
        else Box.from_center(*old_positions[reference], 5.08, 5.08)
        for reference in references
    }
    sizes = {reference: (box.width, box.height) for reference, box in old_boxes.items()}
    nets = _local_nets(snapshot, references)
    rails = _rail_nets(nets, len(references))
    arcs, ambiguities = _directed_arcs(references, nets, rails, old_positions)
    clusters, cluster_for = _functional_clusters(references, nets, rails, old_positions)
    ranks, cycles = _ranks(references, arcs)
    arc_references = {reference for edge in arcs for reference in edge}
    cluster_hub = {record["id"]: record["hub"] for record in clusters}
    for reference in references - arc_references:
        hub = cluster_hub.get(cluster_for[reference])
        if hub in ranks and hub in arc_references:
            ranks[reference] = ranks[str(hub)]
    fixed = set(fixed_references or [])
    unknown_fixed = sorted(fixed - references)
    fixed &= references
    page_extent = parse_paper_extent(text)
    proposed = _proposed_positions(
        references,
        old_positions,
        sizes,
        ranks,
        cluster_for,
        fixed,
        page_extent,
    )
    topology, proposed_segments = _wire_topology(nets, rails, proposed)
    _old_topology, old_conceptual_segments = _wire_topology(nets, rails, old_positions)
    saved_segments = [
        (f"wire-{index}", segment)
        for index, segment in enumerate(
            (
                (start, end)
                for wire in geometry.wires
                for start, end in zip(wire.points, wire.points[1:], strict=False)
            )
        )
    ]
    new_boxes = _boxes_at(references, proposed, old_positions, old_boxes)
    page_box = Box(0.0, 0.0, page_extent[0], page_extent[1])
    offsheet = sorted(
        reference for reference, box in new_boxes.items() if not box.inside(page_box, margin_mm=5.0)
    )
    old_area = _area(old_boxes)
    new_area = _area(new_boxes)
    label_count = len(geometry.labels)
    old_wire_length = sum(math.dist(start, end) for _, (start, end) in old_conceptual_segments)
    new_wire_length = sum(math.dist(start, end) for _, (start, end) in proposed_segments)
    saved_wire_length = sum(math.dist(start, end) for _, (start, end) in saved_segments)
    constraints: list[JsonRecord] = [
        {
            "code": "planning_only",
            "severity": "hard",
            "reason": "no CAD mutation path is exposed by this command",
        },
        {
            "code": "pin_anchor_resolution_required",
            "severity": "hard",
            "reason": "proposed wires terminate at component centers, not verified library pins",
        },
        {
            "code": "labels_preserved",
            "severity": "hard",
            "reason": "label count is unchanged; use the dedicated compaction planner separately",
        },
    ]
    if missing_geometry:
        constraints.append(
            {
                "code": "missing_symbol_geometry",
                "severity": "hard",
                "references": missing_geometry,
                "reason": "symbols without parsed placement geometry were excluded",
            }
        )
    if unknown_fixed:
        constraints.append(
            {
                "code": "unknown_fixed_reference",
                "severity": "warning",
                "references": unknown_fixed,
            }
        )
    if cycles:
        constraints.append(
            {
                "code": "directed_feedback_cycles_collapsed",
                "severity": "warning",
                "groups": cycles,
                "reason": "each strongly connected feedback group shares one signal-flow rank",
            }
        )
    if offsheet:
        constraints.append(
            {
                "code": "candidate_off_sheet",
                "severity": "hard",
                "references": offsheet,
                "page_extent_mm": list(page_extent),
                "reason": "rank column capacity exceeds the saved drawing sheet",
            }
        )
    movement = [
        {
            "reference": reference,
            "value": str(local_components[reference].get("value", "")),
            "rank": ranks[reference],
            "cluster": cluster_for[reference],
            "fixed": reference in fixed,
            "old_mm": list(old_positions[reference]),
            "proposed_mm": list(proposed[reference]),
            "delta_mm": [
                round(proposed[reference][0] - old_positions[reference][0], 4),
                round(proposed[reference][1] - old_positions[reference][1], 4),
            ],
        }
        for reference in sorted(references, key=lambda ref: (ranks[ref], proposed[ref][1], ref))
    ]
    old_cost = {
        "symbol_overlaps": _overlap_count(old_boxes),
        "wire_crossings": _crossings(old_conceptual_segments),
        "total_manhattan_wire_length_mm": round(old_wire_length, 4),
        "label_count": label_count,
        "label_density_per_1000mm2": round(label_count / max(old_area, 1) * 1000, 4),
        "compactness_area_mm2": old_area,
    }
    new_cost = {
        "symbol_overlaps": _overlap_count(new_boxes),
        "wire_crossings": _crossings(proposed_segments),
        "total_manhattan_wire_length_mm": round(new_wire_length, 4),
        "label_count": label_count,
        "label_density_per_1000mm2": round(label_count / max(new_area, 1) * 1000, 4),
        "compactness_area_mm2": new_area,
    }
    regressions = []
    if new_cost["symbol_overlaps"] > old_cost["symbol_overlaps"]:
        regressions.append("symbol_overlaps")
    if new_cost["wire_crossings"] > old_cost["wire_crossings"]:
        regressions.append("wire_crossings")
    if new_wire_length > old_wire_length + 1e-4:
        regressions.append("total_manhattan_wire_length_mm")
    if new_area > old_area + 1e-4:
        regressions.append("compactness_area_mm2")
    if offsheet:
        regressions.append("sheet_bounds")
    old_score = (
        old_cost["symbol_overlaps"],
        old_cost["wire_crossings"],
        old_cost["total_manhattan_wire_length_mm"],
        old_cost["compactness_area_mm2"],
    )
    new_score = (
        new_cost["symbol_overlaps"],
        new_cost["wire_crossings"],
        new_cost["total_manhattan_wire_length_mm"],
        new_cost["compactness_area_mm2"],
    )
    improves_lexicographically = new_score < old_score
    if not improves_lexicographically:
        regressions.append("lexicographic_score_not_improved")
    regressions = sorted(set(regressions))
    if regressions:
        constraints.append(
            {
                "code": "cost_regression",
                "severity": "hard",
                "metrics": regressions,
                "reason": "candidate is reported for diagnosis but refused as an apply baseline",
            }
        )
    accepted = not regressions
    return {
        "schema_version": "1.0",
        "status": "proposal" if accepted else "blocked",
        "read_only": True,
        "project": snapshot["project"],
        "sheet": {"name": selected_sheet["name"], "file": selected_sheet["file"]},
        "graph": {
            "components": len(references),
            "nets": len(nets),
            "directed_arcs": [list(edge) for edge in sorted(arcs)],
            "rail_nets": [
                {
                    "name": name,
                    "fanout": next(len(net["references"]) for net in nets if net["name"] == name),
                    "reason": "power-name or high-fanout shared-net rail",
                }
                for name in sorted(rails)
            ],
            "direction_ambiguities": ambiguities,
            "feedback_groups": cycles,
        },
        "anchors": _anchor_roles(references, nets, arcs),
        "functional_clusters": clusters,
        "placements": movement if accepted else [],
        "wire_topology": topology if accepted else [],
        "diagnostic_candidate": None
        if accepted
        else {
            "placements": movement,
            "wire_topology": topology,
            "usable": False,
            "reason": "hard acceptance gate rejected this candidate",
        },
        "cost": {
            "authority": {
                "old_wires": "conceptual component-center railway topology at saved positions",
                "new_wires": "same conceptual component-center topology at proposed positions",
            },
            "saved_schematic_geometry": {
                "wire_crossings": _crossings(saved_segments),
                "wire_length_mm": round(saved_wire_length, 4),
                "note": "observed geometry only; not mixed into conceptual old/new acceptance",
            },
            "old": old_cost,
            "new": new_cost,
        },
        "constraints": constraints,
        "acceptance": {
            "accepted": accepted,
            "lexicographic_order": [
                "symbol_overlaps",
                "wire_crossings",
                "total_manhattan_wire_length_mm",
                "compactness_area_mm2",
            ],
            "old_score": list(old_score),
            "new_score": list(new_score),
            "regressions": regressions,
        },
        "future_apply_gate": [
            "resolve every topology endpoint to an exact library pin anchor",
            "route around measured bodies and fields without unplanned intersections",
            "prove complete exported-netlist equality",
            "pass ERC, visual QA, and render review before source promotion",
        ],
    }


def format_schematic_graph_placement(plan: JsonRecord) -> str:
    """Render the proposal as a compact, stable railway planning report."""

    graph = plan["graph"]
    old = plan["cost"]["old"]
    new = plan["cost"]["new"]
    lines = [
        f"SCHEMATIC RAILWAY PLAN  status={plan['status']} "
        f"sheet={plan['sheet']['name']} read_only=true",
        f"graph parts={graph['components']} nets={graph['nets']} rails={len(graph['rail_nets'])} "
        f"arcs={len(graph['directed_arcs'])} clusters={len(plan['functional_clusters'])}",
        "RANKS" if plan["acceptance"]["accepted"] else "DIAGNOSTIC RANKS (rejected candidate)",
    ]
    if not plan["acceptance"]["accepted"]:
        lines.append(
            "BLOCKED candidate hidden from usable placements; regressions="
            f"{plan['acceptance']['regressions']}"
        )
    by_rank: dict[int, list[str]] = {}
    rank_source = plan["placements"] or plan["diagnostic_candidate"]["placements"]
    for item in rank_source:
        by_rank.setdefault(int(item["rank"]), []).append(str(item["reference"]))
    for rank, references in sorted(by_rank.items()):
        lines.append(f"  {rank}: {' '.join(references)}")
    lines.append("RAILS")
    lines.extend(f"  {item['name']} fanout={item['fanout']}" for item in graph["rail_nets"])
    if plan["placements"]:
        lines.append("PLACEMENT")
        lines.extend(
            f"  {item['reference']:<12} {item['old_mm']} -> {item['proposed_mm']} "
            f"rank={item['rank']} {item['cluster']}{' FIXED' if item['fixed'] else ''}"
            for item in plan["placements"]
        )
    lines.append(
        "COST "
        f"overlap {old['symbol_overlaps']}->{new['symbol_overlaps']}  "
        f"cross {old['wire_crossings']}->{new['wire_crossings']}  "
        f"wire {old['total_manhattan_wire_length_mm']}->{new['total_manhattan_wire_length_mm']}mm  "
        f"labels {old['label_count']}->{new['label_count']}  "
        f"area {old['compactness_area_mm2']}->{new['compactness_area_mm2']}mm2"
    )
    lines.append("REFUSAL pin anchors unresolved; topology is conceptual and cannot be applied")
    return "\n".join(lines)
