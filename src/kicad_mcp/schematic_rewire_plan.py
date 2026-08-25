"""Read-only planner for conservative local schematic label compaction.

This module deliberately stops at evidence and candidate geometry.  It never
rewrites KiCad source.  A future applier must prove the exported netlist is
unchanged before promoting any proposal produced here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models.visual_qa import LabelItem, parse_labels, parse_placed_symbols
from .schematic_spatial import (
    SchematicGeometry,
    SpatialLabel,
    parse_schematic_geometry,
    select_schematic_sheet,
)
from .utils.geometry import Box

type JsonRecord = dict[str, Any]
type Point = tuple[float, float]
type Segment = tuple[Point, Point]

_TOLERANCE = 1e-4
_SYMBOL_CLEARANCE_MM = 0.5
_CLUSTER_GAP_MM = 16.0
_MAX_LOCAL_ROUTE_MM = 20.0


def _canonical_net_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _point_key(point: Point) -> tuple[float, float]:
    return round(point[0], 4), round(point[1], 4)


def _same_point(first: Point, second: Point) -> bool:
    return abs(first[0] - second[0]) <= _TOLERANCE and abs(first[1] - second[1]) <= _TOLERANCE


def _wire_segments(geometry: SchematicGeometry) -> list[Segment]:
    return [
        (start, end)
        for wire in geometry.wires
        for start, end in zip(wire.points, wire.points[1:], strict=False)
    ]


def _point_on_segment(point: Point, segment: Segment) -> bool:
    (x, y), ((x1, y1), (x2, y2)) = point, segment
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > _TOLERANCE:
        return False
    return (
        min(x1, x2) - _TOLERANCE <= x <= max(x1, x2) + _TOLERANCE
        and min(y1, y2) - _TOLERANCE <= y <= max(y1, y2) + _TOLERANCE
    )


def _segment_intersection(first: Segment, second: Segment) -> tuple[str, Point | None]:
    """Return ``none``, ``point``, or positive-length ``overlap`` for Manhattan lines."""

    (ax1, ay1), (ax2, ay2) = first
    (bx1, by1), (bx2, by2) = second
    first_horizontal = abs(ay1 - ay2) <= _TOLERANCE
    second_horizontal = abs(by1 - by2) <= _TOLERANCE
    if first_horizontal and second_horizontal:
        if abs(ay1 - by1) > _TOLERANCE:
            return "none", None
        low = max(min(ax1, ax2), min(bx1, bx2))
        high = min(max(ax1, ax2), max(bx1, bx2))
        if high < low - _TOLERANCE:
            return "none", None
        if high - low > _TOLERANCE:
            return "overlap", None
        return "point", (low, ay1)
    if not first_horizontal and not second_horizontal:
        if abs(ax1 - bx1) > _TOLERANCE:
            return "none", None
        low = max(min(ay1, ay2), min(by1, by2))
        high = min(max(ay1, ay2), max(by1, by2))
        if high < low - _TOLERANCE:
            return "none", None
        if high - low > _TOLERANCE:
            return "overlap", None
        return "point", (ax1, low)
    horizontal, vertical = (first, second) if first_horizontal else (second, first)
    (hx1, hy), (hx2, _) = horizontal
    (vx, vy1), (_, vy2) = vertical
    point = (vx, hy)
    if (
        min(hx1, hx2) - _TOLERANCE <= vx <= max(hx1, hx2) + _TOLERANCE
        and min(vy1, vy2) - _TOLERANCE <= hy <= max(vy1, vy2) + _TOLERANCE
    ):
        return "point", point
    return "none", None


def _segment_crosses_box(segment: Segment, box: Box) -> bool:
    (x1, y1), (x2, y2) = segment
    if abs(y1 - y2) <= _TOLERANCE:
        return box.y_min < y1 < box.y_max and max(min(x1, x2), box.x_min) < min(
            max(x1, x2), box.x_max
        )
    return box.x_min < x1 < box.x_max and max(min(y1, y2), box.y_min) < min(max(y1, y2), box.y_max)


def _path_points(start: Point, end: Point) -> list[tuple[Point, ...]]:
    if _same_point(start, end):
        return [(start,)]
    if abs(start[0] - end[0]) <= _TOLERANCE or abs(start[1] - end[1]) <= _TOLERANCE:
        return [(start, end)]
    candidates = [(start, (end[0], start[1]), end), (start, (start[0], end[1]), end)]
    return sorted(candidates)


def _segments(points: tuple[Point, ...]) -> list[Segment]:
    return [
        (start, end)
        for start, end in zip(points, points[1:], strict=False)
        if not _same_point(start, end)
    ]


def _label_id(label: SpatialLabel, ordinal: int) -> str:
    return label.uuid or (f"{label.kind}:{label.name}@{label.x_mm:.4f},{label.y_mm:.4f}#{ordinal}")


def _label_record(label: SpatialLabel, ordinal: int, reference_at: Point) -> JsonRecord:
    return {
        "id": _label_id(label, ordinal),
        "uuid": label.uuid or None,
        "name": label.name,
        "kind": label.kind,
        "anchor_mm": [round(label.x_mm, 4), round(label.y_mm, 4)],
        "rotation_deg": label.rotation,
        "distance_to_reference_mm": round(
            math.hypot(label.x_mm - reference_at[0], label.y_mm - reference_at[1]), 4
        ),
    }


def _matching_label_item(label: SpatialLabel, items: list[LabelItem]) -> LabelItem | None:
    kind_map = {"label": "local", "global_label": "global", "hierarchical_label": "hierarchical"}
    expected_kind = kind_map[label.kind]
    return next(
        (
            item
            for item in items
            if item.text == label.name
            and item.kind == expected_kind
            and _same_point((item.x, item.y), (label.x_mm, label.y_mm))
        ),
        None,
    )


def _overlap_count(items: list[LabelItem]) -> int:
    count = 0
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            if first.box().intersection_area(second.box()) > 0:
                count += 1
    return count


def _spatial_clusters(labels: list[SpatialLabel], gap_mm: float) -> list[list[SpatialLabel]]:
    """Return deterministic connected components under a local-distance threshold."""

    remaining = set(range(len(labels)))
    clusters: list[list[SpatialLabel]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        members = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            near = {
                candidate
                for candidate in remaining
                if math.hypot(
                    labels[current].x_mm - labels[candidate].x_mm,
                    labels[current].y_mm - labels[candidate].y_mm,
                )
                <= gap_mm
            }
            remaining -= near
            members |= near
            frontier.extend(sorted(near))
        clusters.append([labels[index] for index in sorted(members)])
    return clusters


def _evaluate_path(
    points: tuple[Point, ...],
    *,
    existing_wires: list[Segment],
    symbol_obstacles: list[tuple[str, Box]],
    label_obstacles: list[tuple[str, Box]],
    ignored_label_ids: set[str],
    accepted_segments: list[Segment],
) -> list[JsonRecord]:
    reasons: list[JsonRecord] = []
    path_segments = _segments(points)
    endpoints = {points[0], points[-1]}
    for segment_index, segment in enumerate(path_segments):
        for wire_index, wire in enumerate(existing_wires):
            kind, point = _segment_intersection(segment, wire)
            if kind == "overlap":
                reasons.append(
                    {
                        "code": "existing_wire_overlap",
                        "segment": segment_index,
                        "existing_wire": wire_index,
                        "reason": (
                            "candidate would duplicate or obscure an existing wire of unproven net"
                        ),
                    }
                )
            elif (
                kind == "point"
                and point is not None
                and not any(_same_point(point, endpoint) for endpoint in endpoints)
            ):
                reasons.append(
                    {
                        "code": "unknown_wire_intersection",
                        "segment": segment_index,
                        "existing_wire": wire_index,
                        "at_mm": [round(point[0], 4), round(point[1], 4)],
                        "reason": "crossed wire ownership cannot be proven from geometry alone",
                    }
                )
        for reference, box in symbol_obstacles:
            if _segment_crosses_box(segment, box):
                reasons.append(
                    {
                        "code": "symbol_body_intersection",
                        "segment": segment_index,
                        "reference": reference,
                        "reason": "candidate enters a measured symbol body keepout",
                    }
                )
        for label_id, box in label_obstacles:
            if label_id not in ignored_label_ids and _segment_crosses_box(segment, box):
                reasons.append(
                    {
                        "code": "label_text_intersection",
                        "segment": segment_index,
                        "label_id": label_id,
                        "reason": "candidate would run through visible label text",
                    }
                )
        for accepted_index, accepted in enumerate(accepted_segments):
            kind, point = _segment_intersection(segment, accepted)
            shared_endpoint = point is not None and any(
                _same_point(point, endpoint) for endpoint in endpoints
            )
            if kind == "overlap" or (kind == "point" and not shared_endpoint):
                reasons.append(
                    {
                        "code": "candidate_route_crossing",
                        "segment": segment_index,
                        "accepted_segment": accepted_index,
                        "reason": "independent proposals would require an unplanned junction",
                    }
                )
    unique: dict[tuple[object, ...], JsonRecord] = {}
    for reason in reasons:
        key = (
            reason["code"],
            reason.get("segment"),
            reason.get("existing_wire"),
            reason.get("reference"),
            reason.get("label_id"),
            tuple(reason.get("at_mm", [])),
        )
        unique[key] = reason
    return list(unique.values())


def plan_label_compaction(
    snapshot: JsonRecord,
    project_dir: str | Path,
    *,
    reference: str,
    sheet: str = "",
    radius_mm: float = 35.0,
) -> JsonRecord:
    """Plan same-kind, same-net label consolidation near one placed component."""

    if not reference:
        raise ValueError("reference is required")
    if not 5.0 <= radius_mm <= 100.0:
        raise ValueError("radius_mm must be between 5 and 100")
    selected_sheet = select_schematic_sheet(snapshot, sheet, reference)
    source = Path(project_dir).expanduser().resolve() / str(selected_sheet["file"])
    text = source.read_text(encoding="utf-8", errors="ignore")
    geometry = parse_schematic_geometry(text)
    component = next(
        (item for item in geometry.symbols if item.reference.casefold() == reference.casefold()),
        None,
    )
    if component is None:
        raise ValueError(f"reference was not found on selected page: {reference}")
    reference_at = (component.x_mm, component.y_mm)
    connected_nets = sorted(
        {
            str(net["name"])
            for net in snapshot["schematic"]["nets"]
            if any(
                str(node["reference"]).casefold() == reference.casefold() for node in net["nodes"]
            )
        }
    )
    canonical_connected = {_canonical_net_name(name) for name in connected_nets}
    nearby = [
        label
        for label in geometry.labels
        if label.name in canonical_connected
        and math.hypot(label.x_mm - reference_at[0], label.y_mm - reference_at[1]) <= radius_mm
    ]
    grouped: dict[tuple[str, str], list[SpatialLabel]] = {}
    for label in nearby:
        grouped.setdefault((label.name, label.kind), []).append(label)

    visual_labels = parse_labels(text)
    placed_symbols = parse_placed_symbols(text)
    symbol_obstacles = [
        (item.reference, item.body.expanded(_SYMBOL_CLEARANCE_MM)) for item in placed_symbols
    ]
    spatial_ordinals = {id(label): index for index, label in enumerate(geometry.labels, start=1)}
    label_obstacles: list[tuple[str, Box]] = []
    item_by_spatial_id: dict[int, LabelItem] = {}
    for label in geometry.labels:
        item = _matching_label_item(label, visual_labels)
        if item is not None:
            item_by_spatial_id[id(label)] = item
            label_obstacles.append(
                (_label_id(label, spatial_ordinals[id(label)]), item.box().expanded(0.25))
            )

    existing_wires = _wire_segments(geometry)
    accepted_segments: list[Segment] = []
    clusters: list[JsonRecord] = []
    removed_ids: set[str] = set()
    all_refusals: list[JsonRecord] = []
    added_wire_length = 0.0

    local_clusters: list[tuple[str, str, int, list[SpatialLabel]]] = []
    for (net_name, kind), labels in sorted(grouped.items()):
        crowded = [
            cluster
            for cluster in _spatial_clusters(
                sorted(labels, key=lambda item: (item.x_mm, item.y_mm, item.uuid)),
                _CLUSTER_GAP_MM,
            )
            if len(cluster) >= 2
        ]
        local_clusters.extend(
            (net_name, kind, cluster_index, cluster)
            for cluster_index, cluster in enumerate(crowded, start=1)
        )
    for net_name, kind, cluster_index, labels in local_clusters:
        ordered = sorted(
            labels,
            key=lambda label: (
                math.hypot(label.x_mm - reference_at[0], label.y_mm - reference_at[1]),
                label.x_mm,
                label.y_mm,
                label.uuid,
            ),
        )
        retained = ordered[0]
        cluster: JsonRecord = {
            "net": net_name,
            "label_kind": kind,
            "cluster_index": cluster_index,
            "cluster_gap_mm": _CLUSTER_GAP_MM,
            "instances": [
                _label_record(label, spatial_ordinals[id(label)], reference_at) for label in ordered
            ],
            "retain": _label_record(retained, spatial_ordinals[id(retained)], reference_at),
            "proposals": [],
        }
        if kind == "hierarchical_label":
            refusal = {
                "code": "hierarchical_interface_label",
                "net": net_name,
                "reason": (
                    "hierarchical labels are sheet-interface contracts and are never compacted"
                ),
            }
            cluster["status"] = "refused"
            cluster["refusals"] = [refusal]
            all_refusals.append(refusal)
            clusters.append(cluster)
            continue

        for redundant in ordered[1:]:
            redundant_id = _label_id(redundant, spatial_ordinals[id(redundant)])
            retained_id = _label_id(retained, spatial_ordinals[id(retained)])
            proposal: JsonRecord = {
                "remove": _label_record(redundant, spatial_ordinals[id(redundant)], reference_at),
                "connect_to": retained_id,
                "candidate_routes": [],
            }
            redundant_at = (redundant.x_mm, redundant.y_mm)
            retained_at = (retained.x_mm, retained.y_mm)
            unattached = [
                label_id
                for label_id, point in ((redundant_id, redundant_at), (retained_id, retained_at))
                if not any(_point_on_segment(point, wire) for wire in existing_wires)
            ]
            if unattached and not _same_point(redundant_at, retained_at):
                reasons = [
                    {
                        "code": "label_anchor_attachment_unproven",
                        "label_id": label_id,
                        "reason": (
                            "anchor does not touch a parsed wire; it may be a direct pin attachment"
                        ),
                    }
                    for label_id in unattached
                ]
                proposal["status"] = "refused"
                proposal["refusals"] = reasons
                all_refusals.extend({"net": net_name, **reason} for reason in reasons)
                cluster["proposals"].append(proposal)
                continue

            selected: tuple[Point, ...] | None = None
            for points in _path_points(redundant_at, retained_at):
                route_length = sum(math.dist(start, end) for start, end in _segments(points))
                reasons = _evaluate_path(
                    points,
                    existing_wires=existing_wires,
                    symbol_obstacles=symbol_obstacles,
                    label_obstacles=label_obstacles,
                    ignored_label_ids={redundant_id, retained_id},
                    accepted_segments=accepted_segments,
                )
                if route_length > _MAX_LOCAL_ROUTE_MM:
                    reasons.append(
                        {
                            "code": "nonlocal_route_length",
                            "length_mm": round(route_length, 4),
                            "limit_mm": _MAX_LOCAL_ROUTE_MM,
                            "reason": "candidate is too long to qualify as a local compaction",
                        }
                    )
                candidate = {
                    "points_mm": [[round(x, 4), round(y, 4)] for x, y in points],
                    "segments": len(_segments(points)),
                    "length_mm": round(route_length, 4),
                    "safe": not reasons,
                    "refusals": reasons,
                }
                proposal["candidate_routes"].append(candidate)
                if selected is None and not reasons:
                    selected = points
            if selected is None:
                proposal["status"] = "refused"
                proposal["refusals"] = [
                    reason
                    for candidate in proposal["candidate_routes"]
                    for reason in candidate["refusals"]
                ]
                all_refusals.extend({"net": net_name, **reason} for reason in proposal["refusals"])
            else:
                chosen_segments = _segments(selected)
                accepted_segments.extend(chosen_segments)
                length = sum(math.dist(start, end) for start, end in chosen_segments)
                added_wire_length += length
                removed_ids.add(redundant_id)
                proposal["status"] = "planned"
                proposal["selected_route"] = {
                    "points_mm": [[round(x, 4), round(y, 4)] for x, y in selected],
                    "segments": len(chosen_segments),
                    "length_mm": round(length, 4),
                }
            cluster["proposals"].append(proposal)
        proposal_statuses = {item["status"] for item in cluster["proposals"]}
        cluster["status"] = (
            "planned"
            if proposal_statuses == {"planned"}
            else "refused"
            if proposal_statuses == {"refused"}
            else "partial"
        )
        clusters.append(cluster)

    local_visual = [
        item
        for item in visual_labels
        if math.hypot(item.x - reference_at[0], item.y - reference_at[1]) <= radius_mm
    ]
    removed_visual_ids = {
        id(item_by_spatial_id[id(spatial)])
        for spatial in nearby
        if _label_id(spatial, spatial_ordinals[id(spatial)]) in removed_ids
        and id(spatial) in item_by_spatial_id
    }
    after_visual = [item for item in local_visual if id(item) not in removed_visual_ids]
    before_overlap = _overlap_count(local_visual)
    after_overlap = _overlap_count(after_visual)
    planned_count = len(removed_ids)
    cluster_count = len(clusters)
    status = (
        "noop"
        if cluster_count == 0
        else "planned"
        if planned_count > 0 and not all_refusals
        else "review"
        if planned_count > 0
        else "blocked"
    )
    area = math.pi * radius_mm**2
    return {
        "schema_version": "1.0",
        "status": status,
        "read_only": True,
        "project": snapshot["project"],
        "sheet": {
            "name": selected_sheet["name"],
            "file": selected_sheet["file"],
        },
        "reference": {
            "reference": component.reference,
            "anchor_mm": [component.x_mm, component.y_mm],
            "radius_mm": radius_mm,
            "connected_nets": connected_nets,
        },
        "planner_limits": {
            "cluster_gap_mm": _CLUSTER_GAP_MM,
            "max_local_route_mm": _MAX_LOCAL_ROUTE_MM,
        },
        "clusters": clusters,
        "refusals": all_refusals,
        "prediction": {
            "local_labels_before": len(local_visual),
            "local_labels_after": len(after_visual),
            "label_count_delta": len(after_visual) - len(local_visual),
            "density_per_1000mm2_before": round(len(local_visual) / area * 1000, 4),
            "density_per_1000mm2_after": round(len(after_visual) / area * 1000, 4),
            "overlap_pairs_before": before_overlap,
            "overlap_pairs_after": after_overlap,
            "overlap_pair_delta": after_overlap - before_overlap,
            "candidate_wire_length_added_mm": round(added_wire_length, 4),
        },
        "invariants": [
            "planning_only_no_source_mutation",
            "only_same-name_same-kind labels are considered one cluster",
            "every removed label must have a selected wire path to the retained anchor",
            "future apply must preserve the complete KiCad exported netlist",
            "future apply must pass ERC and visual QA before promotion",
        ],
    }


def format_label_compaction_plan(plan: JsonRecord) -> str:
    """Render a compact terminal summary while retaining exact anchors."""

    reference = plan["reference"]
    lines = [
        f"LABEL COMPACTION PLAN  status={plan['status']}  read_only=true",
        f"sheet={plan['sheet']['name']} ref={reference['reference']} "
        f"anchor={reference['anchor_mm']} radius={reference['radius_mm']}mm",
    ]
    if not plan["clusters"]:
        lines.append("no crowded same-net/same-kind label clusters found")
    for cluster in plan["clusters"]:
        lines.append(
            f"{cluster['status'].upper():7} net={cluster['net']} kind={cluster['label_kind']} "
            f"instances={len(cluster['instances'])} retain={cluster['retain']['id']}"
        )
        for proposal in cluster["proposals"]:
            remove = proposal["remove"]
            if proposal["status"] == "planned":
                route = proposal["selected_route"]
                lines.append(
                    f"  REMOVE {remove['id']} anchor={remove['anchor_mm']} "
                    f"WIRE {route['points_mm']} length={route['length_mm']}mm"
                )
            else:
                codes = sorted({reason["code"] for reason in proposal.get("refusals", [])})
                lines.append(
                    f"  REFUSE {remove['id']} anchor={remove['anchor_mm']} reasons={codes}"
                )
    prediction = plan["prediction"]
    lines.append(
        "PREDICT "
        f"labels {prediction['local_labels_before']}->{prediction['local_labels_after']} "
        f"overlaps {prediction['overlap_pairs_before']}->{prediction['overlap_pairs_after']} "
        f"wire+={prediction['candidate_wire_length_added_mm']}mm"
    )
    lines.append("APPLY disabled; future apply must prove netlist equality + ERC + visual QA")
    return "\n".join(lines)
