"""Schematic cosmetic auto-fixers (Visual Excellence Loop, Phase B).

These tools *improve the appearance* of a schematic — snapping to grid,
straightening wires, de-conflicting labels, uprighting power symbols, and
normalising fonts — while treating **electrical connectivity as a hard
invariant**.

KiCad connectivity is pure coordinate coincidence (a pin tip, a wire end, a
label anchor and a junction are connected iff they share a coordinate). Any
mutation that moves connectivity-bearing geometry can therefore silently break a
net while leaving every object present — which the schematic structural-loss
guard (a presence/count check) would *not* catch. So every fixer here routes
through :func:`_run_cosmetic_fix`, which:

1. computes a coordinate-independent connectivity signature *before*;
2. applies the mutation to an in-memory copy and re-derives the signature;
3. in ``apply=False`` (the default, dry-run) mode reports what would change and
   whether connectivity is preserved — never writing;
4. in ``apply=True`` mode refuses (and writes nothing) if the signature changed,
   otherwise commits through the transactional writer (checkpoint + structural
   guard + mutation snapshot for ``sch_render_visual_diff``).

The signature is derived from :func:`_build_connectivity_groups`, which is pure
Python and headless, so the safety check runs anywhere — including CI without a
live KiCad.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..models import visual_qa
from ..utils.sexpr import _extract_block
from .metadata import headless_compatible
from .schematic import (
    _connectivity_signature_for_text,
    _fmt_mm,
    _get_schematic_file,
    _label_justify_from_block,
    _set_label_justify,
    _shift_symbol_block,
    _snap_schematic_coord,
    _split_lib_id,
    get_pin_positions,
    transactional_write,
)

# A wire whose minor axis is within this distance of zero is treated as
# "almost orthogonal" and straightened onto the axis.
STRAIGHTEN_TOLERANCE_MM = 0.5
# Font sizes differing from the sheet's dominant size by more than this are
# normalised onto it.
FONT_SIZE_TOLERANCE_MM = 0.01

_AT3_RE = re.compile(r"\(at\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
_AT2_RE = re.compile(r"\(at\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
_XY_RE = re.compile(r"\(xy\s+(-?[\d.]+)\s+(-?[\d.]+)\)")
_FONT_SIZE_RE = re.compile(r"\(size\s+(-?[\d.]+)\s+(-?[\d.]+)\)")


class CosmeticFixError(ValueError):
    """A cosmetic mutation could not be planned (bad input / nothing to act on)."""


# ---------------------------------------------------------------------------
# Connectivity invariant
# ---------------------------------------------------------------------------

ConnectivitySignature = frozenset[tuple[frozenset[str], frozenset[tuple[str, str]]]]


def _connectivity_signature(text: str) -> ConnectivitySignature:
    """Return a coordinate-independent connectivity signature for ``text``.

    Each electrical group contributes ``(net names, {(ref, pin)})``. Because the
    signature captures *which* pins/labels are common — not *where* they sit — a
    layout-only change (translate, rotate-about-pin, justify flip) yields an
    identical signature, while any change that detaches a pin or merges two nets
    changes it. That is exactly the safety predicate the fixers need.
    """

    return _connectivity_signature_for_text(text)


def _run_cosmetic_fix(
    fix_name: str,
    mutator: Callable[[str], tuple[str, list[dict[str, Any]]]],
    *,
    apply: bool,
) -> str:
    """Plan (and optionally apply) a cosmetic mutation under the connectivity invariant."""

    sch_file = _get_schematic_file()
    before = sch_file.read_text(encoding="utf-8", errors="ignore")
    try:
        after, changes = mutator(before)
    except CosmeticFixError as exc:
        return json.dumps({"fix": fix_name, "status": "error", "message": str(exc)}, indent=2)

    report: dict[str, Any] = {
        "fix": fix_name,
        "change_count": len(changes),
        "changes": changes,
    }

    if not changes or before == after:
        report["status"] = "no_change"
        report["message"] = "The schematic is already clean for this check."
        return json.dumps(report, indent=2)

    before_sig = _connectivity_signature(before)
    after_sig = _connectivity_signature(after)
    connectivity_preserved = before_sig == after_sig
    report["connectivity_preserved"] = connectivity_preserved

    if not apply:
        report["status"] = "dry_run"
        report["message"] = (
            "Dry run — no file was written. Re-run with apply=true to commit."
            if connectivity_preserved
            else "Dry run — applying this would change net connectivity, so apply would be refused."
        )
        return json.dumps(report, indent=2)

    if not connectivity_preserved:
        report["status"] = "refused"
        report["message"] = (
            "Refused to apply: the mutation would change net connectivity. Nothing was written."
        )
        return json.dumps(report, indent=2)

    def _committing_mutator(current: str) -> str:
        mutated, _ = mutator(current)
        return mutated

    try:
        reload_note = transactional_write(_committing_mutator, sch_file=sch_file)
    except ValueError as exc:
        report["status"] = "error"
        report["message"] = str(exc)
        return json.dumps(report, indent=2)

    report["status"] = "applied"
    report["message"] = f"Applied {len(changes)} change(s). {_reload_schematic_note(reload_note)}"
    return json.dumps(report, indent=2)


def _reload_schematic_note(write_result: str) -> str:
    text = str(write_result or "").strip()
    return text or "Schematic updated."


# ---------------------------------------------------------------------------
# Block iteration + span rewriting
# ---------------------------------------------------------------------------


def _iter_blocks(content: str, pattern: re.Pattern[str]) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, block)`` for every balanced block whose head matches."""

    for match in pattern.finditer(content):
        block, length = _extract_block(content, match.start())
        if not block:
            continue
        yield match.start(), match.start() + length, block


def _apply_span_replacements(content: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping ``(start, end, text)`` replacements, right-to-left."""

    result = content
    for start, end, new_text in sorted(replacements, key=lambda item: item[0], reverse=True):
        result = result[:start] + new_text + result[end:]
    return result


_SYMBOL_HEAD_RE = re.compile(r"\(symbol\s+\(lib_")
_LABEL_HEAD_RE = re.compile(r"\((?:label|global_label|hierarchical_label)\b")
_WIRE_HEAD_RE = re.compile(r"\(wire\b")
_JUNCTION_HEAD_RE = re.compile(r"\(junction\b")


def _symbol_placement(block: str) -> tuple[float, float, int] | None:
    match = _AT3_RE.search(block)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), int(round(float(match.group(3))))


def _symbol_lib_id(block: str) -> str:
    match = re.search(r'\(lib_id\s+"([^"]+)"', block)
    return match.group(1) if match else ""


def _symbol_reference(block: str) -> str:
    match = re.search(r'\(property\s+"Reference"\s+"([^"]*)"', block)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Mutators — each returns (new_content, [change dicts])
# ---------------------------------------------------------------------------


def _mutate_align_to_grid(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Snap symbol origins, label anchors, wire ends and junctions onto the grid.

    Symbols are translated as a whole (origin + fields + pins by the same delta),
    so a symbol whose pins were on-grid stays on-grid. Because coincident points
    always snap to the same grid point, exact pin/wire coincidences are preserved
    — the invariant check confirms it and blocks the rare on-segment exception.
    """

    changes: list[dict[str, Any]] = []
    repls: list[tuple[int, int, str]] = []

    for start, end, block in _iter_blocks(content, _SYMBOL_HEAD_RE):
        placement = _symbol_placement(block)
        if placement is None:
            continue
        x, y, _ = placement
        sx, sy = _snap_schematic_coord(x), _snap_schematic_coord(y)
        if (sx, sy) == (x, y):
            continue
        repls.append((start, end, _shift_symbol_block(block, sx - x, sy - y)))
        changes.append(
            {
                "kind": "symbol",
                "ref": _symbol_reference(block),
                "from": [x, y],
                "to": [sx, sy],
            }
        )

    for start, end, block in _iter_blocks(content, _LABEL_HEAD_RE):
        match = _AT3_RE.search(block)
        if match is None:
            continue
        x, y = float(match.group(1)), float(match.group(2))
        sx, sy = _snap_schematic_coord(x), _snap_schematic_coord(y)
        if (sx, sy) == (x, y):
            continue
        new_at = f"(at {_fmt_mm(sx)} {_fmt_mm(sy)} {match.group(3)})"
        new_block = block[: match.start()] + new_at + block[match.end() :]
        repls.append((start, end, new_block))
        changes.append({"kind": "label", "from": [x, y], "to": [sx, sy]})

    for start, end, block in _iter_blocks(content, _JUNCTION_HEAD_RE):
        match = _AT2_RE.search(block)
        if match is None:
            continue
        x, y = float(match.group(1)), float(match.group(2))
        sx, sy = _snap_schematic_coord(x), _snap_schematic_coord(y)
        if (sx, sy) == (x, y):
            continue
        new_at = f"(at {_fmt_mm(sx)} {_fmt_mm(sy)})"
        new_block = block[: match.start()] + new_at + block[match.end() :]
        repls.append((start, end, new_block))
        changes.append({"kind": "junction", "from": [x, y], "to": [sx, sy]})

    for start, end, block in _iter_blocks(content, _WIRE_HEAD_RE):
        new_block, moved = _snap_xy_points(block)
        if not moved:
            continue
        repls.append((start, end, new_block))
        changes.append({"kind": "wire", "points_moved": moved})

    return _apply_span_replacements(content, repls), changes


def _snap_xy_points(block: str) -> tuple[str, int]:
    moved = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal moved
        x, y = float(match.group(1)), float(match.group(2))
        sx, sy = _snap_schematic_coord(x), _snap_schematic_coord(y)
        if (sx, sy) != (x, y):
            moved += 1
        return f"(xy {_fmt_mm(sx)} {_fmt_mm(sy)})"

    return _XY_RE.sub(repl, block), moved


def _mutate_straighten_wires(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Snap almost-orthogonal wire segments exactly onto the horizontal/vertical axis."""

    changes: list[dict[str, Any]] = []
    repls: list[tuple[int, int, str]] = []

    for start, end, block in _iter_blocks(content, _WIRE_HEAD_RE):
        pts = list(_XY_RE.finditer(block))
        if len(pts) != 2:
            continue
        x1, y1 = float(pts[0].group(1)), float(pts[0].group(2))
        x2, y2 = float(pts[1].group(1)), float(pts[1].group(2))
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx <= 1e-6 or dy <= 1e-6:
            continue  # already orthogonal (or a zero-length stub)
        new_x2, new_y2 = x2, y2
        if dy <= STRAIGHTEN_TOLERANCE_MM and dy <= dx:
            new_y2 = y1  # make it horizontal
        elif dx <= STRAIGHTEN_TOLERANCE_MM and dx < dy:
            new_x2 = x1  # make it vertical
        else:
            continue
        second = f"(xy {_fmt_mm(new_x2)} {_fmt_mm(new_y2)})"
        new_block = block[: pts[1].start()] + second + block[pts[1].end() :]
        repls.append((start, end, new_block))
        changes.append({"kind": "wire", "from": [x2, y2], "to": [new_x2, new_y2]})

    return _apply_span_replacements(content, repls), changes


def _mutate_normalize_text_sizes(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Normalise instance/label font sizes onto the sheet's dominant size.

    Connectivity-neutral: only ``(size ...)`` inside placed-symbol property and
    label effects is touched — never the cached ``lib_symbols`` definitions.
    """

    counts: dict[float, int] = {}
    spans: list[tuple[int, int, str]] = []  # (block start, block end, block)
    for pattern in (_SYMBOL_HEAD_RE, _LABEL_HEAD_RE):
        for start, end, block in _iter_blocks(content, pattern):
            spans.append((start, end, block))
            for size_match in _FONT_SIZE_RE.finditer(block):
                size = round(float(size_match.group(1)), 3)
                counts[size] = counts.get(size, 0) + 1
    if not counts:
        return content, []
    dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    changes: list[dict[str, Any]] = []
    repls: list[tuple[int, int, str]] = []
    for start, end, block in spans:
        touched = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal touched
            size = round(float(match.group(1)), 3)
            if abs(size - dominant) <= FONT_SIZE_TOLERANCE_MM:
                return match.group(0)
            touched += 1
            return f"(size {_fmt_mm(dominant)} {_fmt_mm(dominant)})"

        new_block = _FONT_SIZE_RE.sub(repl, block)
        if touched:
            repls.append((start, end, new_block))
            changes.append({"kind": "font", "ref": _symbol_reference(block), "fields": touched})

    return _apply_span_replacements(content, repls), changes


def _mutate_resolve_label_overlaps(content: str) -> tuple[str, list[dict[str, Any]]]:
    """De-conflict label text by optimizing justification, anchor unmoved.

    Because the anchor never moves, connectivity is preserved by construction —
    only the rendered text direction changes. Every physical label instance is
    scored independently. This matters when a net name appears more than once:
    changing the first label with that name may not touch the colliding instance.

    The objective includes label/label and label/symbol-field intersection area,
    using the same geometry model as visual QA. Each accepted step must strictly
    reduce the global overlap score, so the optimizer is deterministic and cannot
    oscillate.
    """

    entries: list[dict[str, Any]] = []
    for start, end, block in _iter_blocks(content, _LABEL_HEAD_RE):
        parsed = visual_qa.parse_labels(block)
        if len(parsed) != 1:
            continue
        entries.append(
            {
                "start": start,
                "end": end,
                "block": block,
                "item": parsed[0],
                "justify": _label_justify_from_block(block),
            }
        )
    if not entries:
        return content, []

    field_boxes = [
        field.box() for symbol in visual_qa.parse_placed_symbols(content) for field in symbol.fields
    ]
    symbol_boxes = [symbol.body for symbol in visual_qa.parse_placed_symbols(content)]

    def overlap_score() -> float:
        items = [entry["item"] for entry in entries]
        score = 0.0
        threshold = visual_qa.TEXT_OVERLAP_MIN_AREA_MM2
        for idx, first in enumerate(items):
            first_box = first.box()
            for second in items[idx + 1 :]:
                area = first_box.intersection_area(second.box())
                if area >= threshold:
                    score += area
            for field_box in field_boxes:
                area = first_box.intersection_area(field_box)
                if area >= threshold:
                    score += area
            for symbol_box in symbol_boxes:
                area = first_box.intersection_area(symbol_box)
                if area >= threshold:
                    score += area
        return score

    initial_score = overlap_score()
    if initial_score <= 0.0:
        return content, []

    candidates = (
        "",
        "left",
        "right",
        "top",
        "bottom",
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    )
    changes_by_index: dict[int, dict[str, Any]] = {}
    current_score = initial_score
    while True:
        best: tuple[float, int, str, visual_qa.LabelItem] | None = None
        for idx, entry in enumerate(entries):
            item = entry["item"]
            old_justify = entry["justify"]
            for candidate in candidates:
                if candidate == old_justify:
                    continue
                candidate_item = visual_qa.LabelItem(
                    item.text,
                    item.x,
                    item.y,
                    item.kind,
                    item.angle,
                    item.font_mm,
                    frozenset(candidate.split()),
                )
                entry["item"] = candidate_item
                candidate_score = overlap_score()
                entry["item"] = item
                if candidate_score >= current_score - 1e-9:
                    continue
                proposal = (candidate_score, idx, candidate, candidate_item)
                if best is None or proposal[:3] < best[:3]:
                    best = proposal
        if best is None:
            break
        current_score, idx, new_justify, candidate_item = best
        entry = entries[idx]
        entry["item"] = candidate_item
        entry["justify"] = new_justify
        changes_by_index[idx] = {
            "kind": "label",
            "ref": candidate_item.text,
            "position": [candidate_item.x, candidate_item.y],
            "justify": new_justify or "center",
        }

    repls: list[tuple[int, int, str]] = []
    for idx in changes_by_index:
        entry = entries[idx]
        new_block = _set_label_justify(entry["block"], entry["justify"])
        if new_block != entry["block"]:
            repls.append((entry["start"], entry["end"], new_block))

    changes = list(changes_by_index.values())
    if changes:
        changes.append(
            {
                "kind": "objective",
                "overlap_area_before_mm2": round(initial_score, 3),
                "overlap_area_after_mm2": round(current_score, 3),
            }
        )
    return _apply_span_replacements(content, repls), changes


def _flip_justify(current: str) -> str:
    words = current.split()
    if "right" in words:
        return " ".join("left" if word == "right" else word for word in words)
    if "left" in words:
        return " ".join("right" if word == "left" else word for word in words)
    # Centered by default: push text to the left of the anchor.
    return "left"


def _mutate_slide_overlapping_labels(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Slide trapped labels along existing wires and optimize text direction.

    Candidate anchors are sampled only on a wire that already contains the
    label's current anchor. Moving a net label anywhere along that same segment
    preserves its electrical attachment; the cosmetic-fix harness independently
    proves the full connectivity signature before allowing a write.
    """

    entries: list[dict[str, Any]] = []
    for start, end, block in _iter_blocks(content, _LABEL_HEAD_RE):
        parsed = visual_qa.parse_labels(block)
        if len(parsed) == 1:
            entries.append(
                {
                    "start": start,
                    "end": end,
                    "block": block,
                    "item": parsed[0],
                    "justify": _label_justify_from_block(block),
                }
            )
    if not entries:
        return content, []

    placed = visual_qa.parse_placed_symbols(content)
    static_boxes = [field.box() for symbol in placed for field in symbol.fields]
    static_boxes.extend(symbol.body for symbol in placed)
    wires = visual_qa.parse_wires(content)
    threshold = visual_qa.TEXT_OVERLAP_MIN_AREA_MM2

    def score() -> float:
        items = [entry["item"] for entry in entries]
        total = 0.0
        for idx, first in enumerate(items):
            first_box = first.box()
            for second in items[idx + 1 :]:
                area = first_box.intersection_area(second.box())
                if area >= threshold:
                    total += area
            for obstacle in static_boxes:
                area = first_box.intersection_area(obstacle)
                if area >= threshold:
                    total += area
        return total

    def contribution(index: int) -> float:
        item = entries[index]["item"]
        item_box = item.box()
        total = 0.0
        for other_idx, other in enumerate(entries):
            if other_idx == index:
                continue
            area = item_box.intersection_area(other["item"].box())
            if area >= threshold:
                total += area
        for obstacle in static_boxes:
            area = item_box.intersection_area(obstacle)
            if area >= threshold:
                total += area
        return total

    def contains(wire: visual_qa.WireSegment, x: float, y: float) -> bool:
        tolerance = 0.05
        if abs(wire.y1 - wire.y2) <= tolerance:
            return (
                abs(y - wire.y1) <= tolerance
                and min(wire.x1, wire.x2) - tolerance <= x <= max(wire.x1, wire.x2) + tolerance
            )
        if abs(wire.x1 - wire.x2) <= tolerance:
            return (
                abs(x - wire.x1) <= tolerance
                and min(wire.y1, wire.y2) - tolerance <= y <= max(wire.y1, wire.y2) + tolerance
            )
        return False

    def sample_points(item: visual_qa.LabelItem) -> list[tuple[float, float]]:
        points: set[tuple[float, float]] = set()
        grid = 1.27
        for wire in wires:
            if not contains(wire, item.x, item.y):
                continue
            points.update({(wire.x1, wire.y1), (wire.x2, wire.y2)})
            if abs(wire.y1 - wire.y2) <= 0.05:
                low, high = sorted((wire.x1, wire.x2))
                steps = int((high - low) / grid)
                points.update((round(low + step * grid, 4), wire.y1) for step in range(steps + 1))
                points.add(((low + high) / 2.0, wire.y1))
            elif abs(wire.x1 - wire.x2) <= 0.05:
                low, high = sorted((wire.y1, wire.y2))
                steps = int((high - low) / grid)
                points.update((wire.x1, round(low + step * grid, 4)) for step in range(steps + 1))
                points.add((wire.x1, (low + high) / 2.0))
        return sorted(points)

    justifications = (
        "",
        "left",
        "right",
        "top",
        "bottom",
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    )
    initial_score = score()
    current_score = initial_score
    changes_by_index: dict[int, dict[str, Any]] = {}
    while True:
        best: tuple[float, int, float, float, str, visual_qa.LabelItem] | None = None
        for idx, entry in enumerate(entries):
            if contribution(idx) <= 0.0:
                continue
            item = entry["item"]
            for x, y in sample_points(item):
                for justify in justifications:
                    if (x, y) == (item.x, item.y) and justify == entry["justify"]:
                        continue
                    candidate = visual_qa.LabelItem(
                        item.text,
                        x,
                        y,
                        item.kind,
                        item.angle,
                        item.font_mm,
                        frozenset(justify.split()),
                    )
                    entry["item"] = candidate
                    candidate_score = score()
                    entry["item"] = item
                    if candidate_score >= current_score - 1e-9:
                        continue
                    proposal = (candidate_score, idx, x, y, justify, candidate)
                    if best is None or proposal[:5] < best[:5]:
                        best = proposal
        if best is None:
            break
        current_score, idx, x, y, justify, candidate = best
        entry = entries[idx]
        old_item = entry["item"]
        entry["item"] = candidate
        entry["justify"] = justify
        changes_by_index[idx] = {
            "kind": "label_slide",
            "ref": candidate.text,
            "from": [old_item.x, old_item.y],
            "to": [x, y],
            "justify": justify or "center",
        }

    replacements: list[tuple[int, int, str]] = []
    for idx in changes_by_index:
        entry = entries[idx]
        item = entry["item"]
        block = entry["block"]
        at_match = _AT3_RE.search(block)
        if at_match is None:
            continue
        new_at = f"(at {_fmt_mm(item.x)} {_fmt_mm(item.y)} {_fmt_mm(item.angle)})"
        new_block = block[: at_match.start()] + new_at + block[at_match.end() :]
        new_block = _set_label_justify(new_block, entry["justify"])
        replacements.append((entry["start"], entry["end"], new_block))

    changes = list(changes_by_index.values())
    if changes:
        changes.append(
            {
                "kind": "objective",
                "overlap_area_before_mm2": round(initial_score, 3),
                "overlap_area_after_mm2": round(current_score, 3),
            }
        )
    return _apply_span_replacements(content, replacements), changes


def _mutate_normalize_power_orientation(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Upright sideways power/ground symbols by rotating about their pin.

    A power symbol has a single pin. We rotate the placement to 0 degrees and
    translate the symbol so that pin returns to its original coordinate — the pin
    stays put, so the net is preserved while the graphic reads upright.
    """

    changes: list[dict[str, Any]] = []
    repls: list[tuple[int, int, str]] = []

    for start, end, block in _iter_blocks(content, _SYMBOL_HEAD_RE):
        lib_id = _symbol_lib_id(block)
        if not lib_id.lower().startswith("power:"):
            continue
        placement = _symbol_placement(block)
        if placement is None:
            continue
        x, y, angle = placement
        if angle % 180 == 0:
            continue  # already upright (0 or 180)

        try:
            library, symbol_name = _split_lib_id(lib_id)
        except ValueError:
            continue
        pins_before = get_pin_positions(library, symbol_name, x, y, angle)
        pins_upright = get_pin_positions(library, symbol_name, x, y, 0)
        if len(pins_before) != 1 or len(pins_upright) != 1:
            continue  # not a single-pin power symbol; leave it alone
        (pin_number, before_pt), (_, upright_pt) = (
            next(iter(pins_before.items())),
            next(iter(pins_upright.items())),
        )
        dx = before_pt[0] - upright_pt[0]
        dy = before_pt[1] - upright_pt[1]

        # Set the placement angle to 0, then translate the whole symbol by the
        # pin-restoring delta so the pin tip returns to its original coordinate.
        at_match = _AT3_RE.search(block)
        if at_match is None:
            continue
        upright_block = (
            block[: at_match.start()]
            + f"(at {_fmt_mm(x)} {_fmt_mm(y)} 0)"
            + block[at_match.end() :]
        )
        new_block = _shift_symbol_block(upright_block, dx, dy)
        repls.append((start, end, new_block))
        changes.append(
            {
                "kind": "power_symbol",
                "ref": _symbol_reference(block),
                "pin": pin_number,
                "from_angle": angle,
                "to_angle": 0,
            }
        )

    return _apply_span_replacements(content, repls), changes


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register schematic cosmetic auto-fix tools."""

    @mcp.tool()
    @headless_compatible
    def sch_align_to_grid(apply: bool = False) -> str:
        """Snap off-grid symbols, labels, wires, and junctions onto the drawing grid.

        Off-grid anchors are the top cause of ragged sheets and wires that visibly
        miss their pins. Symbols translate as a whole so pins stay aligned. Runs as
        a dry run by default (reports the moves and whether connectivity is
        preserved); pass apply=true to commit. Apply is refused if it would change
        any net.
        """
        return _run_cosmetic_fix("sch_align_to_grid", _mutate_align_to_grid, apply=apply)

    @mcp.tool()
    @headless_compatible
    def sch_straighten_wires(apply: bool = False) -> str:
        """Straighten almost-orthogonal wire segments onto the horizontal/vertical axis.

        Diagonal wires read as rushed; small off-axis deltas are snapped flat. Dry
        run by default; apply=true commits, and is refused if a straighten would
        change connectivity.
        """
        return _run_cosmetic_fix("sch_straighten_wires", _mutate_straighten_wires, apply=apply)

    @mcp.tool()
    @headless_compatible
    def sch_resolve_label_overlaps(apply: bool = False) -> str:
        """Flip the justify of overlapping labels so their text stops colliding.

        Connectivity-safe by construction: the label anchor never moves, only the
        rendered text direction. Dry run by default; apply=true commits.
        """
        return _run_cosmetic_fix(
            "sch_resolve_label_overlaps", _mutate_resolve_label_overlaps, apply=apply
        )

    @mcp.tool()
    @headless_compatible
    def sch_slide_overlapping_labels(apply: bool = False) -> str:
        """Slide trapped label anchors along their existing connected wire segments.

        Samples endpoints, grid points, and the midpoint together with legal text
        justifications, then accepts only strict reductions in rendered overlap.
        Dry run by default; apply=true commits only if the independent electrical
        connectivity signature remains identical.
        """
        return _run_cosmetic_fix(
            "sch_slide_overlapping_labels", _mutate_slide_overlapping_labels, apply=apply
        )

    @mcp.tool()
    @headless_compatible
    def sch_normalize_power_orientation(apply: bool = False) -> str:
        """Upright sideways power/ground symbols, rotating about the pin to keep the net.

        Rotates 90/270-degree power symbols to upright and translates them so the
        pin stays on its original coordinate. Dry run by default; apply=true
        commits, and is refused if the net would change.
        """
        return _run_cosmetic_fix(
            "sch_normalize_power_orientation", _mutate_normalize_power_orientation, apply=apply
        )

    @mcp.tool()
    @headless_compatible
    def sch_normalize_text_sizes(apply: bool = False) -> str:
        """Normalise outlier font sizes onto the sheet's dominant size.

        Connectivity-neutral typography cleanup over placed-symbol and label text
        only (never the library cache). Dry run by default; apply=true commits.
        """
        return _run_cosmetic_fix(
            "sch_normalize_text_sizes", _mutate_normalize_text_sizes, apply=apply
        )


__all__ = ["register"]
