"""Pure orchestration for non-destructive schematic symbol mutations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

type SymbolMatch = tuple[str, int, int, Mapping[str, Any]]
type UpdateSymbolProperty = Callable[[str, str, str], str]
type SetSymbolDnp = Callable[[str, bool, str | None], str]
type ReloadSchematic = Callable[[], str]
type SnapPoint = Callable[[float, float, bool], tuple[float, float]]
type SnapNotice = Callable[[tuple[float, ...], tuple[float, ...]], str]
type FindPlacedSymbolBlock = Callable[[str, str], SymbolMatch | None]
type ResolveSchematicFile = Callable[[str | None, str | None], Path]
type ShiftConnectedBundle = Callable[[str, str, float, float], tuple[str, int, int]]
type UpdateSymbolPropertyInFile = Callable[[Path, str, str, str], str]


class TransactionalWrite(Protocol):
    """Transaction boundary used by symbol move orchestration."""

    def __call__(
        self,
        mutator: Callable[[str], str],
        *,
        allow_node_loss: bool = False,
    ) -> str: ...


class TransactionalWriteToFile(Protocol):
    """Transaction boundary for an explicitly selected child schematic."""

    def __call__(
        self,
        path: Path,
        mutator: Callable[[str], str],
        *,
        allow_node_loss: bool = False,
    ) -> str: ...


class ShiftSymbolBlock(Protocol):
    """Callable contract for moving one placed-symbol block."""

    def __call__(
        self,
        block: str,
        *,
        dx_mm: float,
        dy_mm: float,
    ) -> str: ...


@dataclass(frozen=True)
class SchematicSymbolMutationService:
    """Compose symbol property and placement mutations from injected operations."""

    update_symbol_property: UpdateSymbolProperty
    set_symbol_dnp: SetSymbolDnp
    reload_schematic: ReloadSchematic
    snap_point: SnapPoint
    snap_notice: SnapNotice
    transactional_write: TransactionalWrite
    find_placed_symbol_block: FindPlacedSymbolBlock
    shift_symbol_block: ShiftSymbolBlock
    resolve_schematic_file: ResolveSchematicFile | None = None
    transactional_write_to_file: TransactionalWriteToFile | None = None
    shift_connected_bundle: ShiftConnectedBundle | None = None
    update_symbol_property_in_file: UpdateSymbolPropertyInFile | None = None

    def update_properties(
        self,
        reference: str,
        field: str,
        value: str,
        sheet: str | None = None,
        sheet_file: str | None = None,
    ) -> str:
        """Update one symbol property and reload the schematic."""
        if sheet or sheet_file:
            if self.resolve_schematic_file is None or self.update_symbol_property_in_file is None:
                raise ValueError(
                    "Child-sheet property updates are not configured for this backend."
                )
            target = self.resolve_schematic_file(sheet, sheet_file)
            result = self.update_symbol_property_in_file(target, reference, field, value)
            return f"{result}\nChild schematic updated; reload it in KiCad if open."
        result = self.update_symbol_property(reference, field, value)
        return f"{result}\n{self.reload_schematic()}"

    def set_dnp(self, reference: str, enabled: bool, reason: str | None) -> str:
        """Update native DNP state and reload the schematic."""
        result = self.set_symbol_dnp(reference, enabled, reason)
        return f"{result}\n{self.reload_schematic()}"

    def move_symbol(
        self,
        reference: str,
        x_mm: float,
        y_mm: float,
        snap_to_grid: bool,
        sheet: str | None = None,
        sheet_file: str | None = None,
        with_terminals: bool = False,
    ) -> str:
        """Move one symbol, optionally carrying its isolated terminal stubs."""
        target_x, target_y = self.snap_point(x_mm, y_mm, snap_to_grid)
        snap_note = self.snap_notice((x_mm, y_mm), (target_x, target_y))
        target_path: Path | None = None
        targeted_child = bool(sheet or sheet_file)
        if targeted_child:
            if self.resolve_schematic_file is None or self.transactional_write_to_file is None:
                raise ValueError("Child-sheet symbol movement is not configured for this backend.")
            target_path = self.resolve_schematic_file(sheet, sheet_file)
        moved_wires = 0
        moved_terminals = 0

        def mutator(current: str) -> str:
            nonlocal moved_wires, moved_terminals
            match = self.find_placed_symbol_block(current, reference)
            if match is None:
                raise ValueError(f"Reference '{reference}' was not found in the schematic.")
            block, start, end, parsed = match
            dx_mm = target_x - float(parsed["x"])
            dy_mm = target_y - float(parsed["y"])
            if with_terminals:
                if self.shift_connected_bundle is None:
                    raise ValueError(
                        "Connected-terminal movement is not configured for this backend."
                    )
                updated, moved_wires, moved_terminals = self.shift_connected_bundle(
                    current,
                    reference,
                    dx_mm,
                    dy_mm,
                )
                return updated
            shifted = self.shift_symbol_block(block, dx_mm=dx_mm, dy_mm=dy_mm)
            return current[:start] + shifted + current[end:]

        try:
            if (
                targeted_child
                and target_path is not None
                and self.transactional_write_to_file is not None
            ):
                self.transactional_write_to_file(target_path, mutator)
            else:
                self.transactional_write(mutator)
        except ValueError as exc:
            return str(exc)

        result = (
            "Child schematic updated; reload it in KiCad if open."
            if targeted_child
            else self.reload_schematic()
        )
        lines = [
            result,
            f"Moved symbol '{reference}' to ({target_x:.2f}, {target_y:.2f}) mm.",
        ]
        if target_path is not None:
            lines.append(f"Target schematic: {target_path}")
        if with_terminals:
            lines.append(
                f"Moved {moved_wires} attached stub wire(s) and {moved_terminals} terminal(s)."
            )
        if snap_note:
            lines.append(snap_note)
        return "\n".join(lines)
