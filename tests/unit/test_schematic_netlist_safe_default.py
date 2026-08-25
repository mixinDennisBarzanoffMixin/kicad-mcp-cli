"""Regression tests for collision-safe-by-default net compilation (issue #198).

`sch_build_circuit` used to draw routed Manhattan wires between pins whenever
`auto_layout` was False, which can cross unrelated pins/labels and merge by
geometry into silent shorts. The collision-safe terminal-stub planner must now be
the default for *any* supplied netlist; routed wires must require an explicit
`unsafe_routed_wires=True` opt-in.
"""

from __future__ import annotations

import pytest

from kicad_mcp.models.schematic import AddSymbolInput
from kicad_mcp.tools import schematic as sch

_STATS = {
    "resolved_endpoints": 0,
    "unresolved_endpoints": 0,
    "pin_alias_resolutions": 0,
    "symbol_center_resolutions": 0,
}

_SYMBOLS = [
    {
        "library": "Device",
        "symbol_name": "R",
        "reference": "R1",
        "value": "10k",
        "x_mm": 50.8,
        "y_mm": 50.8,
    }
]
_NETS = [{"name": "SIG", "endpoints": ["R1.1", "R1.2"]}]


@pytest.fixture
def planner_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which net planner the builder dispatches to, with no KiCad needed."""
    calls: list[str] = []

    def spy_terminals(symbols, powers, labels, nets, snap):  # type: ignore[no-untyped-def]
        calls.append("terminals")
        return ([], [], [], [], dict(_STATS))

    def spy_wires(symbols, powers, labels, nets, snap):  # type: ignore[no-untyped-def]
        calls.append("wires")
        return ([], [], dict(_STATS))

    monkeypatch.setattr(sch, "_plan_netlist_pin_terminals", spy_terminals)
    monkeypatch.setattr(sch, "_plan_netlist_wires", spy_wires)
    # Isolate from the installed KiCad symbol libraries.
    monkeypatch.setattr(sch, "get_symbol_available_units", lambda *_a, **_k: set())
    return calls


def test_nets_default_to_collision_safe_terminals(planner_spies: list[str]) -> None:
    sch._prepare_build_circuit_inputs(symbols=_SYMBOLS, nets=_NETS)
    assert planner_spies == ["terminals"]


def test_nets_default_to_terminals_even_without_auto_layout(planner_spies: list[str]) -> None:
    # The previous footgun: auto_layout=False used to route Manhattan wires.
    sch._prepare_build_circuit_inputs(symbols=_SYMBOLS, nets=_NETS, auto_layout=False)
    assert planner_spies == ["terminals"]


def test_routed_wires_require_explicit_opt_in(planner_spies: list[str]) -> None:
    sch._prepare_build_circuit_inputs(symbols=_SYMBOLS, nets=_NETS, unsafe_routed_wires=True)
    assert planner_spies == ["wires"]


def test_auto_layout_with_nets_still_uses_terminals(planner_spies: list[str]) -> None:
    sch._prepare_build_circuit_inputs(symbols=_SYMBOLS, nets=_NETS, auto_layout=True)
    assert planner_spies == ["terminals"]


def test_build_circuit_default_keeps_routed_wires_opt_in() -> None:
    # Keyword-only defaults live in __kwdefaults__; routed wires must default off.
    kwdefaults = sch._prepare_build_circuit_inputs.__kwdefaults__ or {}
    assert kwdefaults.get("unsafe_routed_wires") is False


def test_facing_pin_rows_fall_back_to_labels_directly_on_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal stub must never run through a different symbol's pin.

    This reproduces the dense USB-C-to-ESD-array geometry from Flux: opposing
    pin rows are closer than the normal 5.08 mm terminal stub.  The safe form is
    a named label directly on each pin, with no copper segment between them.
    """

    symbols = [
        AddSymbolInput(
            library="Test",
            symbol_name="Facing",
            reference="U1",
            value="left",
            x_mm=50.0,
            y_mm=50.0,
        ),
        AddSymbolInput(
            library="Test",
            symbol_name="Facing",
            reference="U2",
            value="right",
            x_mm=60.0,
            y_mm=50.0,
        ),
    ]

    def pin_positions(
        _library: str,
        _symbol: str,
        x_mm: float,
        _y_mm: float,
        _rotation: int,
        _unit: int,
    ) -> dict[str, tuple[float, float]]:
        return {"1": (55.0, 50.0)} if x_mm < 55.0 else {"1": (57.54, 50.0)}

    monkeypatch.setattr(sch, "get_pin_positions", pin_positions)
    monkeypatch.setattr(sch, "get_pin_alias_positions", pin_positions)
    monkeypatch.setattr(
        sch,
        "get_pin_metadata",
        lambda *_args: {"1": {"name": "IO", "etype": "passive"}},
    )

    wires, powers, labels, unresolved, stats = sch._plan_netlist_pin_terminals(
        symbols,
        [],
        [],
        [
            {"name": "LEFT_NET", "endpoints": ["U1.1"], "scope": "local"},
            {"name": "RIGHT_NET", "endpoints": ["U2.1"], "scope": "local"},
        ],
        False,
    )

    assert wires == []
    assert powers == []
    assert unresolved == []
    assert stats["resolved_endpoints"] == 2
    assert [(label["name"], label["x_mm"], label["y_mm"]) for label in labels] == [
        ("LEFT_NET", 55.0, 50.0),
        ("RIGHT_NET", 57.54, 50.0),
    ]


def test_intentional_no_connects_resolve_exact_pins_and_reject_wired_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = AddSymbolInput(
        library="Test",
        symbol_name="Part",
        reference="U1",
        value="part",
        x_mm=50.0,
        y_mm=50.0,
    )
    monkeypatch.setattr(sch, "get_pin_positions", lambda *_args: {"9": (55.0, 50.0)})
    monkeypatch.setattr(
        sch,
        "get_pin_alias_positions",
        lambda *_args: {"9": (55.0, 50.0), "NC": (55.0, 50.0)},
    )

    assert sch._resolve_intentional_no_connects([symbol], [], ["U1.NC"], True) == [
        (55.0, 50.0)
    ]

    with pytest.raises(ValueError, match="already assigned to net 'SIGNAL'"):
        sch._resolve_intentional_no_connects(
            [symbol],
            [{"name": "SIGNAL", "endpoints": ["U1.9"]}],
            ["U1.NC"],
            True,
        )


def test_net_compilation_report_announces_routing_mode() -> None:
    safe = sch._render_net_compilation_report(
        symbols=[],
        powers=[],
        labels=[],
        explicit_wires=0,
        nets=[{"name": "SIG"}],
        generated_wires=[],
        unresolved_nets=[],
        resolution_stats=dict(_STATS),
        auto_layout=False,
        terminalized=True,
    )
    assert "terminal labels (collision-safe)" in safe

    unsafe = sch._render_net_compilation_report(
        symbols=[],
        powers=[],
        labels=[],
        explicit_wires=0,
        nets=[{"name": "SIG"}],
        generated_wires=[],
        unresolved_nets=[],
        resolution_stats=dict(_STATS),
        auto_layout=False,
        terminalized=False,
    )
    assert "routed wires (unsafe)" in unsafe
