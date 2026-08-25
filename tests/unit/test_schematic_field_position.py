"""Unit coverage for persisted schematic field geometry."""

from kicad_mcp.tools.schematic import _set_property_position


def test_set_property_position_persists_justification() -> None:
    symbol = """(symbol
\t\t(property "Reference" "U1"
\t\t\t(at 10 20 0)
\t\t\t(effects (font (size 1.27 1.27)))
\t\t)
)"""

    updated = _set_property_position(
        symbol,
        "Reference",
        30.0,
        40.0,
        0.0,
        frozenset({"right"}),
    )

    assert "(at 30 40 0)" in updated
    assert "(justify right)" in updated


def test_set_property_position_removes_stale_justification() -> None:
    symbol = """(symbol
\t\t(property "Value" "IC"
\t\t\t(at 10 20 0)
\t\t\t(effects (font (size 1.27 1.27)) (justify left))
\t\t)
)"""

    updated = _set_property_position(symbol, "Value", 30.0, 40.0, 0.0)

    assert "(justify" not in updated
