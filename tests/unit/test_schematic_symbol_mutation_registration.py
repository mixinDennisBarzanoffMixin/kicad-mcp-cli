from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from kicad_mcp.tools.metadata import get_tool_metadata
from kicad_mcp.tools.schematic_symbol_mutation import (
    SchematicSymbolMutationDependencies,
    register,
)


class FakeSymbolMutationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def update_properties(self, reference: str, field: str, value: str) -> str:
        self.calls.append(("update_properties", (reference, field, value)))
        return "updated"

    def set_dnp(self, reference: str, enabled: bool, reason: str | None) -> str:
        self.calls.append(("set_dnp", (reference, enabled, reason)))
        return "dnp"

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
        self.calls.append(
            (
                "move_symbol",
                (
                    reference,
                    x_mm,
                    y_mm,
                    snap_to_grid,
                    sheet,
                    sheet_file,
                    with_terminals,
                ),
            )
        )
        return "moved"


def _registered() -> tuple[FastMCP, FakeSymbolMutationService]:
    server = FastMCP("schematic-symbol-mutation-test")
    service = FakeSymbolMutationService()
    register(server, SchematicSymbolMutationDependencies(service=service))
    return server, service


def test_registration_preserves_names_descriptions_and_schemas() -> None:
    server, _service = _registered()
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

    assert set(tools) == {
        "sch_update_properties",
        "sch_set_dnp",
        "sch_modify_property",
        "sch_move_symbol",
    }
    assert tools["sch_update_properties"].description == ("Update a property on a placed symbol.")
    assert tools["sch_set_dnp"].description == (
        "Set KiCad's native Do Not Populate flag on a placed symbol.\n\n"
        "When ``reason`` is given it is stored in the ``DNP Reason`` property so\n"
        "``sch_get_population_status`` and variant BOMs can report why the part\n"
        "is unpopulated.\n"
    )
    assert tools["sch_modify_property"].description == (
        "Modify a schematic symbol property by reference."
    )
    assert tools["sch_move_symbol"].description == (
        "Move a symbol, optionally carrying its isolated label/power stubs.\n\n"
        "Set ``with_terminals=True`` after ``sch_add_pin_labels``. The move is\n"
        "refused if an attached wire ends on another component or participates\n"
        "in a larger wire graph, preventing silent disconnection.\n"
    )
    assert tools["sch_update_properties"].parameters["required"] == [
        "reference",
        "field",
        "value",
    ]
    assert tools["sch_set_dnp"].parameters["required"] == ["reference"]
    assert tools["sch_set_dnp"].parameters["properties"]["enabled"]["default"] is True
    assert tools["sch_set_dnp"].parameters["properties"]["reason"]["default"] is None
    assert tools["sch_move_symbol"].parameters["required"] == [
        "reference",
        "x_mm",
        "y_mm",
    ]
    assert tools["sch_move_symbol"].parameters["properties"]["snap_to_grid"]["default"] is True


def test_registration_preserves_headless_metadata() -> None:
    _server, _service = _registered()

    for name in (
        "sch_update_properties",
        "sch_set_dnp",
        "sch_modify_property",
        "sch_move_symbol",
    ):
        metadata = get_tool_metadata(name)
        assert metadata is not None
        assert metadata.headless_compatible is True


def test_registration_delegates_and_preserves_modify_alias() -> None:
    server, service = _registered()
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

    assert tools["sch_update_properties"].fn("R1", "Value", "10k") == "updated"
    assert tools["sch_set_dnp"].fn("R2", False, "variant") == "dnp"
    assert tools["sch_modify_property"].fn("R3", "MPN", "ABC") == "updated"
    assert tools["sch_move_symbol"].fn("U1", 10.0, 20.0, False) == "moved"
    assert service.calls == [
        ("update_properties", ("R1", "Value", "10k")),
        ("set_dnp", ("R2", False, "variant")),
        ("update_properties", ("R3", "MPN", "ABC")),
        ("move_symbol", ("U1", 10.0, 20.0, False, None, None, False)),
    ]


def test_registration_preserves_move_validation() -> None:
    server, _service = _registered()
    tool = {tool.name: tool for tool in server._tool_manager.list_tools()}["sch_move_symbol"]

    with pytest.raises(ValidationError):
        tool.fn("", 10.0, 20.0, True)
