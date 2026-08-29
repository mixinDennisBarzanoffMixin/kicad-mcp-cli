from __future__ import annotations

from kicad_mcp.deep_inspection import (
    _board_net_names,
    _board_semantic_state,
    _net_identity,
    _track_records,
    _via_records,
)

KICAD_10_BOARD = r"""(kicad_pcb
  (version 20260206)
  (generator "pcbnew")
  (gr_rect (start 0 0) (end 20 10)
    (stroke (width 0.05) (type solid)) (fill no) (layer "Edge.Cuts"))
  (footprint "R_0603"
    (layer "F.Cu")
    (at 5 5)
    (property "Reference" "R1")
    (property "Value" "10k")
    (fp_rect (start -1 -0.5) (end 1 0.5)
      (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.5 0) (size 0.6 0.6)
      (layers "F.Cu" "F.Mask") (net "GND"))
    (pad "2" smd rect (at 0.5 0) (size 0.6 0.6)
      (layers "F.Cu" "F.Mask") (net "SIGNAL")))
  (segment (start 4.5 5) (end 2 5) (width 0.25)
    (layer "F.Cu") (net "GND"))
  (via (at 2 5) (size 0.6) (drill 0.3)
    (layers "F.Cu" "B.Cu") (net "GND"))
  (zone (net "GND") (net_name "GND") (layer "B.Cu"))
)"""


def test_kicad_10_name_based_net_identity() -> None:
    assert _net_identity('(net "GND")') == (0, "GND")
    assert _net_identity('(net 7 "GND")') == (7, "GND")
    assert _net_identity("(net 7)") == (7, "")


def test_kicad_10_board_net_catalog_includes_pad_and_routing_nets() -> None:
    assert _board_net_names(KICAD_10_BOARD) == ["GND", "SIGNAL"]

    tracks = _track_records(KICAD_10_BOARD)
    vias = _via_records(KICAD_10_BOARD)
    assert tracks[0]["net"] == "GND"
    assert tracks[0]["net_code"] == 0
    assert vias[0]["net"] == "GND"
    assert vias[0]["net_code"] == 0

    state = _board_semantic_state(KICAD_10_BOARD)
    assert state["nets"] == ["GND", "SIGNAL"]
    assert state["tracks"][0]["net"] == "GND"
