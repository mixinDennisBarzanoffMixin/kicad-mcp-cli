# Shell-first KiCad interface

`kicadq` exposes the KiCad MCP backend as a normal Unix command. Standard output
is reserved for data, errors go to standard error, and non-success states use
stable exit codes (`0` success, `1` no match/tool failure, `2` usage/runtime
error).

## Discover and inspect

```bash
kicadq tools routing --format jsonl | jq -r '.name'
kicadq tools --category pcb_read --format tsv | column -t -s $'\t'
kicadq schema pcb_get_nets | jq '.input_schema'
```

## Invoke operations

```bash
kicadq -C ./board call kicad_get_project_info --format raw | jq
kicadq -C ./board call pcb_get_tracks --set filter_layer=F_Cu --format raw | jq
printf '{"query":"USB"}' | kicadq -C ./board call lib_search_symbols --args - --format raw
```

`--args` accepts an inline JSON object, `-` for stdin, or `@file.json`. Repeated
`--set KEY=VALUE` flags merge into it; values are decoded as JSON when possible.

## Search project source

```bash
kicadq -C ./board files | jq -r '.path'
kicadq -C ./board grep 'USB|VBUS' | jq -r '[.path,.line,.text] | @tsv'
kicadq -C ./board grep GND --format lines | rg regulator
```

## Deep project inspection and maps

The deep snapshot uses KiCad's XML netlist export plus the board file, so it sees
hierarchical sheets, component/pin connectivity, placed pads, tracks, vias, and
board geometry without requiring the GUI to be open.

```bash
kicadq -C ./board inspect | jq '.schematic.counts, .board.counts'
kicadq -C ./board inspect --net USB --format jsonl
kicadq -C ./board map --zoom 0              # sheets/subsystems
kicadq -C ./board map --zoom 1 --sheet LTE  # components
kicadq -C ./board map --zoom 2 --net SPI    # pin-level nets
kicadq -C ./board map --zoom 3 --width 120  # PCB geometry
```

Route planning is a dry run by default. It first tries direct Manhattan geometry,
then uses a deterministic grid/A* search around footprint and existing-track
obstacles. Critical power, ground, RF, clock, and USB differential nets are
refused unless explicitly overridden. Applying a plan also requires write mode
and a second confirmation flag; the live KiCad backend performs the actual track
creation.

```bash
kicadq -C ./board route GPIO17 | jq '.segments'
kicadq -C ./board --mode write route GPIO17 --apply --yes
```

Placement planning is also dry-run-first. It derives weighted component
connectivity from the schematic, respects `Edge.Cuts`, lets mechanical anchors
stay fixed, and accepts absolute rectangular keepouts. The result is ordinary
JSON suitable for `jq`, review, and diffs.

```bash
kicadq -C ./board place --fix J1 --fix J2 --keepout 0,0,25,12 | jq '.placements'
kicadq -C ./board --mode write place --fix J1 --apply --yes
```

Both planners are proposals, not substitutes for KiCad DRC or electrical,
thermal, RF, and mechanical review.

## Compact MCP facade

`kicad-mcp-cli` publishes only three MCP tools:

- `kicad_catalog` searches the operation catalog.
- `kicad_run` executes a discovered operation by name.
- `kicad_grep` searches textual KiCad project files.

The full backend remains available behind `kicad_run`, but it no longer expands
into hundreds of first-class tools in an agent's context.
