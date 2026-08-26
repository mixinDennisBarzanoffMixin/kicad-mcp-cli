# Shell-first KiCad interface

`kicadq` exposes the KiCad MCP backend as a normal Unix command. Standard output
is reserved for data, errors go to standard error, and non-success states use
stable exit codes (`0` success, `1` no match/tool failure, `2` usage/runtime
error, `3` rejected/refused verification, `4` blocked planning).

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
board geometry. When KiCad IPC is live, it also compares the open PCB semantically
with the saved board and labels the authority used by every section.

```bash
kicadq -C ./board inspect | jq '.schematic.counts, .board.counts'
kicadq -C ./board inspect --net USB --format jsonl
kicadq -C ./board map --zoom 0              # sheets/subsystems
kicadq -C ./board map --zoom 1 --sheet LTE  # components
kicadq -C ./board map --zoom 2 --net SPI    # pin-level nets
kicadq -C ./board map --zoom 3 --width 120  # PCB geometry
```

## Authority and proof

Do not guess which representation is current. `backend` reports the exact
authority for reads, writes, checks, renders, and exports. KiCad 10 uses native
IPC for the open PCB, `kicad-cli` for netlist/ERC/DRC/render/export, and guarded
file transactions only for schematic edits that the KiCad 10 IPC API does not
expose.

```bash
kicadq -C ./board backend | jq '{status,authorities,live_ipc,policy}'
kicadq -C ./board prove --sheet Power --format jsonl |
  jq -c 'select(.section == "pin") | [.reference,.pin,.net,.peers]'
kicadq -C ./board verify --sheet Power --artifacts build/power-proof |
  jq '{status,checks,artifacts}'
```

`prove` expands every selected pin into `pin → net → peer pins`, separates
intentional no-connects from accidental dangling pins, and compares PCB pad nets
when a synchronized board exists. `verify` bundles source-integrity checks,
connectivity proof, native KiCad ERC JSON, and hop-over SVG renders.

## Arrange circuit specs

`arrange-spec` is a read-only JSON transformer for circuit specs. It ranks the
symbols as a railway graph, evaluates bounded deterministic candidates, fills
only missing `x_mm`/`y_mm` fields, preserves every explicit coordinate, and
forces `auto_layout=false`. Its `spec` format contains only the original build
payload plus those coordinate changes; planning evidence stays in `report` and
`json` formats.

```bash
kicadq arrange-spec power.json --format report
kicadq arrange-spec power.json --format json |
  jq '{selected:.layout.selected_candidate, costs:.layout.ranked_candidates, rails:.layout.graph.rail_nets}'
jq '.symbols |= map(del(.x_mm,.y_mm))' power.json |
  kicadq arrange-spec - --format spec > power-arranged.json
```

Before building, pass the generated fields through the backend analyzer:

```bash
kicadq arrange-spec power.json --format spec |
  jq '{symbols,nets,wires,labels,power_symbols,snap_to_grid,auto_layout}' |
  kicadq -C ./board call sch_analyze_net_compilation --args - --format raw
```

## Read-only schematic railway planning

`plan-rewire` resolves exact pin tips from the selected sheet's cached library
symbols, classifies local connections separately from shared and cross-sheet
rails, then proposes orthogonal wires and explicit label operations. Repeated
`--ref` options define one component cluster; omit every `--ref` to inspect the
whole selected sheet. The command has no apply mode and does not modify KiCad
files.

```bash
kicadq -C ./board plan-rewire --sheet LTE --ref C15 --ref R11
kicadq -C ./board plan-rewire --sheet LTE --format json |
  jq '{status, operations, refused: [.nets[] | select(.status == "refused")]}'
```

The JSON includes source and symbol-position hashes plus the expected
name-inclusive connectivity fingerprint. `apply-rewire` consumes only
explicitly selected safe nets, plans again, executes the physical operations in
a disposable project clone, and promotes exactly one sheet only after the
connectivity and symbol-position fingerprints, source integrity, and ERC pass.

```bash
kicadq -C ./board --profile agent_full --mode write apply-rewire \
  --sheet LTE --net LTE_RT --net LTE_CT \
  --artifacts output/verified/lte-railways --yes --format json |
  jq '{status,nets:.selected_nets,promoted:.transaction.promoted}'
```

## Atomic schematic edits

Use `edit` instead of direct write calls for routine schematic authoring. It
copies the project to a temporary staging directory, invokes one `sch_*` tool,
generates a unified diff and before/after SVGs, rejects structural/connectivity
failures or new ERC errors, then atomically promotes exactly one verified
schematic file. A failed edit leaves the source project byte-for-byte unchanged.

```bash
kicadq -C ./board --mode write edit sch_modify_property \
  --sheet Power \
  --set reference=R7 \
  --set field=Value \
  --set value=100k \
  --set sheet_file=02_Power.kicad_sch \
  --yes | jq '{status,promoted,changed_files,diff,artifacts}'

kicadq -C ./board --mode write edit sch_build_circuit \
  --sheet Power --args @power-circuit.json --yes | jq '.status'
```

By default, evidence is written below
`build/kicadq-transactions/<tool>-<id>/`; pass `--artifacts DIR` to choose a
stable destination.

Route planning is a dry run by default. It first tries direct Manhattan geometry,
then uses a deterministic grid/A* search around footprint and existing-track
obstacles. Critical power, ground, RF, clock, and USB differential nets are
refused unless explicitly overridden. Applying a plan also requires write mode
and a second confirmation flag; the live KiCad backend performs the actual track
creation. Planning is blocked when live IPC and the saved board differ. Applied
routes are grouped in one native IPC commit, serialized without saving, checked
with KiCad DRC, and pushed or dropped as a unit.

```bash
kicadq -C ./board route GPIO17 | jq '.segments'
kicadq -C ./board --mode write route GPIO17 --apply --yes
```

Placement planning is also dry-run-first. It derives weighted component
connectivity from the schematic, respects `Edge.Cuts`, lets mechanical anchors
stay fixed, and accepts absolute rectangular keepouts. The result is ordinary
JSON suitable for `jq`, review, and diffs.

When the design-intent JSON contains `placement_floorplan`, `--spec` also loads
its margin, grid, iteration/seed controls, anchors, keepouts, fixed references,
and hierarchical-sheet cluster regions. Explicit command-line placement options
override those stored defaults, so a reviewed floorplan is reproducible with one
short command while experiments remain possible.

```bash
kicadq -C ./board place --fix J1 --fix J2 --keepout 0,0,25,12 | jq '.placements'
kicadq -C ./board place --spec .kicad-mcp/project_spec.json | jq '.quality_gate'
kicadq -C ./board --mode write place --fix J1 --apply --yes
```

Applied placement moves use the same native commit/DRC/drop transaction. Evidence
includes the before/staged board sources, a unified diff, and before/staged DRC
JSON summaries under `build/kicadq-transactions/` unless `--artifacts` overrides
the destination. The DRC gate rejects every new physical finding and any increase
in the unrouted-item count; a lower total violation count cannot hide a newly
introduced problem.

## Pad-aware power-loop inspection and placement

`power-loops` checks declared IC/capacitor groups using the actual power-pad and
ground-pad coordinates. It reports the forward rail distance, ground return
distance, loop estimate, and the limiting pad pair instead of using footprint
centres as an electrical proxy.

```bash
kicadq -C ./board power-loops --ref U1 --format json |
  jq '.groups[] | {ic_ref,status,members}'
kicadq -C ./board power-loops --format jsonl |
  jq -c 'select(.section == "member" and .status != "pass")'
```

`place-power-loops` searches grid-aligned capacitor root transforms around each
host. It rotates pad geometry rigidly, preserves concave KiCad courtyard loops,
checks keepouts and other component courtyards, and scores both the supply path
and ground return. Existing capacitor orientation is preferred unless rotation
materially improves the electrical result. Capacitors declared for the same host
are packed with a bounded deterministic cluster search, so a locally attractive
first capacitor cannot consume the only legal position for a later one. A failed
search reports the blocking footprint references and hit counts.

```bash
kicadq -C ./board place-power-loops --ref U1 | jq '.placements,.after.groups'
kicadq -C ./board --mode write place-power-loops --ref U1 \
  --apply --yes --artifacts output/u1-placement | jq '.transaction'
```

An apply request never writes optimistically. It first constructs an offline
candidate from the saved board, verifies every requested root transform and all
rigid footprint children, and runs exact `kicad-cli` DRC. With synchronized native
KiCad authority, the same transforms are held in an unpushed IPC commit, compared
with the verified candidate, pushed, read back, verified again, and only then
saved. A mismatch is reverted. If native IPC is unavailable, the verified
candidate and diff are retained as evidence with `candidate_verified: true`, but
the live/saved board is not modified and the command exits blocked (`4`).
Candidate artifacts include `before-drc.json`, `staged-drc.json`, and
`drc-regressions.json`; finding identity is based on stable KiCad UUID relations,
not coordinates that legitimately change during a footprint move.

Both planners are proposals, not substitutes for KiCad DRC or electrical,
thermal, RF, and mechanical review.

## Compact MCP facade

`kicad-mcp-cli` publishes only three MCP tools:

- `kicad_catalog` searches the operation catalog.
- `kicad_run` executes a discovered operation by name.
- `kicad_grep` searches textual KiCad project files.

The full backend remains available behind `kicad_run`, but it no longer expands
into hundreds of first-class tools in an agent's context.
