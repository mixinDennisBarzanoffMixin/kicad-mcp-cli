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

## Compact MCP facade

`kicad-mcp-cli` publishes only three MCP tools:

- `kicad_catalog` searches the operation catalog.
- `kicad_run` executes a discovered operation by name.
- `kicad_grep` searches textual KiCad project files.

The full backend remains available behind `kicad_run`, but it no longer expands
into hundreds of first-class tools in an agent's context.
