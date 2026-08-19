# Contract: the wiring manifest and typed ports

The interface every faculty exposes to every other. Consumed by: the kernel (routing), the gates (p6, region charge, dormancy), the neuter harness, and the cockpit.

## Port declaration (per faculty)
```
Port(name, region, shape, dtype, precision_bits, unit, encoding, kind)
  IN  ports MUST declare join_policy   ∈ {sum, max, concat, typed-merge, arbitrate}
  OUT ports MUST declare delivery_policy (all-or-none | graded(association_source))
  REF ports are flagged, enumerated, and audited — the governed escape hatch
```

## Manifest (lake-committed artifact, NIR-HDF5 serialized)
```
edges: [(src_faculty, src_port, dst_faculty, dst_port)]     # 4-tuples, port-to-port
checks at CI (same commit as any manifest change):
  validate_structure()   # endpoints exist, no duplicate edges
  infer_types()          # shapes propagate from declared inputs
  check_types()          # producer/consumer mismatch = RED
  region_check()         # port.region == brain_atlas.region_of(declaring_module)
  join_check()           # >1 edge into an IN port without a declared policy = RED
```

## Charge measurement (the P3-honest half — what makes a port real)
- Per-port neuter: pinned-constant payload substitution (dict-subclass that records reads); consumer output MUST change live vs neutered.
- Per-consumption cleanup-margin record: `{winner, margin_to_runner_up}` — a consumer whose margin never moves is not consuming.
- Replay A/B: same segment replayed with/without one port neutered; divergence attributes the port.

## Region charge gate (sourced here)
- green requires: region reachable (dormancy-gate path) AND ≥1 charged port edge attributed to the region.
- Registration alone NEVER greens. Exempt rows: `{region, reason, owner, date, expiry}`; exemption count is a ratchet (only tightens).

## Reds this contract must be able to fire (proven at landing, each with a positive control)
1. Undeclared-join red (two producers, one summing input, no policy).
2. Region-mismatch red (port declares a region its module does not belong to).
3. Neuter-insensitivity finding (a declared consumer whose output never changes).
4. Manifest/`p6_gate` divergence red (gate reads a manifest that does not match the committed copy).
