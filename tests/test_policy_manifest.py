"""Tests for gates/policy_manifest.py — parsers, reconciliation, manifest.

Every negative here is CONSTRUCTED and demonstrated (SI 5): the reconciliation
red, the double-count red, the empty-walk red and the junk-token control each
prove the corresponding assert can fire, so the live positives are not vacuous.
"""

import re

import pytest

from snn_data_lake.gates import policy_manifest as pm

REAL = pm.POLICIES.read_text(encoding="utf-8")


# ---------------------------------------------------------------- T007: parsers


def test_the_identity_holds_on_the_real_tree():
    # Assert the IDENTITY, not frozen numbers: parents + table_only == table rows.
    policies = pm.parse(REAL)
    parents = [p for p in policies if not p.subclause_of and not p.table_only]
    table_only = [p for p in policies if p.table_only]
    top = len(parents) + len(table_only)
    assert len(table_only) == len(pm.TABLE_ONLY_IDS)
    # parse() already raised if the identity failed; re-state it from the output:
    n_rows_claimed = len(parents) + len(pm.TABLE_ONLY_IDS)
    assert top == n_rows_claimed
    # And the dated observation (9 + 1 == 10 at HEAD) as a sanity floor, not a pin:
    assert len(parents) >= 9
    assert "e-prop" in {p.id for p in table_only}


def test_a_missing_table_row_is_a_reconciliation_red():
    # Drop P6's enforcement row: parents stays 9, rows drops to 9 -> 9 + 1 != 9.
    broken = "\n".join(line for line in REAL.splitlines() if not line.startswith("| **P6"))
    assert broken != REAL, "fixture did not remove the P6 row"
    with pytest.raises(pm.DerivationError, match=r"parents=\d+ table_only=\d+ table_rows=\d+"):
        pm.parse(broken)


def test_a_table_only_id_gaining_a_heading_is_a_red_too(monkeypatch):
    # The double-count direction: a TABLE_ONLY id that also has a parent heading
    # must refuse, not count twice. PARENT_RE only matches P<digit> so "e-prop"
    # cannot be given a heading textually; drive the guard by pointing the
    # table-only set at P8, which has BOTH a heading and a table row. Identity
    # arithmetic still balances (9 + 1 == 10), so ONLY the double-count guard
    # can catch this shape.
    monkeypatch.setattr(pm, "TABLE_ONLY_IDS", frozenset({"P8"}))
    with pytest.raises(pm.DerivationError, match="double count"):
        pm.parse(REAL)


def test_an_empty_walk_refuses_loudly():
    with pytest.raises(pm.DerivationError, match="0 policies parsed"):
        pm.parse("# nothing here\n\njust prose\n")


def test_junk_tokens_are_citations_not_commands():
    # Accept-filter negative control: prose/citation tokens yield ZERO commands.
    junk = "`dormancy_gate.py:71` and `main()` and `.backward()` and `mlp_q` and `python -m`"
    assert pm.extract_commands(junk) == []
    # Positive control in the same test: a real token IS extracted.
    good = pm.extract_commands("`tools/p6_gate.py --assert` beside `main()`")
    assert [c.source_token for c in good] == ["tools/p6_gate.py --assert"]
    assert good[0].argv[1:] == ["tools/p6_gate.py", "--assert"]


def test_governed_tokens_map_to_module_invocations():
    (cmd,) = pm.extract_commands("`gates/dormancy_gate.py`")
    assert cmd.argv[1:] == ["-m", "snn_data_lake.gates.dormancy_gate"]


def test_a_gate_token_is_the_registrys_call_never_a_package_prefix():
    """G8 F1: `<pkg>/<mod>.py` is a gate command where the live registry places <mod> in <pkg>,
    with its CI standing read from the runner's registry-resolved names; the same module under a
    package it does not live in is a citation, and `tools/` scripts are unchanged."""
    from snn_data_lake import package_of

    pkg = package_of("dormancy_gate")
    (cmd,) = pm.extract_commands(f"`{pkg}/dormancy_gate.py --assert`")
    assert cmd.argv[1:] == ["-m", f"snn_data_lake.{pkg}.dormancy_gate", "--assert"]
    for elsewhere in ("substrate", "gates", "cortex"):
        if elsewhere != pkg:
            assert pm.extract_commands(f"`{elsewhere}/dormancy_gate.py`") == []
    (tool,) = pm.extract_commands("`tools/p6_gate.py`")
    assert tool.argv[1:] == ["tools/p6_gate.py"]
    (checked,) = pm.cross_check([pm.Policy("PX", "t", "absolute", [cmd])])[0].commands
    assert checked.in_selfchecks and checked.in_ci


def test_the_unclaimed_census_keeps_its_pre_g8_surface(monkeypatch, tmp_path):
    """G8 F1 at M1: the census still covers every gate-named module the runner registers from the
    pre-G8 `substrate` label, wherever it now lives: strip a moved gate's name from a copy of
    POLICIES.md and it re-surfaces; move_gate stays out BY NAME (MOVE_GATE_DISPOSITION)."""
    stripped = "\n".join(ln for ln in REAL.splitlines() if "gradient_claim_gate" not in ln)
    assert stripped != REAL, "fixture stripped nothing"
    fake = tmp_path / "POLICIES.md"
    fake.write_text(stripped, encoding="utf-8")
    monkeypatch.setattr(pm, "POLICIES", fake)
    drift = pm.census([])
    assert "gradient_claim_gate" in drift["unclaimed_gates"]
    assert "move_gate" not in drift["unclaimed_gates"]


def test_a_gate_born_in_gates_after_g8_is_counted(monkeypatch, tmp_path):
    """2026-09-29 audit (evidence 3.19): keyed on the frozen label alone, the census could not see
    any gate added after G8. A planted unclaimed gate the live registry places in `gates` must
    surface; the same gate named in POLICIES.md is claimed; move_gate stays exempt by name; a
    gate-named brain module outside `gates` with a non-`substrate` label stays out."""
    plant = "zz_planted_gate"
    real_mods, real_pkg, real_region = pm._selfcheck_modules, pm.package_of, pm.region_of
    monkeypatch.setattr(pm, "_selfcheck_modules", lambda: real_mods() | {plant})
    monkeypatch.setattr(pm, "package_of", lambda m: "gates" if m == plant else real_pkg(m))
    monkeypatch.setattr(pm, "region_of", lambda m: None if m == plant else real_region(m))
    drift = pm.census([])
    assert plant in drift["unclaimed_gates"]
    assert "move_gate" not in drift["unclaimed_gates"]
    assert "reticular_gate" not in drift["unclaimed_gates"]
    assert "developmental_envelope" not in drift["unclaimed_gates"]
    claimed = tmp_path / "POLICIES.md"
    claimed.write_text(REAL + f"\n`{plant}`\n", encoding="utf-8")
    monkeypatch.setattr(pm, "POLICIES", claimed)
    assert plant not in pm.census([])["unclaimed_gates"]


def test_a_post_g8_gate_outside_gates_reds_the_assert(monkeypatch, tmp_path, capsys):
    """Panel fold (evidence 3.19): a gate-named module born after G8 OUTSIDE `gates` (no frozen
    label) is counted too -- fail-closed -- and reds `--assert` at its unclaimed-growth check. The
    committed manifest is re-derived to include the plant so the sync check cannot decide first."""
    plant = "zz_planted_gate"
    real_mods, real_pkg, real_region = pm._selfcheck_modules, pm.package_of, pm.region_of
    monkeypatch.setattr(pm, "_selfcheck_modules", lambda: real_mods() | {plant})
    monkeypatch.setattr(pm, "package_of", lambda m: "instruments" if m == plant else real_pkg(m))
    monkeypatch.setattr(pm, "region_of", lambda m: None if m == plant else real_region(m))
    policies = pm.cross_check(pm.derive())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(pm.committed_payload(policies, pm.census(policies)), encoding="utf-8")
    monkeypatch.setattr(pm, "MANIFEST_JSON", manifest)
    assert pm.run_assert() == 1
    assert "NEW unclaimed gate(s) ['zz_planted_gate']" in capsys.readouterr().out


# ---------------------------------------------------------- T011: manifest rows


def test_kind_derivation_on_the_real_tree():
    by_id = {p.id: p for p in pm.parse(REAL)}
    # Review-only policies render honest (never green, never a fake red):
    for pid in ("P0", "P1", "P2", "P5"):
        assert by_id[pid].kind == "review-only", pid
        assert by_id[pid].commands == []
    # P6 is a ratchet by its own row/body wording; P3/P4 mechanized:
    assert by_id["P6"].kind == "ratchet"
    assert by_id["P3"].commands, "P3 lists its three gates"
    # e-prop: PARTIAL split, never averaged, with both gates attached:
    assert by_id["e-prop"].kind == "partial"
    assert {c.source_token for c in by_id["e-prop"].commands} == {
        "gates/eprop_transport_gate.py",
        "gates/gradient_claim_gate.py",
    }


def test_p4b_is_not_machine_decided_the_overclaim_control():
    # P4b must NEVER inherit P4's replication_guard: the durable-lake clause is
    # review-only, and rendering it mechanized is the overclaim FR-008 forbids.
    by_id = {p.id: p for p in pm.parse(REAL)}
    p4b = by_id["P4b"]
    assert p4b.subclause_of == "P4"
    assert p4b.kind == "review-only"
    assert p4b.commands == []
    rendered = pm.render(pm.parse(REAL))
    assert re.search(r"^\s+P4b\s+NOT MACHINE-DECIDED\s+.*\sNONE$", rendered, re.M)


def test_p4c_command_comes_from_the_gate_output_table():
    by_id = {p.id: p for p in pm.parse(REAL)}
    p4c = by_id["P4c"]
    assert p4c.kind == "ratchet"
    # The gate-output row names --assert, so the derived command is the
    # ASSERTING form -- a bare report-run would be a PASS that cannot fail.
    assert [c.source_token for c in p4c.commands] == ["tools/p4c_ratchet.py --assert"]
    # And the amendment never borrows it:
    assert by_id["P4c amendment"].commands == []


def test_sc007_a_tenth_policy_appears_with_no_list_edited():
    # Synthetic P9: append a heading AND an enforcement-table row; the manifest
    # gains exactly one top-level row, correctly kinded, still reconciling --
    # with ZERO code edits. A hardcoded policy list fails this.
    doc = REAL.replace(
        "| *(not a policy — repo hygiene)* citations resolve",
        "| P9 synthetic-policy | — | yes |\n| *(not a policy — repo hygiene)* citations resolve",
        1,
    )
    doc += "\n## P9 — synthetic tenth policy for SC-007\n\nprose body.\n"
    assert doc != REAL
    base = {p.id for p in pm.parse(REAL)}
    grown = {p.id: p for p in pm.parse(doc)}
    assert set(grown) - base == {"P9"}
    assert grown["P9"].kind == "review-only"
    assert grown["P9"].commands == []


# ------------------------------------------------------- T011: census + drift


def test_the_unclaimed_census_on_the_real_tree_is_now_empty():
    # FLIPPED at T028 (2026-08-18): this test originally PINNED the three live
    # reds (doc_budget_gate, ingest_guard, safety_envelope); US4 claimed all
    # three in the enforcement table, so the census is now asserted EMPTY. The
    # detector itself is still proven live by the removal fixture below.
    policies = pm.cross_check(pm.parse(REAL))
    drift = pm.census(policies)
    assert drift["unclaimed_gates"] == []


def test_the_unclaimed_detector_is_still_live_after_the_flip(monkeypatch, tmp_path):
    # An empty census must be a CLEAN TREE, never a dead detector (SI 3): strip
    # ingest_guard's claiming row from a copy of POLICIES.md and the census
    # must re-surface it.
    stripped = "\n".join(ln for ln in REAL.splitlines() if "ingest_guard" not in ln)
    assert stripped != REAL, "fixture stripped nothing"
    fake = tmp_path / "POLICIES.md"
    fake.write_text(stripped, encoding="utf-8")
    monkeypatch.setattr(pm, "POLICIES", fake)
    drift = pm.census([])
    assert "ingest_guard" in drift["unclaimed_gates"]
    # The instrument's own exclusion is NAMED, never silent:
    assert pm.SELF_NAME not in drift["unclaimed_gates"]
    assert "policy_manifest" in drift["self_disposition"]
    assert "EXCLUDED" in drift["self_disposition"]
    assert pm.MOVE_GATE_DISPOSITION in pm.render_audit([], drift)


def test_not_in_ci_fires_on_a_constructed_orphan_command():
    # A mechanized policy claiming a command nothing in CI runs -> NOT_IN_CI.
    ghost = pm.Policy(
        id="PX",
        title="constructed",
        kind="absolute",
        commands=[
            pm.GateCommand(
                argv=["python", "tools/ghost_gate.py"], source_token="tools/ghost_gate.py"
            )
        ],
    )
    drift = pm.census(pm.cross_check([ghost]))
    assert "tools/ghost_gate.py" in drift["not_in_ci"]
    # Positive control: the real tree's claimed commands are all CI-reachable.
    real = pm.census(pm.cross_check(pm.parse(REAL)))
    assert real["not_in_ci"] == []


def test_the_gate_predicate_is_stated_and_selective():
    # The predicate must ADMIT the gate-shaped registrations and REJECT the
    # bulk: fewer than half of run_selfchecks' substrate modules are gates.
    mods = pm._selfcheck_modules()
    assert len(mods) > 20, "census surface unexpectedly small -- selfchecks parse broke?"
    gates = {m for m in mods if pm.GATE_NAME_RE.search(m)}
    assert {"dormancy_gate", "replication_guard", "import_manifest", "safety_envelope"} <= gates
    assert "cockpit" not in gates and "chat_memory" not in gates
    assert len(gates) < len(mods) / 2


# ---------------------------------------------------- T017: audit + certify


def _stub(policy_id="PS", kind="absolute", code="pass"):
    import sys as _sys

    return pm.Policy(
        id=policy_id,
        title="fixture",
        kind=kind,
        commands=[
            pm.GateCommand(argv=[_sys.executable, "-c", code], source_token=f"fixture:{code}")
        ],
    )


def test_fail_and_cannot_run_are_distinct_in_one_run():
    # Two constructed rows, one audit: FAIL (ran, exit 1) vs CANNOT_RUN (no exe).
    failing = _stub("PF", code="raise SystemExit(1)")
    missing = pm.Policy(
        id="PM",
        title="fixture",
        kind="absolute",
        commands=[pm.GateCommand(argv=["missing_exe_xyz_021"], source_token="fixture:missing")],
    )
    results, commands_run = pm.run_audit([failing, missing])
    by = {r.policy_id: r for r in results}
    assert by["PF"].verdict == "FAIL" and "exit 1" in by["PF"].detail
    assert by["PM"].verdict == "CANNOT_RUN" and "missing_exe_xyz_021" in by["PM"].detail
    assert by["PF"].detail != by["PM"].detail
    assert len(commands_run) == 2 and all("seconds" in c for c in commands_run)


def test_certify_level_constructed_reds_survive_a_clean_census():
    # SC-004's demonstration must survive US4 clearing the census: with ZERO
    # drift, a FAIL row and a CANNOT_RUN row must each refuse certification
    # and name the policy.
    clean = {"unclaimed_gates": [], "not_in_ci": []}
    for verdict in ("FAIL", "CANNOT_RUN", "NO_GATE"):
        certified, reasons = pm.decide_certified(
            [pm.AuditResult("PX", verdict, "constructed")], clean
        )
        assert not certified, verdict
        assert any("PX" in r and verdict in r for r in reasons), verdict


def test_certify_decision_on_the_real_tree_after_t028():
    # FLIPPED at T028: the census is claimed, so the decide-level real-tree
    # verdict is certifiable (gate execution not exercised here; the refusal
    # red is proven by the constructed certify-level tests above, which is
    # what lets this flip without losing SC-004's demonstration).
    drift = pm.census(pm.cross_check(pm.parse(REAL)))
    certified, reasons = pm.decide_certified([], drift)
    assert certified and reasons == []


def test_green_path_positive_control_certificate_is_issuable(monkeypatch, tmp_path):
    # An all-stub-pass fixture with a clean census: certified true is REACHABLE,
    # commands_run non-empty with durations, review_only disclosed. The writer
    # is pointed at tmp -- a test must never republish a tracked artifact.
    monkeypatch.setattr(pm, "CERTIFICATION_JSON", tmp_path / "cert.json")
    monkeypatch.setattr(
        pm,
        "census",
        lambda policies: {
            "unclaimed_gates": [],
            "not_in_ci": [],
            "self_disposition": pm.SELF_DISPOSITION,
        },
    )
    fixture = [
        _stub("PA"),
        pm.Policy(id="PR", title="fixture", kind="review-only"),
    ]
    rc = pm.certify(fixture, full=False)
    assert rc == 0
    import json as _json

    record = _json.loads((tmp_path / "cert.json").read_text(encoding="utf-8"))
    assert record["certified"] is True
    assert record["review_only"] == ["PR"]
    assert record["commands_run"] and all("seconds" in c for c in record["commands_run"])
    assert isinstance(record["tree"]["dirty"], bool)


def test_refusing_certificate_is_recorded_not_just_printed(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "CERTIFICATION_JSON", tmp_path / "cert.json")
    rc = pm.certify([_stub("PF", code="raise SystemExit(1)")], full=False)
    assert rc == 1
    import json as _json

    record = _json.loads((tmp_path / "cert.json").read_text(encoding="utf-8"))
    assert record["certified"] is False
    assert any("PF" in r for r in record["refusal_reasons"])


# ------------------------------------------------ Phase 8 convergence: T035/T036


def test_audit_json_mode_mirrors_the_schema():
    # T035: --audit --json emits one object (results + drift), per contracts/cli.md.
    import json as _json
    import subprocess as _sp
    import sys as _sys

    proc = _sp.run(
        [_sys.executable, "-m", "snn_data_lake.gates.policy_manifest", "--audit", "--json"],
        capture_output=True,
        text=True,
        cwd=str(pm.REPO),
        timeout=600,
    )
    obj = _json.loads(proc.stdout)
    assert set(obj) == {"results", "drift"}
    assert all({"policy", "verdict", "detail"} <= set(r) for r in obj["results"])
    assert {"unclaimed_gates", "not_in_ci", "self_disposition"} <= set(obj["drift"])


def test_text_render_carries_binds_what():
    # T036: FR-001's "what it binds" appears in the reviewer-facing table too.
    rendered = pm.render(pm.parse(REAL))
    assert "binds" in rendered.splitlines()[2]
    assert "Brain-native" in rendered  # P1's title text visible in the table
