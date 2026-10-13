"""G6 all-eligible-circuit static census -- tests.

STATIC ONLY. This file imports the census module and pytest and stdlib. It does NOT import
the engine, construct a Brain or an AutonomousArm, instantiate any population, or open a home.
The runtime refusal is honoured here as a TESTED property, not as a promise:
`test_census_module_never_imports_or_constructs_the_engine` reads the module's own source and
fails if it ever imports the engine or calls a driver constructor.

P8. Every criterion carries a RED-FIRST test: a deliberately broken throwaway is built, the
criterion is shown firing on it, and the unmutated tree is shown green. A criterion with no
witnessed red would be deleted, not weakened -- the witnesses below are what stop that.
"""

import ast
import dataclasses
import os
import shutil

import pytest

from snn_data_lake.instruments import eligible_circuit_census as ecc

REPO = ecc._repo_root()


# --------------------------------------------------------------------------------------
# Throwaway copies. The census reads SOURCE TEXT, so a witness needs only the files it reads
# -- not a checkout, not a git worktree, and certainly not a running brain.
# --------------------------------------------------------------------------------------


def _source_files_needed(root):
    """Every file the census reads: the three drivers, every cited file, and every module
    that defines a discovered callee."""
    needed = {rel for _cls, rel in ecc.DRIVERS}
    needed |= {rec.file for rec in ecc.RECORDS}
    for site in ecc.discover_sites(root):
        if site.defining_module not in ("-", "LOCAL"):
            needed.add("src/snn_data_lake/{}.py".format(site.defining_module.replace(".", "/")))
    return sorted(needed)


def _throwaway(tmp_path, root=REPO):
    """A minimal source copy the census can run against, with no engine import anywhere."""
    dest = tmp_path / "throwaway"
    for rel in _source_files_needed(root):
        src = os.path.join(root, *rel.split("/"))
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    return str(dest)


def _patch(path, old, new):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert old in text, f"plant anchor not found in {path}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new, 1))


# --------------------------------------------------------------------------------------
# The owner stop, as a tested property.
# --------------------------------------------------------------------------------------


def test_census_module_never_imports_or_constructs_the_engine():
    """The module must stay static: no engine import, no driver construction.

    RED witness: plant an `import` of the engine into a copy of the module and this fails.
    """
    path = os.path.join(REPO, "src", "snn_data_lake", "instruments", "eligible_circuit_census.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "engine" not in alias.name, "census must not import the engine"
        elif isinstance(node, ast.ImportFrom):
            assert "engine" not in (node.module or ""), "census must not import the engine"
        elif isinstance(node, ast.Call):
            callee = (
                node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            )
            assert callee not in {
                "Brain",
                "AutonomousArm",
                "LiveBrain",
                "NeurogenesisSelfImprove",
            }, f"census must not construct a driver or population, found {callee}()"

    # red-before: the same property, checked against a deliberately broken copy
    broken = src.replace(
        "import ast\n", "import ast\nfrom snn_data_lake.engine.brain import Brain\n", 1
    )
    assert "from snn_data_lake.engine.brain import Brain" in broken, "RED PLANT FAILED"
    bad_tree = ast.parse(broken)
    hits = [
        a.name
        for n in ast.walk(bad_tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if "engine" in (n.module or "")
    ]
    assert hits == ["Brain"], "red control failed: the broken copy must be caught"


def test_census_module_defines_no_second_framework():
    assert ecc.no_second_framework() is True


# --------------------------------------------------------------------------------------
# Green controls on the real tree.
# --------------------------------------------------------------------------------------


def test_named_minimums_present_on_the_real_tree():
    assert ecc.named_minimums_present() is True


def test_every_record_is_grounded_on_the_real_tree():
    assert ecc.no_duplicate_disposition() is True
    assert ecc.every_nested_row_names_a_real_container() is True
    assert ecc.every_citation_resolves(REPO) is True
    assert ecc.every_unresolved_row_names_its_gap() is True
    assert ecc.no_invented_born_identity() is True


def test_every_discovered_handle_is_disposed_on_the_real_tree():
    assert ecc.undisposed_sites(REPO) == ()
    assert ecc.every_object_handle_is_disposed(REPO) is True


def test_runtime_port_is_refused():
    """The runtime half must never be reported done by this static module."""
    with pytest.raises(AssertionError) as excinfo:
        ecc.runtime_port_is_refused()
    message = str(excinfo.value)
    # Wording updated 2026-10-12 to the truthful reason (guarded sequence approved at
    # #1084/#1086, hygiene closed at #1226). The ASSERTION is unchanged and NOT weaker: it
    # still requires the refusal to RAISE, and now additionally pins the two substantive
    # facts -- the runtime slice is UNBUILT and it needs its own grant.
    assert "UNBUILT" in message
    assert "own explicit grant" in message


# --------------------------------------------------------------------------------------
# The map's own inventory must be covered. If the authoritative map (#1061) names a
# population, this census must say something about it -- not silently omit it.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "population",
    [
        "Brain.attractor",
        "Brain.neuro._model",
        "Brain.neuro.online_readout",
        "Brain.brain._ixn",
        "Brain._plasticity",
        "AutonomousArm.learner",
        "Brain._vis_learner",
        "Brain._aud_learner",
        "Brain._ll",
        "Brain._affect_int",
        "Brain._reason",
        "LiveBrain._pc",
        "Brain._percept_readout",
        "Brain._percept_surprise",
        "AutonomousArm.jepa",
        "Brain._affect_appraisal",
        "Brain._gsm",
        "Brain._cfsim",
        "Brain._task_set",
        "Brain._learned_attn",
        "Brain._learned_evc",
        "Brain._episode_consolidator",
        "Brain._time_clock",
        "Brain._acq_lm",
        "Brain._skills",
        "AutonomousArm.amyg",
        "AutonomousArm._valence",
        "G3 born units",
        "G3/G4/G5 readout",
    ],
)
def test_the_authoritative_map_populations_are_each_disposed(population):
    """Every population the map enumerates must carry a census row.

    RED witness: remove the row and this fails, because the map's coverage is asserted.
    """
    flat = " | ".join(rec.owner_path for rec in ecc.RECORDS)
    assert population in flat, f"map population {population!r} has no census row"


def test_the_reservoir_pool_row_cites_its_own_source():
    """The pool row's citation is a real file, not just a claimed name."""
    pool = [r for r in ecc.RECORDS if "G3/G4/G5 readout" in r.owner_path]
    assert len(pool) == 1
    assert pool[0].file == "src/snn_data_lake/reservoir/recurrent_pool.py"
    assert os.path.isfile(os.path.join(REPO, *pool[0].file.split("/")))


def test_dispositions_come_from_the_maps_own_vocabulary():
    assert all(rec.disposition in ecc.DISPOSITIONS for rec in ecc.RECORDS)


def test_unresolved_rows_exist_and_are_not_vacuous():
    """An all-eligible census with zero UNRESOLVED rows would be claiming a completeness
    the source does not support. The map's own disposition is UNRESOLVED/UNBUILT."""
    unresolved = [r for r in ecc.RECORDS if r.disposition == "UNRESOLVED"]
    assert unresolved, "a static census must retain its UNRESOLVED rows"
    assert all(len(r.reason) >= 25 for r in unresolved)


def test_only_the_g3_row_claims_a_born_identity():
    for rec in ecc.RECORDS:
        if rec.identity.startswith("NATIVE_BORN_ID"):
            assert rec.owner_path.startswith("G3 born units"), (
                f"{rec.owner_path} claims a born identity but is not the G3 population"
            )


def test_legacy_index_identities_are_never_dressed_as_born_ids():
    """The map's line 42: 'Legacy names are not born IDs'."""
    for rec in ecc.RECORDS:
        if "LIST_INDEX" in rec.identity or "LABEL_INDEX" in rec.identity:
            assert "NATIVE_BORN_ID" not in rec.identity
            assert rec.owner_path != "G3 born units"


# --------------------------------------------------------------------------------------
# RED-FIRST witnesses. Each builds a broken THROWAWAY, shows the criterion firing, then
# shows the unmutated tree green. A criterion without one of these would be deleted, not
# weakened.
# --------------------------------------------------------------------------------------


def test_red_a_new_learned_population_in_the_brain_is_caught(tmp_path):
    """RED A -- the exhaustiveness red. Plant a brand-new object handle in a copy of
    brain.py; every_object_handle_is_disposed must raise and name the site."""
    root = _throwaway(tmp_path)
    brain = os.path.join(root, "src", "snn_data_lake", "engine", "brain.py")
    # green control FIRST: the unmutated throwaway is clean, so the red below is attributable
    assert ecc.every_object_handle_is_disposed(root) is True
    assert ecc.undisposed_sites(root) == ()

    _patch(
        brain,
        "        self._closed = False",
        "        self._closed = False\n        self._planted_new_learned_bank = NightlySkills()",
    )
    with pytest.raises(AssertionError) as excinfo:
        ecc.every_object_handle_is_disposed(root)
    message = str(excinfo.value)
    assert "_planted_new_learned_bank" in message, "the red must NAME the undisposed site"
    # ...and the real checkout was never touched by the plant
    assert ecc.undisposed_sites(REPO) == ()


def test_red_b_renaming_a_declared_handle_leaves_it_undisposed(tmp_path):
    """RED B -- if a population's attribute is renamed and the census is not updated, the
    old declaration stops covering anything and the new name is undisposed."""
    root = _throwaway(tmp_path)
    live = os.path.join(root, "src", "snn_data_lake", "engine", "live_brain.py")
    _patch(live, "self._ixn = (", "self._ixn_renamed = (")
    undisposed = {s.key for s in ecc.undisposed_sites(root)}
    assert "LiveBrain._ixn_renamed" in undisposed, "the renamed handle must be undisposed"
    assert ecc.undisposed_sites(REPO) == ()


def test_red_c_a_citation_to_a_missing_file_is_caught():
    """RED C -- a record citing a file that does not exist must be refused."""
    broken = ecc.RECORDS + (
        dataclasses.replace(
            ecc.RECORDS[0],
            owner_path="ghost.population",
            sites=("Brain.ghost",),
            file="src/snn_data_lake/does/not/exist.py",
            line=1,
        ),
    )
    with pytest.raises(AssertionError) as excinfo:
        _citations(broken)
    assert "does/not/exist" in str(excinfo.value)
    assert ecc.every_citation_resolves(REPO) is True


def test_red_d_a_citation_past_end_of_file_is_caught():
    """RED D -- a citation line outside the file must be refused, not clamped."""
    broken = ecc.RECORDS + (
        dataclasses.replace(
            ecc.RECORDS[0],
            owner_path="ghost.population",
            sites=("Brain.ghost",),
            line=10**9,
        ),
    )
    with pytest.raises(AssertionError) as excinfo:
        _citations(broken)
    assert "outside the file" in str(excinfo.value)
    assert ecc.every_citation_resolves(REPO) is True


def _citations(records):
    """every_citation_resolves over an arbitrary record tuple (module fn reads RECORDS)."""
    saved = ecc.RECORDS
    try:
        ecc.RECORDS = records
        return ecc.every_citation_resolves(REPO)
    finally:
        ecc.RECORDS = saved


def test_red_e_an_unresolved_row_without_a_cause_is_caught():
    """RED E -- 'UNRESOLVED' with no machine-readable cause is not a disposition."""
    broken = tuple(
        dataclasses.replace(rec, reason="unknown") if rec.disposition == "UNRESOLVED" else rec
        for rec in ecc.RECORDS
    )
    with pytest.raises(AssertionError) as excinfo:
        ecc.every_unresolved_row_names_its_gap(broken)
    assert "machine-readable cause" in str(excinfo.value)
    assert ecc.every_unresolved_row_names_its_gap() is True


def test_red_f_relabelling_a_legacy_index_as_a_born_id_is_caught():
    """RED F -- the map's 'Legacy names are not born IDs' failure, mechanized."""
    broken = tuple(
        dataclasses.replace(rec, identity="NATIVE_BORN_ID (minted by the learner)")
        if rec.owner_path == "Brain._reason"
        else rec
        for rec in ecc.RECORDS
    )
    with pytest.raises(AssertionError) as excinfo:
        ecc.no_invented_born_identity(broken)
    assert "NATIVE_BORN_ID" in str(excinfo.value)
    assert ecc.no_invented_born_identity() is True


def test_red_g_one_handle_disposed_twice_is_caught():
    """RED G -- the map's 'the stack is the SAME object' rule: two rows for one
    population would double count it."""
    broken = ecc.RECORDS + (
        dataclasses.replace(ecc.RECORDS[0], owner_path="duplicate.of.attractor"),
    )
    with pytest.raises(AssertionError) as excinfo:
        ecc.no_duplicate_disposition(broken)
    assert "disposed twice" in str(excinfo.value)
    assert ecc.no_duplicate_disposition() is True


def test_red_h_a_nested_row_with_no_container_is_caught():
    """RED H -- a row may not claim to be nested in a handle nothing disposes of."""
    broken = tuple(
        dataclasses.replace(rec, nested_in="Brain.a_handle_that_does_not_exist")
        if not rec.sites and not rec.nested_in.startswith("NO ")
        else rec
        for rec in ecc.RECORDS
    )
    with pytest.raises(AssertionError) as excinfo:
        ecc.every_nested_row_names_a_real_container(broken)
    assert "no container handle" in str(excinfo.value)
    assert ecc.every_nested_row_names_a_real_container() is True


def test_red_i_a_dropped_named_minimum_is_caught():
    """RED I -- dropping one of the four owner-named minimums must raise."""
    broken = tuple(rec for rec in ecc.RECORDS if "Brain.attractor" not in rec.owner_path)
    with pytest.raises(AssertionError) as excinfo:
        ecc.named_minimums_present(broken)
    assert "Brain.attractor" in str(excinfo.value)
    assert ecc.named_minimums_present() is True


# --------------------------------------------------------------------------------------
# The discovery itself must be honest about what it can and cannot see.
# --------------------------------------------------------------------------------------


def test_the_three_non_direct_construction_shapes_are_detected(tmp_path):
    """The census must see the conditional form and the setattr form, not just `self.X = C()`.

    These are real: live_brain.py:160 builds _ixn under `if interaction_learning else None`,
    and brain.py:1062-1067 builds the vis/aud GrowingLearners by setattr over a
    data-dependent name. A naive scan misses both.
    """
    keys = {s.key for s in ecc.object_handle_sites(REPO)}
    assert "LiveBrain._ixn" in keys, "the conditional-construction population must be visible"
    conditional = {(s.key, s.callee) for s in ecc.discover_sites(REPO) if s.conditional}
    assert ("LiveBrain._ixn", "SpikingInteractionLearner") in conditional
    # the arm's five conditional handles, missed by a direct `self.X = Call()` scan
    for key in (
        "AutonomousArm.workspace",
        "AutonomousArm.bg",
        "AutonomousArm.intero",
        "AutonomousArm.pfc",
    ):
        assert key in keys, f"{key} is built under an if/else and must still be discovered"
    assert "<dynamic>" in {s.attribute for s in ecc.discover_sites(REPO)}, (
        "the setattr-over-a-tuple learners must appear as a data-dependent site"
    )
    kinds = {s.callee_kind for s in ecc.discover_sites(REPO)}
    assert "DATA_DEPENDENT" in kinds, "an unresolvable setattr target must be visible, not skipped"


def test_the_report_states_the_scope_and_establishes_nothing():
    rep = ecc.census_report(REPO)
    assert rep["frozen_base"] == ecc.FROZEN_BASE
    assert rep["undisposed"] == []
    assert rep["establishes"].startswith("scope only")
    assert "UNBUILT" in rep["runtime_port"]
    assert rep["sites_total"] >= 200


def test_the_report_does_not_claim_completeness():
    """The map's disposition is UNRESOLVED/UNBUILT. This row may not round that up."""
    rep = ecc.census_report(REPO)
    assert rep["undisposed"] == [], "the DISPOSITION obligation is discharged..."
    assert rep["records"] > 0
    unresolved = [r for r in ecc.RECORDS if r.disposition == "UNRESOLVED"]
    assert unresolved, "...but the coverage obligation is NOT a completeness claim."
