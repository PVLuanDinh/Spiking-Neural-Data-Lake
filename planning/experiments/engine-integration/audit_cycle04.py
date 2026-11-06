"""Independent read-only receipt/journal rescore. No Brain, learning or lake opening."""
import base64
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import zlib


ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/engine-runtime')


def load(name): return json.loads((ROOT / name).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pairs(value):
    if type(value) is list and len(value) == 2 and all(type(x) is str for x in value): return [value]
    if type(value) is list: return [p for v in value for p in pairs(v)]
    raise ValueError('unexpected prediction shape')


def main():
    old, grow = load('cycle-04-old.json'), load('cycle-04-grow.json')
    prior = load('cycle-03-failed-retention.json')
    # Only registered metadata/codes for already consumed fixture sets, never model execution.
    code = '''import json
from inspect import signature
from snn_data_lake.instruments.taught_problems import PROBLEMS
from snn_data_lake.instruments import inseparable_mapping as im, delayed_cue as dc
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner
d={p:{s:[[[t.text,t.role,t.label] for t in e] for e in PROBLEMS[p].episodes(9900,s)] for s in ('validation','future')} for p in ('aii','ii.xor')}
print(json.dumps({'episodes':d,'ranks':{t[0]:im.ranks(t[0]) for v in d.values() for es in v.values() for e in es for t in e},'old_buckets':list(dc.BUCKETS_AII.values()),'labels':list(im._IXN_INTENTS),'t_post':im.T_POST,'tau':signature(SpikingInteractionLearner).parameters['tau'].default}))'''
    meta = json.loads(subprocess.check_output(['/bin/zsh', str(ROOT / 'run-engine.sh'), '-c', code],
        env=dict(os.environ, ENGINE_SOURCE_ROOT=str(RUNTIME / 'working-source')), cwd=RUNTIME / 'working-source', text=True))
    checks, scores = {}, {}

    def score(name, value, pid, split='validation'):
        pp = pairs(value); expected = [t[2] for e in meta['episodes'][pid][split] for t in e if t[1] == 'probe']
        assert [b for a, b in pp] == expected, (name, 'target sequence mismatch')
        result = {'correct': sum(a == b for a, b in pp), 'presentations': len(pp)}
        scores[name] = result
        return result

    for name, d in (('old', old), ('grow', grow)):
        for pid, value in d['validation'].items(): score(f'{name}.validation.{pid}', value, pid)
    for pid, value in grow['final_reads'].items(): score(f'grow.final.{pid}', value, pid, 'future')
    for pid, value in grow['grown_off'].items(): score(f'grow.grown_off.{pid}', value, pid)
    for pid, value in prior['validation'].items(): score(f'prior_allplastic.{pid}', value, pid)
    for mode, d in grow['channel_probes'].items():
        rescored = score(f'grow.channel.{mode}', d['reads'], 'ii.xor')
        assert rescored['correct'] == d['correct']

    # Read raw framing with rb only; do not instantiate BinLog (its constructor may repair tails).
    lake = Path(load('cycle-04-home.json')['home']) / 'brain_lake.bin'
    before = digest(lake); raw = lake.read_bytes(); assert raw[:8] == b'SNDLBIN1'
    offset, events, frames = 8, [], 0
    while offset < len(raw):
        assert offset + 4 <= len(raw)
        size = struct.unpack_from('<I', raw, offset)[0]; offset += 4
        assert size > 0 and offset + size <= len(raw)
        payload = raw[offset:offset + size]; offset += size; frames += 1
        if payload[:1] == b'\x0b': events.append(json.loads(payload[1:]))
    checks['lake_rb_only_no_change'] = before == digest(lake)
    epoch_reports = {}
    for pid, epochs, teaches, deposits in (('aii', 22, 88, 66), ('ii.xor', 107, 428, 0)):
        rows = [r for r in events if r.get('kind') == 'engine_epoch' and r.get('problem_id') == pid]
        ends = [r for r in events if r.get('kind') == 'engine_attempt_end' and r.get('problem_id') == pid]
        assert len(rows) == epochs and len(ends) == 1
        streak = 0; first_all = None
        expected = [t[2] for e in meta['episodes'][pid]['validation'] for t in e]
        for i, r in enumerate(rows, 1):
            pp = pairs(r['reads']); correct = sum(a == b for a, b in pp)
            assert [b for a, b in pp] == expected
            streak = streak + 1 if correct == len(pp) else 0
            assert r['epoch'] == i and r['teaches'] == i * 4
            assert r['correct'] == correct and r['presentations'] == len(pp) == 16 and r['stable_run'] == streak
            if correct == 16 and first_all is None: first_all = i
        end = ends[0]['exposure']
        assert end['epochs'] == epochs and end['teach_episodes'] == teaches and end['stable_run'] == 10
        assert end['shared_plasticity_deposits'] == deposits and end['validation_presentations'] == 16 * epochs
        epoch_reports[pid] = {'epochs': epochs, 'teach_presentations': teaches, 'validation_presentations': epochs * 16,
            'shared_plasticity_deposits': deposits, 'first_all_correct_epoch': first_all,
            'final_stable_run': streak, 'all_epoch_rows_rescored': len(rows)}
    # Console is a subset; verify every printed raw row against actual journal rows.
    for phase in ('old', 'grow'):
        for line in (ROOT / f'cycle-04-{phase}.log').read_text().splitlines():
            if line.startswith('ENGINE '):
                row = json.loads(line[7:])
                if row['kind'] == 'progress': assert row['event'] in events
    checks['raw_console_rows_equal_journal'] = True
    checks['canonical_neurons_and_complete_stack_exact_old_to_grow'] = old['core_state'] == grow['core_state']
    assert checks['canonical_neurons_and_complete_stack_exact_old_to_grow']

    sources = [r for r in events if r.get('kind') == 'engine_source']
    assert sources == grow['source_events'] and len(sources) == 1
    source = sources[0]
    assert source['evaluations'] == 1024 and source['admitted_ids'] == list(range(18)) and source['complete_batch']
    assert grow['problem']['spend_units'] == 18 and grow['problem']['spend_evals'] == 1024
    checks['source_18_of_1024_one_batch_stream9900'] = source['stream_seed'] == 9900
    schedule_summary = {}
    for pid, actor in grow['actors'].items():
        carrier = actor['factory']['engine_binding']; data = zlib.decompress(base64.b64decode(carrier['payload']))
        assert len(data) == carrier['raw_size'] and hashlib.sha256(data).hexdigest() == carrier['sha256']
        config = json.loads(data); assert config['cycle'] == carrier['cycle']
        counts = [len(epoch) for epoch in config['schedule']]
        unique = {tuple(tuple(t) for t in ep) for epoch in config['schedule'] for ep in epoch}
        assert set(counts) == {4} and len(unique) == 4
        schedule_summary[pid] = {'unique_teach_items': len(unique), 'teach_items_per_epoch': 4,
            'validation_unique_texts': len({t[0] for e in meta['episodes'][pid]['validation'] for t in e}),
            'validation_presentations': 16, 'repetitions_per_unique_item': 4, 'actor_teaches': actor['teaches']}

    channels = {}; baseline = grow['channel_probes']['baseline']
    for mode, d in grow['channel_probes'].items():
        p, b = d['pool_trace'], baseline['pool_trace']; g, bg = d['grown_trace'], baseline['grown_trace']
        assert len(p) == len(g) == 16 and [r['text'] for r in p] == [r['text'] for r in b]
        l2 = math.sqrt(sum((x - y) ** 2 for a, z in zip(p, b) for x, y in zip(a['features'], z['features'])))
        assert math.isclose(l2, d['pool_feature_l2_delta'], abs_tol=1e-12)
        channels[mode] = {'score': scores[f'grow.channel.{mode}'], 'pool_feature_l2_delta': l2,
            'feed_l2_minmax': [min(r['feed_l2'] for r in p), max(r['feed_l2'] for r in p)],
            'return_weights_l2': d['return_weights_l2'], 'afferent_l1_minmax': [min(r['afferent_l1'] for r in g), max(r['afferent_l1'] for r in g)],
            'changed_latency_entries': sum(x != y for a, z in zip(g, bg) for x, y in zip(a['latencies'], z['latencies'])),
            'grown_drive_l2_delta': math.sqrt(sum((x - y) ** 2 for a, z in zip(g, bg) for x, y in zip(a['grown_drives'], z['grown_drives'])))}

    # Independent arithmetic check of planted old-feature lesion, using saved actual
    # canonical weights and captured actual grown drives. Part D is unaffected by
    # this canonical-core-only lesion, so its captured return continuation is fixed.
    neurons = grow['core_state']['neurons']; weights = [r['attrs']['w'] for r in neurons]
    assert len(weights) == 9 and all(len(w) == 52 for w in weights)
    def predict(text, grown, lesion=False):
        ranks = meta['ranks'][text]; values = []
        for w, gg in zip(weights, grown):
            values.append(sum((0.0 if lesion and j in meta['old_buckets'] else value) * (math.exp(-(meta['t_post'] - t) / meta['tau']) if meta['t_post'] - t > 0 else 0.0)
                              for j, (value, t) in enumerate(zip(w, ranks))) + gg)
        return meta['labels'][max(range(9), key=lambda i: values[i])]
    lesion = {}
    for pid in ('aii', 'ii.xor'):
        episodes = meta['episodes'][pid]['validation']; values = []
        for i, ep in enumerate(episodes):
            t = ep[-1]; gg = [0.0] * 9 if pid == 'aii' else baseline['grown_trace'][i]['grown_drives']
            base, cut = predict(t[0], gg), predict(t[0], gg, True)
            assert base == grow['validation'][pid][i][0][0]
            values.append((cut, t[2]))
        lesion[pid] = {'correct': sum(a == b for a, b in values), 'presentations': 16,
            'predictions_unchanged': [a for a, b in values] == [p[0] for p in pairs(grow['validation'][pid])]}
    assert lesion['aii']['correct'] == grow['lesioned_old_correct'] == 0
    assert lesion['ii.xor']['predictions_unchanged'] and grow['old_skill_lesion_refused']

    restores = {}
    for name in ('cycle-04-grow-restore-2.json', 'cycle-04-grow-restore-journal.json'):
        d = load(name)
        for pid, value in d['validation'].items(): score(f'{name}.{pid}', value, pid)
        exact = {k: d[k] == grow[k] for k in ('model_digest', 'core_state', 'actors', 'validation', 'source_events')}
        assert all(exact.values()) and d['fresh_process_equal'] and d['final_repeat_refused']
        restores[name] = {'journal_only': d['journal_only'], 'exact': exact, 'final_repeat_refused': True,
            'reason': d.get('final_repeat_reason', 'refusal asserted by recorded controller')}
    production = None; provenance = {}
    for phase in ('old', 'grow', 'restore-2', 'restore_journal'):
        d = load(f'cycle-04-{phase}-controller.json'); assert d['exit_code'] == 0
        if production is None: production = d['source_sha256']
        assert production == d['source_sha256']
        for path, sha in production.items(): assert digest(ROOT / f'cycle-04-{phase}-source' / path) == sha
        assert digest(ROOT / f'cycle-04-{phase}-script.py') == d['script_sha256']
        provenance[phase] = {'exit_code': d['exit_code'], 'controller_seconds': d['seconds'], 'source_hashes_verified': len(production)}
    result = {'scope': 'independent artifact and raw-journal audit, no Brain or learning run', 'seed': 9900,
        'checks': checks, 'scores': scores, 'actual_epoch_history': epoch_reports, 'schedule_summary': schedule_summary,
        'source': {k: source[k] for k in ('batch_id', 'stream_seed', 'evaluations', 'admitted_ids', 'selected_source_indices')},
        'channel_interventions': channels, 'lesion_arithmetic_rescore': lesion, 'restores': restores,
        'provenance': provenance, 'lake': {'sha256': before, 'frames': frames, 'goal_events': len(events)},
        'retained_negative_control': {'allplastic_A': scores['prior_allplastic.aii'], 'allplastic_B': scores['prior_allplastic.ii.xor'],
            'no_frozen_selection': prior['selection'] is None, 'no_final_claims': prior['claims'] == {}},
        'limitations': ['Single already exposed engineering seed; no independent statistical cohort or efficacy comparison.',
            'Four unique constructed text/code items per task, repeated four times per read set; not 16 independent cases.',
            'Future/final is a different presentation order of those same four items, not a new semantic generalization set.',
            'Task-conditioned context, corrective feedback, explicit protected core, one admitted batch of18, no pruning; no autonomous RSI claim.',
            'Feed ablation changes Part D features but retains16/16; no demonstrated final-answer feed benefit.',
            'Return ablation reduces B to8/16 on this fixture; not a general causal/statistical effect estimate.',
            'Canonical neurons+complete stack are independently equal; full shared-state digest is compared as recorded, not recomputed from an unavailable full shared snapshot.',
            'Lesion rescore uses saved core weights plus actual captured grown-drive continuation; controller independently asserted real lesion refusal.',
            'First checkpoint restore controller retained an assertion failure caused by tuple/list normalization; corrected checkpoint and journal-only controllers pass.',
            'No SHD result or full operator battery is implied.']}
    (ROOT / 'cycle-04-independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checks': checks, 'scores': scores, 'epochs': epoch_reports, 'channels': channels, 'lesion': lesion, 'restores': restores}, indent=2))


if __name__ == '__main__': main()
