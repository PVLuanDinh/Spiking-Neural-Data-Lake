"""Local stage-06 diagnostics. Adopts stage-05 carriers and upstream operations."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "05-artifacts"
spec = importlib.util.spec_from_file_location("class05", OLD / "class_block.py")
c5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c5)
s5 = c5.s5
dumps, hashed, require, save = c5.dumps, c5.hashed, c5.require, c5.save
HEAD = s5.HEAD
SEEDS = (8000, 8001, 8002, 8003)
PROFILE = "sndl.stage06.v1"
NAME = "stage06-result"


def tupled(x):
    return tuple(tupled(v) for v in x) if isinstance(x, list) else x


def seal(body):
    return dumps({"body": body, "sha256": hashed(body)}).encode("ascii")


def decode(blob, expected):
    require(blob is not None, "missing state")
    require(c5.digest(blob) == expected, "stale/corrupt receipt")
    env = json.loads(blob)
    require(hashed(env["body"]) == env["sha256"], "internal checksum")
    body = env["body"]
    require(body["profile"] == PROFILE and body["head"] == HEAD, "profile/head mismatch")
    return body


def write_state(body):
    from snn_data_lake.substrate.home import save_named_state
    blob = seal({"profile": PROFILE, "head": HEAD, **body})
    require(save_named_state(NAME, blob, path=str(s5.lake_file())), "write refused")
    return c5.digest(blob)


def disk_state(expected):
    from snn_data_lake.substrate.home import load_named_state
    return decode(load_named_state(NAME, path=str(s5.lake_file())), expected)


def ll_snapshot(unit):
    return {"state": unit.state_dict(), "config": {k: getattr(unit, k) for k in
            ("dim", "budget", "replay_on", "prune_on", "prune_mode", "task_decay", "replay_batch", "epochs", "seed")},
            "memory_hash": hashed(unit.memory.buf), "memory_i": unit.memory._i,
            "memory_capacity": unit.memory.capacity, "memory_rng": unit.memory.rng.getstate(),
            "rng": unit.rng.getstate()}


def ll_rebuild(body):
    from snn_data_lake.sleep.lifelong_consolidation import LifelongConsolidator
    snap = body["snapshot"]
    cfg = snap["config"]
    unit = LifelongConsolidator(dim=cfg["dim"], budget=cfg["budget"], replay=cfg["replay_on"],
          prune=cfg["prune_on"], prune_mode=cfg["prune_mode"], task_decay=cfg["task_decay"],
          replay_batch=cfg["replay_batch"], epochs=cfg["epochs"], seed=cfg["seed"])
    require(unit.load_state_dict(snap["state"]), "invalid prototype structure")
    unit.rebuild_memory(body["trains"])
    require(hashed(unit.memory.buf) == snap["memory_hash"] and unit.memory._i == snap["memory_i"], "rebuilt memory mismatch")
    require(hashed(unit.memory.rng.getstate()) == hashed(snap["memory_rng"]), "rebuilt memory rng mismatch")
    unit.rng.setstate(tupled(snap["rng"]))
    require(hashed(ll_snapshot(unit)) == hashed(snap), "restore state mismatch")
    return unit


def ll_read(unit, xs):
    from snn_data_lake.growth.growing_learner import GrowingLearner
    before, disk = hashed(ll_snapshot(unit)), s5.lake_hash()
    def forbidden(*a, **k):
        raise RuntimeError("learning during evaluation")
    with patch.object(type(unit), "_observe", forbidden), patch.object(type(unit), "learn_task", forbidden), patch.object(GrowingLearner, "observe", forbidden):
        answers = [[unit.classify(x) for x in group] for group in xs]
    require(before == hashed(ll_snapshot(unit)), "evaluation changed learner or memory")
    require(disk == s5.lake_hash(), "evaluation wrote lake")
    return {"answers": answers, "state_hash": before}


def classify_cost(unit, xs):
    """Actual cosine call/coordinate counts, then separate uninstrumented timings."""
    from snn_data_lake.growth import growing_learner as gl
    count = {"cosines": 0, "coordinates": 0}
    original = gl._cosine
    def counted(a, b):
        count["cosines"] += 1
        count["coordinates"] += min(len(a), len(b))
        return original(a, b)
    with patch.object(gl, "_cosine", counted):
        answers = [unit.classify(x) for group in xs for x in group]
    n = sum(map(len, xs))
    require(count["coordinates"] == n * unit.n_units * unit.dim, "work instrument mismatch")
    times = []
    for _ in range(5):
        t, cpu = time.monotonic(), time.process_time()
        for _repeat in range(3):
            got = [unit.classify(x) for group in xs for x in group]
            require(got == answers, "timing changed answers")
        times.append({"wall_s": time.monotonic() - t, "cpu_s": time.process_time() - cpu, "queries": 3 * n})
    return {**count, "queries": n, "timings": times}


def sabotage(unit):
    n = unit.n_units
    victims = set(sorted(range(n), key=lambda i: (-unit.utility[i], i))[:max(0, n - unit.budget)])
    keep = [i for i in range(n) if i not in victims]
    unit.learner.prototypes = [unit.learner.prototypes[i] for i in keep]
    unit.labels = [unit.labels[i] for i in keep]
    unit.utility = [unit.utility[i] for i in keep]


def duration(payload):
    data = payload["data"]
    body = c5.load_checkpoint("b30", payload["receipt"])
    require(body["seed"] == data["seed"] and body["pool"] == data["pool"], "wrong b30 recipe")
    unit, replay = s5.restore_unit(body)
    before = s5.snapshot(unit, replay)
    rows = [data["train"][i] for order in data["orders"][:payload["passes"]] for i in order]
    labels = [r["target"] for r in rows]
    if payload["shuffle"]:
        from snn_data_lake.instruments.taught_campaign import _shuffled_labels
        labels = _shuffled_labels(data["seed"])(labels)
    start, cpu = time.monotonic(), time.process_time()
    for r, label in zip(rows, labels, strict=True):
        unit.learn(r["x"], label)
    cost = {"wall_s": time.monotonic() - start, "cpu_s": time.process_time() - cpu,
            "presentations": len(rows), "labels": labels, "rows_hash": hashed(rows)}
    require(replay == before["replay"], "replay memory changed")
    after = s5.snapshot(unit, replay)
    receipt = write_state({"kind": "duration", "seed": data["seed"], "pool": data["pool"],
             "state": after, "passes": payload["passes"], "shuffle": payload["shuffle"]})
    reads = {sp: s5.read(unit, replay, data[sp]) for sp in ("dev", "final")}
    return {"before": before, "after": after, "receipt": receipt, "reads": reads,
            "scores": {sp: c5.scores(reads[sp]["answers"], data[sp]) for sp in reads}, "work": cost,
            "lake_hash": s5.lake_hash()}


def ll_base(seed):
    from snn_data_lake.sleep import lifelong_consolidation as ll
    tasks = ll._make_tasks(seed)
    unit = ll.LifelongConsolidator(dim=ll.D, budget=ll.BUDGET, replay=True, prune=False, seed=seed)
    xs = [[x for x, _ in t["test"]] for t in tasks]
    history = []
    for i, task in enumerate(tasks):
        unit.learn_task(i, task["train"])
        history.append({"task": i, "n_units": unit.n_units, "read": ll_read(unit, xs)})
    body = {"kind": "prune", "seed": seed, "snapshot": ll_snapshot(unit), "trains": [t["train"] for t in tasks]}
    receipt = write_state(body)
    return {"tasks": tasks, "history": history, "receipt": receipt, "snapshot": ll_snapshot(unit)}


def prune(payload):
    from snn_data_lake.instruments import sleep_pruner_census as cen
    body = disk_state(payload["receipt"])
    unit = ll_rebuild(body)
    before = ll_snapshot(unit)
    arm = payload["arm"]
    unit.prune_on = arm != "off"
    unit.prune_mode = "random" if arm == "random" else "magnitude"
    rec = cen.Recorder()
    sites = {k: v for k, v in cen.SITES.items() if v[0] == "P-LL"}
    t, cpu = time.monotonic(), time.process_time()
    if arm == "sabotage":
        sabotage(unit)
    else:
        with cen._patched(cen.counting_patches(rec, sites=sites)):
            unit._prune_to_budget()
    selection = {"wall_s": time.monotonic()-t, "cpu_s": time.process_time()-cpu,
                 "native_calls": rec.calls("P-LL"), "native_removals": rec.removals("P-LL"),
                 "removed": len(before["state"]["labels"]) - unit.n_units,
                 "candidate_units": len(before["state"]["labels"]), "hyperparameter_candidates": 1}
    after = ll_snapshot(unit)
    old = before["state"]
    kept = [old["learner"]["prototypes"].index(p) for p in after["state"]["learner"]["prototypes"]]
    require([old["labels"][i] for i in kept] == unit.labels, "label alignment")
    require([old["utility"][i] for i in kept] == unit.utility, "utility alignment")
    require(before["memory_hash"] == after["memory_hash"] and before["memory_rng"] == after["memory_rng"], "selection changed replay")
    prefix = s5.lake_file().read_bytes()
    receipt = write_state({**body, "snapshot": after, "arm": arm, "kept": kept})
    require(s5.lake_file().read_bytes().startswith(prefix), "lake history was removed")
    reads = ll_read(unit, payload["xs"])
    work = classify_cost(unit, payload["xs"])
    return {"before": before, "after": after, "receipt": receipt, "read": reads,
            "selection": selection, "kept": kept, "work": work, "lake_hash": s5.lake_hash()}


def restore(payload):
    disk = s5.lake_hash()
    body = disk_state(payload["receipt"])
    if body["kind"] == "duration":
        unit, replay = s5.restore_unit(body)
        rows = c5.rebuild(body, payload["stimuli"])
        reads = {sp: s5.read(unit, replay, rows[sp]) for sp in rows}
        result = {"state": s5.snapshot(unit, replay), "answers": {sp: r["answers"] for sp,r in reads.items()}}
    else:
        unit = ll_rebuild(body)
        result = {"state": ll_snapshot(unit), "answers": ll_read(unit, payload["xs"])["answers"]}
    require(disk == s5.lake_hash(), "restore wrote lake")
    result["lake_hash"] = disk
    return result


def controls():
    from snn_data_lake.sleep.lifelong_consolidation import LifelongConsolidator
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    previous = c5.controls()
    def fixture():
        u = LifelongConsolidator(dim=4, budget=2, seed=8900)
        u.learner.prototypes = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.]]
        u.labels, u.utility = [0,1,0], [20.,10.,1.]
        return u
    u = fixture()
    u._prune_to_budget()
    require(u.labels == [0,1] and u.utility == [20.,10.], "prune rank positive")
    xs = [[[1.,0.,0.,0.],[0.,1.,0.,0.]]]
    require(ll_read(u,xs)["answers"] == [[0,1]], "scorer positive")
    bad = fixture(); sabotage(bad)
    require(ll_read(bad,xs)["answers"] != [[0,1]], "sabotage did not lose")
    rnd = fixture(); rnd.prune_mode = "random"; rnd._prune_to_budget()
    require(rnd.n_units == u.n_units == bad.n_units == 2, "matched masks")
    checks = {"prune_rank": True, "matched_random": True, "sabotage_loses": True}
    blob = seal({"profile": PROFILE, "head": HEAD})
    def catches(name, fn):
        try:
            fn()
        except (RuntimeError, ValueError):
            checks[name] = True
        else:
            checks[name] = False
    catches("missing", lambda: decode(None,"x"))
    catches("stale", lambda: decode(blob,"x"))
    broken = seal({"profile":"wrong", "head":HEAD})
    catches("profile",lambda:decode(broken,c5.digest(broken)))
    badbody = json.loads(blob); badbody["body"]["head"] = "wrong"
    broken = dumps(badbody).encode()
    catches("checksum",lambda:decode(broken,c5.digest(broken)))
    catches("width",lambda:ll_read(u,[[[0.]]]))
    def leaked_read():
        with patch.object(type(u), "classify", lambda self,x:self.learn_task(0,[(x,0)])):
            ll_read(u,xs)
    catches("learning_in_eval", leaked_read)
    require(all(checks.values()), "a stage06 control failed")
    baseline = delta(["GOAL","GREET"],256)
    missing = [k for k in ("neurons","_replay","_deposit","interact","encode","_argmax_label","stack") if not hasattr(baseline,k)]
    require(bool(missing), "stage contract unexpectedly available")
    return {"previous":previous,"checks":checks,"stages":{"status":"NOT_EVALUATED", "missing_delta_contract":missing,
          "reason":"Native staged runner consumes SpikingInteractionLearner and its rest stack; stage05 delta has no such contract. No rule added."}}


def worker(args):
    s5.s4.stage03.runtime()
    import torch
    import numpy as np
    import snn_data_lake
    from snn_data_lake.substrate.home import require_isolated_home
    require_isolated_home()
    require(Path(snn_data_lake.__file__).resolve().is_relative_to(Path(os.environ["STAGE06_SOURCE"])), "wrong source")
    require(Path(torch.__file__).resolve().is_relative_to(Path(os.environ["STAGE06_SITE"])), "wrong torch")
    require(Path(np.__file__).resolve().is_relative_to(Path(os.environ["STAGE06_SITE"])), "wrong numpy")
    torch.set_num_threads(1)
    payload = json.loads(Path(args.input).read_text())
    start,cpu = time.monotonic(),time.process_time()
    with contextlib.redirect_stdout(io.StringIO()):
        if args.worker == "controls": result = controls()
        elif args.worker == "duration": result = duration(payload)
        elif args.worker == "ll_base": result = ll_base(payload["seed"])
        elif args.worker == "prune": result = prune(payload)
        else: result = restore(payload)
    require(not sys._is_gil_enabled(), "GIL enabled")
    print(dumps({"result":result,"cost":{"wall_s":time.monotonic()-start,"cpu_s":time.process_time()-cpu,
                "maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                "runtime":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"gil":False,
                "source":snn_data_lake.__file__,"torch_file":torch.__file__,"numpy_file":np.__file__}}))


def run(args):
    out = HERE / args.run_id
    out.mkdir()
    require(subprocess.check_output(["git","-C",str(ROOT/"repo"),"rev-parse","HEAD"],text=True).strip()==HEAD,"wrong HEAD")
    require(not subprocess.check_output(["git","-C",str(ROOT/"repo"),"status","--porcelain"]),"dirty checkout")
    source = Path(args.source_root).resolve(); site = Path(args.runtime_site).resolve()
    manifest = {"profile":PROFILE,"head":HEAD,"status":"RUNNING","runs":[],"seeds":list(SEEDS),
                "source":str(source),"site":str(site),"sources":{},"input_receipts":{}}
    for p in [Path(__file__),HERE/"analyze.py",HERE/"protocol.md",OLD/"class_block.py",OLD/"evaluate.py",OLD/"class_block_analysis.py",
              HERE.parent/"04-artifacts/evaluate.py",HERE.parent/"03-artifacts/evaluate.py"]:
        manifest["sources"][str(p.relative_to(ROOT))] = c5.digest(p.read_bytes())
        dest = out/"provenance"/p.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
    start=time.monotonic()
    def job(key,kind,payload,home):
        require(time.monotonic()-start<1200,"block limit")
        home.mkdir(parents=True,exist_ok=True)
        inp=out/(key+".input.json"); save(inp,payload)
        cmd=[str(ROOT/".venv-04/bin/python"),"-B",str(Path(__file__)),"--worker",kind,"--input",str(inp)]
        env=os.environ.copy(); env.update(PYTHONPATH=str(source/"src")+os.pathsep+str(site),NORD_HOME=str(home),
            STAGE06_SOURCE=str(source),STAGE06_SITE=str(site),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONUTF8="1")
        t=time.monotonic()
        p=subprocess.run(cmd,cwd=source,env=env,capture_output=True,timeout=120)
        (out/(key+".json")).write_bytes(p.stdout); (out/(key+".stderr.txt")).write_bytes(p.stderr)
        manifest["runs"].append({"key":key,"kind":kind,"command":cmd,"home":str(home),"rc":p.returncode,"wall_s":time.monotonic()-t})
        save(out/"manifest.json",manifest)
        require(p.returncode==0,f"worker {key} failed: {p.stderr.decode()[-2000:]}")
        result=json.loads(p.stdout)["result"]
        print(key,"OK",flush=True)
        return result
    try:
        job("controls","controls",{},out/"homes/controls")
        for seed in SEEDS:
            old=OLD/"class-pilot-3"
            data=json.loads((old/f"inputs-{seed}.json").read_text())
            discovery=json.loads((old/f"discovery-{seed}.json").read_text())["result"]
            receipt=discovery["looks"]["30"]["receipt"]
            original=old/f"homes/discovery-{seed}"
            require(original.exists(),"missing source lake")
            manifest["input_receipts"][str(original/"brain_lake.bin")]=c5.digest((original/"brain_lake.bin").read_bytes())
            stimuli=json.loads((old/f"stimuli-{seed}.json").read_text())
            configs=[(n,False) for n in (0,1,2,5,10)]+[(10,True)]
            if seed==8000: configs += [(1,False)]
            for index,(passes,shuffle) in enumerate(configs):
                key=f"duration-{seed}-{passes}"+("-shuffle" if shuffle else "")+("-repeat" if index==6 else "")
                home=out/"homes"/key; shutil.copytree(original,home)
                result=job(key,"duration",{"data":data,"receipt":receipt,"passes":passes,"shuffle":shuffle},home)
                if passes in (0,10) and not shuffle:
                    oldarm="none" if passes==0 else "replay"
                    prior=json.loads((old/f"{oldarm}-{seed}.json").read_text())["result"]
                    require(result["after"]==prior["after"],"05 endpoint did not reproduce")
                restored=job("restore-"+key,"restore",{"receipt":result["receipt"],"stimuli":stimuli},home)
                require(restored["state"]==result["after"],"duration restart state mismatch")
                require(restored["answers"]=={sp:r["answers"] for sp,r in result["reads"].items()},"duration restart answers mismatch")
                require(restored["lake_hash"]==result["lake_hash"],"duration restart wrote lake")
            basehome=out/"homes"/f"ll-base-{seed}"
            base=job(f"ll-base-{seed}","ll_base",{"seed":seed},basehome)
            xs=[[x for x,_ in t["test"]] for t in base["tasks"]]
            for index,arm in enumerate(("off","magnitude","random","sabotage")+(("magnitude",) if seed==8000 else ())):
                key=f"prune-{seed}-{arm}"+("-repeat" if index==4 else "")
                home=out/"homes"/key; shutil.copytree(basehome,home)
                result=job(key,"prune",{"arm":arm,"receipt":base["receipt"],"xs":xs},home)
                restored=job("restore-"+key,"restore",{"receipt":result["receipt"],"xs":xs},home)
                require(hashed(restored["state"])==hashed(result["after"]),"prune restart state mismatch")
                require(restored["answers"]==result["read"]["answers"],"prune restart answers mismatch")
                require(restored["lake_hash"]==result["lake_hash"],"prune restart wrote lake")
        manifest["status"]="COMPLETE"
    except BaseException as exc:
        manifest["status"]="FAILED"; manifest["error"]=str(exc)
        raise
    finally:
        manifest["wall_s"]=time.monotonic()-start
        save(out/"manifest.json",manifest)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--worker",choices=("controls","duration","ll_base","prune","restore"))
    p.add_argument("--input")
    p.add_argument("--run-id")
    p.add_argument("--source-root")
    p.add_argument("--runtime-site")
    args=p.parse_args()
    worker(args) if args.worker else run(args)
