"""Recompute stage06 from records only. No SNDL, Torch, or learning imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"05-artifacts/class-pilot-3"
SEEDS=(8000,8001,8002,8003)


def sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def check(ok,why):
    if not ok: raise ValueError(why)


def score(answers,targets):
    check(len(answers)==len(targets) and bool(targets),"answer count")
    return {"correct":sum(a==b for a,b in zip(answers,targets)),"n":len(targets)}


def duration_scores(answers,rows):
    out={}
    for name,label in (("A","GREET"),("B","GOAL")):
        indices=[i for i,r in enumerate(rows) if r["target"]==label]
        out[name]=score([answers[i] for i in indices],[rows[i]["target"] for i in indices])
    return out


def delta_predictions(state,rows):
    beta=state["readout"]["beta"];labels=state["readout"]["labels"]
    out=[]
    for row in rows:
        check(len(row["x"])==len(beta),"beta width")
        values=[sum(x*w[j] for x,w in zip(row["x"],beta)) for j in range(len(labels))]
        out.append(labels[max(range(len(labels)),key=values.__getitem__)])
    return out


def lake_body(path,expected):
    raw=path.read_bytes();check(raw[:8]==b"SNDLBIN1","lake magic")
    pos=8; blobs=[]
    while pos<len(raw):
        check(pos+4<=len(raw),"torn frame")
        size=struct.unpack_from("<I",raw,pos)[0];pos+=4
        payload=raw[pos:pos+size];pos+=size
        check(len(payload)==size and payload[:1]==b"\x07","lake frame")
        n=payload[1];name=payload[2:2+n].decode("ascii")
        check(len(payload)>=2+n+4,"named frame length")
        # home.py::_parse_state_record: u32 is the sequence, never a blob length.
        _sequence=struct.unpack_from("<I",payload,2+n)[0];blob=payload[2+n+4:]
        if name=="stage06-result":blobs.append(blob)
    check(bool(blobs),"missing named state")
    check(hashlib.sha256(blobs[-1]).hexdigest()==expected,"latest lake receipt")
    env=json.loads(blobs[-1]);check(sha(env["body"])==env["sha256"],"lake checksum")
    return raw,env["body"]


def cosine(a,b):
    denom=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/denom if denom else 0.


def ll_predictions(snapshot,xs):
    state=snapshot["state"]; ps=state["learner"]["prototypes"]
    return [[state["labels"][max(range(len(ps)),key=lambda i:cosine(x,ps[i]))] for x in group] for group in xs]


def validate_duration(result,restored,data,passes,shuffle,old):
    check(result["before"]==old,"input b30 state")
    check(len(data["train"])==32 and all(sorted(o)==list(range(32)) for o in data["orders"]),"epoch membership")
    sets=[{json.dumps(r["texts"]) for r in data[s]} for s in ("train","dev","final")]
    check([len(s) for s in sets]==[32,8,12] and all(not sets[i]&sets[j] for i in range(3) for j in range(i)),"split overlap")
    check(sorted(map(sha,old["replay"]))==sorted(map(sha,data["train"])),"train-only lake replay")
    check(result["after"]["replay"]==old["replay"],"replay memory changed")
    check(result["work"]["presentations"]==32*passes,"duration budget")
    check(result["after"]["readout"]["rule"]["calls"]==old["readout"]["rule"]["calls"]+32*passes,"update counter")
    rows=[data["train"][i] for order in data["orders"][:passes] for i in order]
    check(result["work"]["rows_hash"]==sha(rows),"training rows")
    labels=result["work"]["labels"]
    check(len(labels)==len(rows),"training labels count")
    check(sorted(labels)==sorted(r["target"] for r in rows),"shuffle marginal")
    if not shuffle: check(labels==[r["target"] for r in rows],"training labels")
    check(restored["state"]==result["after"],"restart state")
    check(restored["answers"]=={k:v["answers"] for k,v in result["reads"].items()},"restart answers")
    check(restored["lake_hash"]==result["lake_hash"],"restart lake")
    for split in ("dev","final"):
        reading=result["reads"][split]
        check(reading["answers"]==delta_predictions(result["after"],data[split]),"independent beta answers")
        check(reading["state_sha256"]==sha(result["after"]),"read state receipt")
        check(reading["lake_sha256"]==result["lake_hash"],"read lake receipt")
        scored=duration_scores(reading["answers"],data[split])
        for g in ("A","B"):
            check(scored[g]=={k:result["scores"][split][g][k] for k in ("correct","n")},"scorer mismatch")
    return duration_scores(result["reads"]["final"]["answers"],data["final"])


def validate_prune(result,restored,base,arm):
    before=base["snapshot"]; state=before["state"]
    check(result["before"]==before,"prune input")
    n=len(state["labels"]); budget=before["config"]["budget"]
    kept=result["kept"]
    target=n if arm=="off" else min(n,budget)
    check(len(kept)==target and len(set(kept))==target,"mask cardinality")
    if arm=="magnitude":
        victims=set(sorted(range(n),key=lambda i:(state["utility"][i],i))[:n-target])
        check(kept==[i for i in range(n) if i not in victims],"utility ranking")
    if arm=="sabotage":
        victims=set(sorted(range(n),key=lambda i:(-state["utility"][i],i))[:n-target])
        check(kept==[i for i in range(n) if i not in victims],"sabotage ranking")
    after=result["after"]
    for key in ("labels","utility"):
        check(after["state"][key]==[state[key][i] for i in kept],"unit alignment")
    check(after["state"]["learner"]["prototypes"]==[state["learner"]["prototypes"][i] for i in kept],"prototype alignment")
    for key in ("memory_hash","memory_i","memory_rng","memory_capacity"):
        check(before[key]==after[key],"memory changed")
    xs=[[x for x,_ in t["test"]] for t in base["tasks"]]
    answers=ll_predictions(after,xs)
    check(answers==result["read"]["answers"],"independent classifier mismatch")
    check(sha(after)==result["read"]["state_hash"],"state receipt")
    check(restored["state"]==after and restored["answers"]==answers,"prune restart")
    check(restored["lake_hash"]==result["lake_hash"],"prune restart lake")
    check(result["selection"]["removed"]==n-target,"removal count")
    if arm!="sabotage":
        check(result["selection"]["native_calls"]==1,"native call count")
        check(result["selection"]["native_removals"]==n-target,"native removal counter")
    queries=sum(map(len,xs))
    check(result["work"]["cosines"]==queries*target,"cosine work")
    check(result["work"]["coordinates"]==queries*target*36,"coordinate work")
    return {name:score(got,[y for _,y in t["test"]]) for name,got,t in zip(("A","B","C"),answers,base["tasks"])}


def analyze(records):
    load=lambda key:json.loads((records/(key+".json")).read_text())["result"]
    manifest=json.loads((records/"manifest.json").read_text())
    check(manifest["status"]=="COMPLETE","incomplete manifest")
    expected={"controls"}
    durations=[]; pruners=[]; pairs=0
    for seed in SEEDS:
        data=json.loads((OLD/f"inputs-{seed}.json").read_text())
        old=json.loads((OLD/f"discovery-{seed}.json").read_text())["result"]["looks"]["30"]["state"]
        for passes,shuffle in [(n,False) for n in (0,1,2,5,10)]+[(10,True)]:
            key=f"duration-{seed}-{passes}"+("-shuffle" if shuffle else "")
            expected.update((key,"restore-"+key)); pairs+=1
            r=load(key)
            scores=validate_duration(r,load("restore-"+key),data,passes,shuffle,old)
            home=records/"homes"/key/"brain_lake.bin"
            raw,body=lake_body(home,r["receipt"])
            check(body["state"]==r["after"],"duration lake state")
            check(hashlib.sha256(raw).hexdigest()==r["lake_hash"],"duration lake bytes")
            original=(OLD/f"homes/discovery-{seed}/brain_lake.bin").read_bytes()
            check(raw.startswith(original),"old lake history changed")
            if passes in (0,10) and not shuffle:
                prior=json.loads((OLD/f"{'none' if passes==0 else 'replay'}-{seed}.json").read_text())["result"]
                check(r["after"]==prior["after"],"05 endpoint mismatch")
            durations.append({"seed":seed,"passes":passes,"shuffle":shuffle,"scores":scores,"work":r["work"]})
        base=load(f"ll-base-{seed}");expected.add(f"ll-base-{seed}")
        for arm in ("off","magnitude","random","sabotage"):
            key=f"prune-{seed}-{arm}";expected.update((key,"restore-"+key)); pairs+=1
            r=load(key);scores=validate_prune(r,load("restore-"+key),base,arm)
            raw,body=lake_body(records/"homes"/key/"brain_lake.bin",r["receipt"])
            check(body["snapshot"]==r["after"],"prune lake structure")
            check(hashlib.sha256(raw).hexdigest()==r["lake_hash"],"prune lake bytes")
            check(raw.startswith((records/f"homes/ll-base-{seed}/brain_lake.bin").read_bytes()),"prune removed lake records")
            t=r["work"]["timings"]
            pruners.append({"seed":seed,"arm":arm,"scores":scores,"units":len(r["after"]["state"]["labels"]),
               "kept":r["kept"],"selection":r["selection"],"coordinates":r["work"]["coordinates"],
               "median_cpu_s_per_query":statistics.median(x["cpu_s"]/x["queries"] for x in t),
               "median_wall_s_per_query":statistics.median(x["wall_s"]/x["queries"] for x in t)})
    for key,kind in (("duration-8000-1","duration"),("prune-8000-magnitude","prune")):
        a,b=load(key),load(key+"-repeat")
        check(a["after"]==b["after"],"repeat state mismatch")
        field="reads" if kind=="duration" else "read"
        check(a[field]==b[field],"repeat reading mismatch")
        restored=load("restore-"+key+"-repeat")
        check(restored["state"]==b["after"],"repeat restart mismatch")
        expected.update((key+"-repeat","restore-"+key+"-repeat"));pairs+=1
    check(expected=={r["key"] for r in manifest["runs"]},"missing or unexpected worker")
    check(len(expected)==len(manifest["runs"]),"duplicate worker")
    check(all(r["rc"]==0 for r in manifest["runs"]),"failed worker")
    controls=load("controls");check(all(controls["checks"].values()),"controls failed")
    costs=[json.loads((records/(r["key"]+".json")).read_text())["cost"] for r in manifest["runs"]]
    summary={"status":"VALIDATED","duration":durations,"pruning":pruners,"restore_pairs":pairs,
        "stages":controls["stages"],"cost":{"block_wall_s":manifest["wall_s"],"workers":len(costs),
        "worker_wall_s":sum(r["wall_s"] for r in manifest["runs"]),"function_cpu_s":sum(c["cpu_s"] for c in costs),
        "max_worker_rss_bytes":max(c["maxrss_bytes"] for c in costs),
        "lake_sizes_bytes":{str(p.relative_to(records)):p.stat().st_size for p in sorted((records/"homes").glob("*/brain_lake.bin"))}}}
    return summary


def sensitivity(records):
    def load(k):return json.loads((records/(k+".json")).read_text())["result"]
    base=load("ll-base-8000");r=load("prune-8000-magnitude");restored=load("restore-prune-8000-magnitude")
    validate_prune(r,restored,base,"magnitude")
    faults={}
    def fault(name,change):
        a,b=copy.deepcopy(r),copy.deepcopy(restored);change(a,b)
        try:validate_prune(a,b,base,"magnitude")
        except (ValueError,IndexError,KeyError):faults[name]=True
        else:faults[name]=False
    fault("mask",lambda a,b:a["kept"].pop())
    fault("answers",lambda a,b:a["read"]["answers"][0].__setitem__(0,-1))
    fault("work",lambda a,b:a["work"].__setitem__("coordinates",0))
    fault("memory",lambda a,b:a["after"].__setitem__("memory_hash","bad"))
    fault("restart",lambda a,b:b["state"]["state"]["labels"].__setitem__(0,-1))
    fault("counter",lambda a,b:a["selection"].__setitem__("native_removals",-1))
    data=json.loads((OLD/"inputs-8000.json").read_text())
    old=json.loads((OLD/"discovery-8000.json").read_text())["result"]["looks"]["30"]["state"]
    dr=load("duration-8000-1");ds=load("restore-duration-8000-1")
    validate_duration(dr,ds,data,1,False,old)
    def dfault(name,change):
        a,b=copy.deepcopy(dr),copy.deepcopy(ds);change(a,b)
        try:validate_duration(a,b,data,1,False,old)
        except (ValueError,IndexError,KeyError):faults[name]=True
        else:faults[name]=False
    dfault("duration_budget",lambda a,b:a["work"].__setitem__("presentations",0))
    dfault("duration_answer",lambda a,b:a["reads"]["final"]["answers"].__setitem__(0,"wrong"))
    dfault("duration_restart",lambda a,b:b["state"]["readout"]["beta"][0].__setitem__(0,99.))
    dfault("duration_replay",lambda a,b:a["after"]["replay"].pop())
    check(all(faults.values()),"analysis fault not caught")
    return {"positive":True,"faults_caught":faults}


def main():
    p=argparse.ArgumentParser();p.add_argument("--records",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    summary=analyze(a.records); faults=sensitivity(a.records)
    a.output.mkdir()
    for name,obj in (("summary",summary),("sensitivity",faults)):
        (a.output/(name+".json")).write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    lines=["| Seed | Replay passes | Shuffle | A final | B final |","|---|---:|---|---:|---:|"]
    for row in summary["duration"]:
        s=row["scores"];lines.append(f'| {row["seed"]} | {row["passes"]} | {row["shuffle"]} | {s["A"]["correct"]}/{s["A"]["n"]} | {s["B"]["correct"]}/{s["B"]["n"]} |')
    lines += ["","| Seed | Pruner | Units | A | B | C | Coordinates/query |","|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["pruning"]:
        s=row["scores"];vals=[f'{s[g]["correct"]}/{s[g]["n"]}' for g in ("A","B","C")]
        lines.append(f'| {row["seed"]} | {row["arm"]} | {row["units"]} | '+" | ".join(vals)+f' | {row["coordinates"]//120} |')
    (a.output/"table.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":summary["status"],"restore_pairs":summary["restore_pairs"],"cost":summary["cost"],"faults":faults},sort_keys=True))


if __name__=="__main__": main()
