"""Disclosed engineering revision: exact root sampler vs wider pool, actual local port.

No Brain, lake, analytical trained weights or classifier lookup. LP certificate is
diagnostic only. A literal-growth arm is explicitly separated from necessity-gated claims.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from snn_data_lake.instruments import taught_problems as tp, inseparable_mapping as im
from snn_data_lake.instruments.taught_campaign import Reader
from snn_data_lake.growth import capacity_certificate as cc
from snn_data_lake.growth.candidate_lifecycle import Pool, contribution, select_block
from snn_data_lake.growth.grown_code import GrownUnit, segment_psp, grown_version
from snn_data_lake.growth.growth_loop import trace_train, label_probabilities
from snn_data_lake.readout.posed_problem import unit_rows, units_from_rows, chat_grown_drives

ROOT = Path(__file__).resolve().parent
SEED = 9900
RECIPES = {
    "curriculum18": {"pool": 1024, "attempts": 3, "literal_growth": False, "eligible": True, "batch_size": 18},
    "curriculum27": {"pool": 1024, "attempts": 3, "literal_growth": False, "eligible": True, "batch_size": 27},
}


def main():
    ap = argparse.ArgumentParser();ap.add_argument("recipe", choices=RECIPES);args=ap.parse_args()
    cfg = RECIPES[args.recipe]
    started = time.monotonic()
    out = ROOT / "candidate-probes-v5";out.mkdir(exist_ok=True)
    p = tp.PROBLEMS["ii.xor"]
    learner = tp.canonical_incumbent(SEED);reader=Reader(learner,history=False)
    old_problem=tp.PROBLEMS["aii"]
    old_validation=list(old_problem.episodes(SEED,"validation"))
    old_texts=[reader.text(e) for e in old_validation];old_targets=[e[-1].label for e in old_validation]
    old_history=[];old_streak=0
    for old_epoch,old_episodes in enumerate(old_problem.teach_epochs(SEED,1200),1):
        for episode in old_episodes:reader.teach(episode,episode[-1].label)
        old_predictions=[learner.route_text(t) for t in old_texts]
        old_correct=sum(x==y for x,y in zip(old_predictions,old_targets))
        old_streak=old_streak+1 if old_correct==len(old_targets) else 0
        old_history.append({"epoch":old_epoch,"correct":old_correct,"n":len(old_targets),"streak":old_streak})
        if old_epoch in (100,400,800,1200):
            diagnostic={"epoch":old_epoch,"predictions":old_predictions,"targets":old_targets,
                        "texts":old_texts,"unique_ranks":{t:im.ranks(t) for t in old_texts},
                        "responses":learner.responses,"weights":learner.weights(),
                        "drives":{t:[learner._drive(n,im.ranks(t)) for n in learner.neurons] for t in old_texts},
                        "snapshot":learner._snapshot()}
            (out/f"{args.recipe}-A-{old_epoch}-state.json").write_text(json.dumps(diagnostic,indent=2)+'\n')
        if old_streak==10:break
    if old_streak!=10:
        failure={"recipe":args.recipe,"scope":"exposed engineering fixture","status":"OLD_A_NOT_STABLE_AFTER_3_ATTEMPTS", "old_history":old_history}
        (out/f"{args.recipe}.json").write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps(failure),flush=True);return
    old_weights_before_b=learner.weights()
    construction = list(next(iter(p.teach_epochs(SEED,1))))
    texts = [reader.text(e) for e in construction];labels=[e[-1].label for e in construction]
    codes = [im.ranks(t) for t in texts]
    validation = list(p.episodes(SEED,"validation"))
    valtexts=[reader.text(e) for e in validation];targets=[e[-1].label for e in validation]
    schedule=iter(p.teach_epochs(SEED,400*cfg["attempts"]))
    result={"scope":"single exposed engineering fixture, not G9 benefit data", "recipe":args.recipe,
            "config":cfg,"seed":SEED,"batch_size":cfg["batch_size"],"window":12,"t_post_g":15.,
            "input_sigma":1.,"threshold_range":[.5,2.],"trace_epochs":20,"trace_eta":.05,
            "feedback_rule":"9000 + (stream_seed-9000+candidate_i)%1000",
            "stream_rule":"9000 + (9900-9000+batch_id)%1000",
            "selection":"actual candidate_lifecycle.select_block, float64 scores, all passes False",
            "source_rows":"exact first construction epoch,4 rows, not16 repeated validation rows",
            "output_init":"zero, 8E+1I byglobal IDmod9; only existing local R-STDP deposits",
            "attempt_epoch_cap":400,"stable_epochs":10,"growth_log":[],"history":[],
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["old_pretraining"]={"problem":"aii","stable_solved":True,"epochs":len(old_history),"exposures":4*len(old_history),"history":old_history}
    result["retention_endpoints"]=[]
    current=[];streak=0;first_endpoint=None;first_all=None;batch_id=0

    def read_old():
        active=learner._g9_context;learner._g9_context=None
        before=json.dumps(learner._snapshot(),sort_keys=True)
        predictions=[learner.route_text(t) for t in old_texts]
        assert json.dumps(learner._snapshot(),sort_keys=True)==before
        learner._g9_context=active
        return predictions

    def drives(text):
        code=im.ranks(text)
        frozen=[learner._drive(n,code) for n in learner.neurons]
        grown=chat_grown_drives(learner,code)
        return [a+b for a,b in zip(frozen,grown,strict=True)]

    def context(units):
        learner._g9_context={"problem_id":p.id,"units":unit_rows(units),"afferents":[[u.unit_id,[]] for u in units],
                             "t_post_g":15.,"out_eligible":[[cfg["eligible"]]*len(learner.responses) for _ in units]}

    def draw(count):
        nonlocal batch_id
        begin=time.monotonic();stream=9000+(SEED-9000+batch_id)%1000;rng=random.Random(stream)
        residual=[];items=[]
        for code,text,label in zip(codes,texts,labels,strict=True):
            probabilities=label_probabilities(drives(text));onehot=[float(x==label) for x in learner.responses]
            residual.append([y-pr for y,pr in zip(onehot,probabilities)]);items.append((code,onehot,probabilities))
        candidates=[];scores=[];features=[]
        for i in range(cfg["pool"]):
            w=tuple(rng.gauss(0.,1.) for _ in range(im.N_IN));threshold=rng.uniform(.5,2.)
            w=trace_train(w,threshold,12,learner.tau,items,.05,20,9000+(stream-9000+i)%1000)
            u=GrownUnit(i,w,threshold,12,1,(0.,)*len(learner.responses))
            feature=[segment_psp((u,),code,learner.tau,15.)[0] for code in codes]
            candidates.append(u);features.append(feature);scores.append(contribution(feature,residual))
        pool=Pool(torch.tensor([u.w_in for u in candidates],dtype=torch.float64),
                  torch.tensor([u.threshold for u in candidates],dtype=torch.float64),
                  torch.tensor(scores,dtype=torch.float64),torch.zeros(len(candidates),dtype=torch.bool),
                  torch.zeros(len(candidates),dtype=torch.bool))
        chosen=select_block(pool,count,count,len(current));nextid=max((u.unit_id for u in current),default=-1)+1
        selected=[]
        if len(chosen)==count:
            for j,i in enumerate(chosen):
                old=candidates[i];uid=nextid+j
                selected.append(GrownUnit(uid,old.w_in,old.threshold,old.window,-1 if uid%9==8 else 1,old.out))
        log={"batch_id":batch_id,"stream_seed":stream,"source_indices":chosen,"contributing":sum(x>0 for x in scores),
             "selected_scores":[scores[i] for i in chosen],"selected_features":[features[i] for i in chosen],
             "units":unit_rows(selected),"candidate_evaluations":len(candidates),"seconds":time.monotonic()-begin}
        result["growth_log"].append(log);batch_id+=1;return selected

    for attempt in range(cfg["attempts"]):
        if cfg["pool"] and (attempt==0 or cfg["literal_growth"]):
            if learner._g9_context is not None:
                current=list(units_from_rows(learner._g9_context["units"],len(learner.responses)))
            current.extend(draw(cfg["batch_size"]))
            context(current)
            diagnosis=cc.diagnose(learner,texts,labels,units=current)
            result["growth_log"][-1]["diagnosis"]={"kind":diagnosis.kind,"log":diagnosis.log}
            streak=0
        for local_epoch in range(1,401):
            episodes=next(schedule)
            for episode in episodes:reader.teach(episode,episode[-1].label)
            predictions=[learner.route_text(t) for t in valtexts]
            correct=sum(x==y for x,y in zip(predictions,targets));streak=streak+1 if correct==len(targets) else 0
            margins=[]
            for text,label in zip(texts,labels):
                d=drives(text);k=learner.responses.index(label);margins.append(d[k]-max(x for j,x in enumerate(d) if j!=k))
            row={"attempt":attempt,"local_epoch":local_epoch,"total_epoch":len(result["history"])+1,
                 "correct":correct,"n":len(targets),"streak":streak,"margins":margins}
            result["history"].append(row)
            if correct==len(targets) and first_endpoint is None:
                saved=learner._snapshot();before=json.dumps(saved,sort_keys=True)
                reads=[[learner.route_text(t) for t in valtexts] for _ in range(10)]
                assert all(v==predictions for v in reads)
                learner._restore_snapshot(saved)
                assert [learner.route_text(t) for t in valtexts]==predictions
                assert json.dumps(learner._snapshot(),sort_keys=True)==before
                first_endpoint={"total_epoch":row["total_epoch"],"margins":margins,
                                "ten_immutable_reads_exact":True,"snapshot_restore_exact":True}
                first_all=row["total_epoch"]
            if streak==10:break
        old_predictions=read_old()
        result["retention_endpoints"].append({"attempt":attempt,"B_epochs":len(result["history"]),"old_correct":sum(x==y for x,y in zip(old_predictions,old_targets)),"old_n":len(old_targets),"old_predictions":old_predictions,"old_read_state_unchanged":True})
        if streak==10:break
    predictions=[learner.route_text(t) for t in valtexts]
    active=learner._g9_context
    learner._g9_context=None;masked=[learner.route_text(t) for t in valtexts];learner._g9_context=active
    snapshot=learner._snapshot();learner._authority.clear()
    assert [learner.route_text(t) for t in valtexts]==predictions;learner._restore_snapshot(snapshot)
    result.update(stable_solved=streak==10,epochs=len(result["history"]),train_exposures=4*len(result["history"]),
                  first_all_correct_epoch=first_all,first_endpoint=first_endpoint,predictions=predictions,targets=targets,
                  final_correct=sum(x==y for x,y in zip(predictions,targets)),masked_correct=sum(x==y for x,y in zip(masked,targets)),
                  active_units=0 if active is None else len(active["units"]),wall_seconds=time.monotonic()-started,
                  authority_clear_preserves_predictions=True)
    if active is not None:result["final_units"]=active["units"]
    old_final=read_old()
    result["old_final_correct"]=sum(x==y for x,y in zip(old_final,old_targets))
    result["old_validation_n"]=len(old_targets)
    result["old_frozen_weight_max_change_during_B"]=max(abs(a-b) for rowa,rowb in zip(old_weights_before_b,learner.weights()) for a,b in zip(rowa,rowb))
    (out/f"{args.recipe}.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ("history","growth_log","final_units","predictions","targets")}),flush=True)


if __name__=="__main__":main()
