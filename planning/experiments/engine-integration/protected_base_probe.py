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
from snn_data_lake.readout.posed_problem import unit_rows, units_from_rows, chat_grown_drives, chat_grown_deposit
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

ROOT = Path(__file__).resolve().parent
SEED = 9900
RECIPES = {
    "protected_base": {"pool": 1024, "attempts": 3, "literal_growth": False, "eligible": True, "batch_size": 18, "rehearsal": False},
}
PHASE = "A"
COUNTS = {key: {"presentations": 0, "teaching_calls": 0, "shared_deposits": 0} for key in ("A", "B", "rehearsal")}
ORIGINAL_TEACH = Reader.teach
ORIGINAL_DEPOSIT = SpikingInteractionLearner._deposit
CORE_REFERENCE = None
CORE_CHECKS = 0
GROWN_DEPOSIT_CALLS = 0


def core_state(unit):
    state=unit.state_dict()
    del state["tree"]["attrs"]["_g9_context"]
    return json.dumps(state,sort_keys=True)


def protected_deposit(unit, li, code, reward):
    global GROWN_DEPOSIT_CALLS
    if PHASE == "B" and unit._g9_context is not None:
        mask = unit._g9_neuron_mask
        if mask is not None:
            assert type(mask) is list and len(mask)==len(unit.neurons) and all(type(v) is bool for v in mask)
            if not mask[li]:return [0.0]*unit.n_in
        # Existing grown local R-STDP only; explicitly no PRE gain/core mutation.
        chat_grown_deposit(unit,li,code,float(reward))
        GROWN_DEPOSIT_CALLS += 1
        return [0.0]*unit.n_in
    return ORIGINAL_DEPOSIT(unit,li,code,reward)


def corrective_teach(reader, turns, label):
    row = COUNTS[PHASE]; row["presentations"] += 1
    if reader.unit.route_text(reader.text(turns)) == label:
        return label
    before = reader.unit.stack._global_t
    answer = ORIGINAL_TEACH(reader, turns, label)
    row["teaching_calls"] += 1; row["shared_deposits"] += reader.unit.stack._global_t - before
    if PHASE=="B":
        global CORE_CHECKS
        assert core_state(reader.unit)==CORE_REFERENCE
        CORE_CHECKS += 1
    return answer



def main():
    ap = argparse.ArgumentParser();ap.add_argument("recipe", choices=RECIPES);args=ap.parse_args()
    global PHASE, CORE_REFERENCE
    cfg = RECIPES[args.recipe]
    Reader.teach = corrective_teach
    SpikingInteractionLearner._deposit = protected_deposit
    started = time.monotonic()
    out = ROOT / "protected-base-probes";out.mkdir(exist_ok=True)
    p = tp.PROBLEMS["ii.xor"]
    learner = tp.canonical_incumbent(SEED);reader=Reader(learner,history=False)
    old_problem=tp.PROBLEMS["aii"]
    old_validation=list(old_problem.episodes(SEED,"validation"))
    old_texts=[reader.text(e) for e in old_validation];old_targets=[e[-1].label for e in old_validation]
    variant = {"scope": "prospective exposed engineering teaching/rehearsal policy control, not full Engine efficacy",
               "seed": SEED, "mode": args.recipe, "feedback": "pure currentprediction wrong => unchangedcanonicallearn_turn; correct=>no deposit",
               "initialization": "working-source PREbaselinefix, allfacultiesenabled",
               "A": "aii stable10 beforeB, max1200 epochs",
               "B": "ii.xor, oneinitial18 pool1024 exactsource stream9900/uniqueB, zerooutputs, max1200epochs",
               "rehearsal": "onecontinuingAteach-epoch(4presentations) aftereachBepoch, Bcontextoff; wrongonly" if cfg["rehearsal"] else "none",
               "endpoint": "BANDoldAvalidationallcorrect10ends, validationpure, nofuture",
               "eligibility_policy": "Bonly: base52neurons/fullstackstatefrozen; existinggrownoutlocaldeposits receive rawreward; wholemaskFalse stillrefuses",
               "core_equality": "fulllearnerstate_dict excludingonly mutablegrowncontext; cache isexcludedbyincumbentcheckpointcontract",
               "comparison": "previousallplasticno-rehearsalnegative retained, notrerun",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/f"{args.recipe}-recipe.json").write_text(json.dumps(variant,indent=2)+'\n')
    old_history=[];old_streak=0
    old_schedule=iter(old_problem.teach_epochs(SEED,2400))
    for old_epoch,old_episodes in enumerate(old_schedule,1):
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
        if old_streak==10 or old_epoch==1200:break
    if old_streak!=10:
        failure={"recipe":args.recipe,"scope":"exposed engineering fixture","status":"OLD_A_NOT_STABLE_AFTER_3_ATTEMPTS", "old_history":old_history}
        (out/f"{args.recipe}.json").write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps({k:v for k,v in failure.items() if k != "old_history"}),flush=True);return
    old_weights_before_b=learner.weights()
    CORE_REFERENCE=core_state(learner)
    PHASE="B"
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
    result["variant"] = variant
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
            if cfg["rehearsal"]:
                active_b=learner._g9_context;learner._g9_context=None;PHASE="rehearsal"
                for old_episode in next(old_schedule):reader.teach(old_episode,old_episode[-1].label)
                learner._g9_context=active_b;PHASE="B"
            predictions=[learner.route_text(t) for t in valtexts]
            old_predictions=read_old()
            old_correct=sum(x==y for x,y in zip(old_predictions,old_targets))
            correct=sum(x==y for x,y in zip(predictions,targets))
            joint=correct==len(targets) and old_correct==len(old_targets)
            streak=streak+1 if joint else 0
            margins=[]
            for text,label in zip(texts,labels):
                d=drives(text);k=learner.responses.index(label);margins.append(d[k]-max(x for j,x in enumerate(d) if j!=k))
            row={"attempt":attempt,"local_epoch":local_epoch,"total_epoch":len(result["history"])+1,
                 "correct":correct,"n":len(targets),"streak":streak,"old_correct":old_correct,"old_n":len(old_targets),"margins":margins,"counts":json.loads(json.dumps(COUNTS))}
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
    result["counts"] = COUNTS
    assert core_state(learner)==CORE_REFERENCE
    result["core_state_exact_Aendpoint_after_each_actual_Bteach_and_final"] = True
    result["core_equality_checks"] = CORE_CHECKS+1
    result["core_state_sha256"] = hashlib.sha256(CORE_REFERENCE.encode()).hexdigest()
    result["grown_deposit_calls"] = GROWN_DEPOSIT_CALLS
    result["old_frozen_weight_max_change_during_B"]=max(abs(a-b) for rowa,rowb in zip(old_weights_before_b,learner.weights()) for a,b in zip(rowa,rowb))
    (out/f"{args.recipe}.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ("history","growth_log","final_units","predictions","targets")}),flush=True)


if __name__=="__main__":main()
