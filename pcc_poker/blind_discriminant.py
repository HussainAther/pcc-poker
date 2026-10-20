"""Prospective label-sealed cross-family incremental prediction, version 1.

Run with --output NEW_DIRECTORY. No prior artifacts are overwritten.
The creator knows the generators; only computational identity access is blinded.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from .engine import equity, apply_action, utility
from .leduc_cfr import LeducCFR, regret_matching
from .policies import OpponentModel, component_scores, _softmax
from .value_floor import UniformContinuationValueModel

ACTIONS = ('check', 'bet', 'fold', 'call', 'raise')
ENDPOINTS = ('value_vs_tight', 'value_vs_loose', 'heldout_card_response')
CONFIG = dict(version=1, replicates=32, cfr_iterations=80, sampled_deals=6,
              ridge_penalty=10., bootstrap=3999, inference_seed=918232,
              practical_benefit_margin=.01, alpha=.05,
              family_size=12, split_salt='beyond-strategy-panel-v1',
              null_superiority='mean normalized MSE baseline minus augmented <= 0',
              null_no_benefit='mean normalized MSE baseline minus augmented >= 0.01',
              alternative_no_benefit='benefit smaller than 1% of training outcome variance',
              inference='paired two-population policy bootstrap, centered null, Holm over all 12 tests')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ('blind_discriminant.py', 'leduc_cfr.py', 'engine.py', 'policies.py',
             'value_floor.py', 'behavioral.py', 'blind_leduc_reference.py')}


def check_freeze(frozen):
    if frozen['sources'] != source_hashes() or frozen['config'] != CONFIG:
        raise RuntimeError('measurement pipeline changed after freeze')


def event(directory, stage, **kwargs):
    path = directory / 'events.jsonl'
    lines = path.read_text().splitlines() if path.exists() else []
    row = dict(sequence=len(lines), utc=datetime.now(timezone.utc).isoformat(),
               stage=stage, previous_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest() if lines else None,
               **kwargs)
    with path.open('a') as f:
        f.write(json.dumps(row, sort_keys=True)+'\n')


def reveal(directory, frozen):
    check_freeze(frozen)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    required=('opaque_policies_generated','measurement_frozen','predictions_frozen')
    stages=[e['stage'] for e in events]
    if any(stage not in stages for stage in required):
        raise RuntimeError('reveal attempted before frozen predictions')
    if [stages.index(stage) for stage in required] != sorted(stages.index(stage) for stage in required):
        raise RuntimeError('invalid freeze order')
    files={'opaque_policies_generated':('sealed_identity.json','identity_commitment'),
           'measurement_frozen':('measurements.json','measurements_sha256'),
           'predictions_frozen':('predictions_frozen.json','predictions_sha256')}
    for stage,(name,field) in files.items():
        row=events[stages.index(stage)]
        if digest(directory/name)!=row[field]:
            raise RuntimeError('frozen artifact changed before reveal')
    return json.loads((directory/'sealed_identity.json').read_text())


def panel(key):
    # Group by public history, excluding seat and cards: matched private-card
    # comparisons stay together, and feature/outcome information sets are disjoint.
    history = key[1].split('|')[-1]
    return int(hashlib.sha256((CONFIG['split_salt']+history).encode()).hexdigest()[:8], 16) % 2


def conventional_policy(solver, seed=None, parameters=None):
    """Equity/pot-odds logit family. No PCC components or latent PCC weights."""
    rng = np.random.default_rng(seed)
    if parameters is None:
        parameters = (rng.uniform(1, 7), rng.uniform(-2, 2), rng.uniform(.3, 1.8), rng.uniform(-.8, .8))
    strength, bias, temperature, history_slope = parameters
    result = {}
    for k, ids in solver.infos.items():
        node = solver.nodes[ids[0]]; s = node.state
        edge = equity(s, s.actor) - s.to_call / (s.pot+s.to_call)
        history = sum(a in ('bet', 'raise') for a in s.history)
        scores = {}
        for a in node.actions:
            scores[a] = (strength*(2*edge-1)+bias+history_slope*history if a in ('bet', 'raise') else
                         strength*edge if a in ('call', 'check') else -strength*edge)
        result[k] = np.array(list(_softmax(scores, temperature).values()))
    return result


def sampled_cfr(solver, seed):
    """Chance-sampled regret matching with unbiased sampled chance weights."""
    rng = np.random.default_rng(seed)
    regrets = {k: np.zeros(len(solver.nodes[v[0]].actions)) for k, v in solver.infos.items()}
    sums = {k: np.zeros_like(v) for k, v in regrets.items()}
    for _ in range(CONFIG['cfr_iterations']):
        strategy = {k: regret_matching(v) for k, v in regrets.items()}
        inc = {k: np.zeros_like(v) for k, v in regrets.items()}
        def walk(i, r0, r1):
            n = solver.nodes[i]
            if n.key is None:
                return utility(n.state, 0)
            p = strategy[n.key]
            if n.key[0] == 0:
                vals = np.array([walk(c, r0*x, r1) for c, x in zip(n.children, p)])
                v = float(p @ vals)
                inc[n.key] += r1*(vals-v)/CONFIG['sampled_deals']
                sums[n.key] += r0*p/CONFIG['sampled_deals']
            else:
                vals = np.array([walk(c, r0, r1*x) for c, x in zip(n.children, p)])
                v = float(p @ vals)
                inc[n.key] += r0*(v-vals)/CONFIG['sampled_deals']
                sums[n.key] += r1*p/CONFIG['sampled_deals']
            return v
        for root in rng.choice(solver.roots, CONFIG['sampled_deals'], replace=True):
            walk(int(root), 1., 1.)
        for k in regrets:
            regrets[k] += inc[k]
    return {k: v/v.sum() if v.sum() else np.full(len(v), 1/len(v)) for k, v in sums.items()}


def validate_policy(solver, policy):
    if set(policy) != set(solver.infos):
        raise ValueError('incomplete policy')
    for k, p in policy.items():
        if len(p) != len(solver.nodes[solver.infos[k][0]].actions) or not np.isfinite(p).all() or min(p)<0 or not np.isclose(sum(p), 1):
            raise ValueError('invalid behavioral probabilities')


def fixed_chaos(solver):
    # Immutable behavioral snapshot of the source score Chaos: empty frozen
    # opponent and own-action histories; no decide() calls or RNG transitions.
    result = {}
    for k, ids in solver.infos.items():
        def probs(s):
            scores = component_scores(s, OpponentModel(), OpponentModel())
            combined = {a: .1*scores['pressure'][a]+.1*scores['control'][a]+.8*scores['chaos'][a] for a in s.legal_actions()}
            return np.array(list(_softmax(combined, .35).values()))
        result[k] = probs(solver.nodes[ids[0]].state)
        for i in ids[1:]:
            if not np.allclose(result[k], probs(solver.nodes[i].state), atol=1e-14):
                raise ValueError('Chaos snapshot leaks hidden state')
    return result


def cross_value(solver, policy, opponent):
    v0 = solver.profile_value({k: policy[k] if k[0]==0 else opponent[k] for k in solver.infos})
    v1 = -solver.profile_value({k: opponent[k] if k[0]==0 else policy[k] for k in solver.infos})
    return (v0+v1)/2


def compile_measurements(solver):
    """Fixed reference covariates, independent of all generated policies/outcomes."""
    oracle = UniformContinuationValueModel(); reference = {}
    for k, ids in solver.infos.items():
        if panel(k) != 0:
            continue
        n = solver.nodes[ids[0]]; s = n.state
        qmap = oracle.action_values(s)
        q = np.array([qmap[a] for a in n.actions]); regret = q.max()-q
        scale = max(s.pot+s.bet_size, 1)
        adequacy = np.maximum(0, 1-regret/scale)
        pressure = []
        for a in n.actions:
            nxt = apply_action(s, a)
            pressure.append((1/len(nxt.legal_actions())+nxt.to_call/max(nxt.pot+nxt.to_call, 1))/3
                            if not nxt.terminal and nxt.actor!=s.actor and nxt.to_call else 0.)
        reference[k] = (regret, np.exp(-regret/scale), adequacy, np.array(pressure))
    return reference


def measure(solver, policy, reference):
    # Input boundary is only key -> probability vectors; labels and generator
    # metadata cannot enter. Uniform over feature infosets, not reachable hands.
    rows=[]; contexts=defaultdict(list)
    for k in sorted(reference, key=repr):
        n=solver.nodes[solver.infos[k][0]]; s=n.state; p=np.array(policy[k])
        freq=np.array([sum(p[j] for j,a in enumerate(n.actions) if a==b) for b in ACTIONS])
        ent=-float(p @ np.log(np.maximum(p, 1e-15)))/np.log(len(p))
        regret, control, adequacy, pressure=reference[k]
        surprisal=-np.log(np.maximum(p, 1e-15))/np.log(len(p))
        base=np.r_[freq, ent, freq[1]+freq[4], float(p @ regret)]
        candidate=np.array([p @ pressure, p @ control, (p*surprisal) @ adequacy])
        rows.append((base,candidate))
        # Conventional public-history and own-card features, fixed strata.
        ctx=(s.round_index, bool(s.to_call), min(2,sum(a in ('bet','raise') for a in s.history)))
        contexts[ctx].append(np.r_[freq,ent, (2*equity(s,s.actor)-1)*(freq[1]+freq[4])])
    baseline=list(np.mean([r[0] for r in rows],axis=0))
    for r in (0,1):
        for f in (False,True):
            for h in (0,1,2):
                vals=contexts.get((r,f,h), [])
                baseline.extend(np.mean(vals,axis=0) if vals else np.zeros(7))
    # Exact exploitability is an explicitly allowed conventional comparator.
    br0=solver.best_response_value(policy,0); br1=solver.best_response_value(policy,1)
    baseline.extend([br0,br1,(br0+br1)/2])
    return np.array(baseline),np.mean([r[1] for r in rows],axis=0)


def matched_response(solver,policy):
    vals=[]
    for k,ids in solver.infos.items():
        if panel(k)!=1: continue
        s=solver.nodes[ids[0]].state
        if s.private[s.actor]!=0: continue
        cards=list(s.private); cards[s.actor]=2
        other=(s.actor, replace(s,private=tuple(cards)).information_key(s.actor))
        if other in policy:
            vals.append(.5*float(np.abs(policy[k]-policy[other]).sum()))
    if not vals: raise ValueError('empty outcome panel')
    return float(np.mean(vals))


def fit_predict(x,y,test,penalty=None):
    # Fit all preprocessing on training family only, including bootstrap fits.
    mu=x.mean(axis=0); sd=x.std(axis=0); sd[sd<1e-10]=1
    z=(x-mu)/sd; t=(test-mu)/sd; center=y.mean(axis=0)
    a=CONFIG['ridge_penalty'] if penalty is None else penalty
    # Dual solve handles p>n; intercept unpenalized.
    return t @ z.T @ np.linalg.solve(z @ z.T+a*np.eye(len(x)),y-center)+center


def holm(values):
    order=np.argsort(values); adjusted=np.empty(len(values)); current=0.
    for j,i in enumerate(order):
        current=max(current,(len(values)-j)*values[i]); adjusted[i]=min(1.,current)
    return adjusted


def paired_inference(trainx,trainc,trainy,testx,testc,testy,seed):
    base=fit_predict(trainx,trainy,testx)
    aug=fit_predict(np.c_[trainx,trainc],trainy,np.c_[testx,testc])
    scale=np.maximum(trainy.var(axis=0),1e-12)
    diffs=((testy-base)**2-(testy-aug)**2)/scale
    rng=np.random.default_rng(seed); boots=[]
    for _ in range(CONFIG['bootstrap']):
        a=rng.integers(len(trainx),size=len(trainx)); b=rng.integers(len(testx),size=len(testx))
        pb=fit_predict(trainx[a],trainy[a],testx[b])
        pa=fit_predict(np.c_[trainx[a],trainc[a]],trainy[a],np.c_[testx[b],testc[b]])
        boots.append(np.mean(((testy[b]-pb)**2-(testy[b]-pa)**2)/scale,axis=0))
    boots=np.array(boots); point=diffs.mean(axis=0); centered=boots-point
    p_super=(1+(centered>=point).sum(axis=0))/(len(boots)+1)
    p_null=(1+(centered<=point-CONFIG['practical_benefit_margin']).sum(axis=0))/(len(boots)+1)
    return dict(base_predictions=base.tolist(),augmented_predictions=aug.tolist(),
                paired_normalized_mse_differences=diffs.tolist(),
                baseline_mse=np.mean((testy-base)**2,axis=0).tolist(),
                augmented_mse=np.mean((testy-aug)**2,axis=0).tolist(),
                improvement=point.tolist(),ci95=np.quantile(boots,[.025,.975],axis=0).T.tolist(),
                upper95=np.quantile(boots,.95,axis=0).tolist(),
                p_superiority=p_super.tolist(),p_no_meaningful_benefit=p_null.tolist(),
                training_variance_scale=scale.tolist(),n_train=len(trainx),n_test=len(testx))


def pack(policy):
    return [[list(k),v.tolist()] for k,v in sorted(policy.items(),key=lambda x:repr(x[0]))]


def unpack(data):
    return {tuple(k):np.array(v) for k,v in data}


def reference_br(solver,policy,player):
    """Independent recursive policy iteration; returns executable BR table.

    Unlike the solver's backward pass, recomputes counterfactual reach by root
    walks and updates the complete behavioral response through policy iteration.
    """
    response={k:np.eye(len(v))[0] for k,v in policy.items() if k[0]==player}
    for _ in range(20):
        weights=defaultdict(float)
        def reach(i,w):
            n=solver.nodes[i]; weights[i]+=w
            if n.key is None: return
            for j,c in enumerate(n.children):
                reach(c,w if n.key[0]==player else w*policy[n.key][j])
        for root in solver.roots: reach(root,1/len(solver.roots))
        values={}
        def value(i):
            n=solver.nodes[i]
            if n.key is None: v=utility(n.state,player)
            else:
                probs=response[n.key] if n.key[0]==player else policy[n.key]
                v=sum(p*value(c) for p,c in zip(probs,n.children))
            values[i]=v; return v
        total=sum(value(i) for i in solver.roots)/len(solver.roots)
        changes=0
        for k,ids in solver.infos.items():
            if k[0]!=player: continue
            q=sum((weights[i]*np.array([values[c] for c in solver.nodes[i].children]) for i in ids),start=np.zeros(len(policy[k])))
            best=int(np.argmax(q)); new=np.eye(len(q))[best]
            changes+=not np.array_equal(new,response[k]); response[k]=new
        if not changes: return total,response
    raise RuntimeError('independent response iteration did not converge')


def evaluate_chaos(solver):
    from . import blind_leduc_reference as reference
    rules=reference.verify_tree(solver)
    target=fixed_chaos(solver); validate_policy(solver,target)
    reference_profile=reference.profile_value(target)
    if abs(reference_profile-solver.profile_value(target))>1e-10: raise AssertionError('rule profile mismatch')
    br=[solver.best_response_value(target,p) for p in (0,1)]
    checks=[]; responses={}; verified=[]
    for p in (0,1):
        value,response=reference_br(solver,target,p); responses[p]=pack(response)
        # Execute the response against the fixed opponent in the original tree.
        profile={k:response[k] if k[0]==p else target[k] for k in target}
        executed=solver.profile_value(profile)*(1 if p==0 else -1)
        independent_rules_value=reference.profile_value(profile,p)
        if abs(independent_rules_value-br[p])>1e-9: raise AssertionError('independent rule BR mismatch')
        verified.append(dict(player=p,reference_value=value,executed_value=executed,solver_value=br[p],independent_rules_value=independent_rules_value))
        checks.append(abs(value-br[p])<1e-9 and abs(executed-br[p])<1e-9)
    if not all(checks): raise AssertionError('best response cross-check failed')
    return dict(target='score PCC Chaos frozen empty-history behavioral snapshot',
                weights=[.1,.1,.8],temperature=.35,history_updates=False,
                scope='new fixed snapshot; not original evolving tournament Chaos',
                unit='net chips per hand',profile_value_player0=solver.profile_value(target),
                br_player0=br[0],br_player1=br[1],nash_conv=sum(br),exploitability=sum(br)/2,
                independent_br_checks=verified,br_tables=responses,target_table=pack(target),
                external_solver_validated=False,independent_rule_checks=rules,independent_profile_value=reference_profile)


def freeze(directory):
    directory.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parent.parent
    seeds={'A':[730001+101*i for i in range(CONFIG['replicates'])],
           'B':[930001+101*i for i in range(CONFIG['replicates'])],
           'opaque_id_shuffle':71201,'bootstrap':CONFIG['inference_seed']}
    frozen=dict(config=CONFIG,sources=source_hashes(),seeds=seeds,
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
                python=platform.python_version(),numpy=np.__version__,
                procedural_blinding_only=True)
    write_new(directory/'freeze.json',frozen)
    event(directory,'pipeline_frozen',freeze_sha256=digest(directory/'freeze.json'))
    return frozen


def run(directory):
    frozen=freeze(directory); solver=LeducCFR()
    # Protect every pre-existing historical validation artifact, not only JSON.
    historical={str(p.relative_to(directory.parent)):digest(p) for p in directory.parent.rglob('*') if p.is_file() and directory not in p.parents}
    write_new(directory/'historical_before.json',historical)
    check_freeze(frozen)
    ids=np.random.default_rng(frozen['seeds']['opaque_id_shuffle']).permutation(2*CONFIG['replicates'])
    mapping={}; policies=[]; index=0
    for cohort in ('A','B'):
        for rep,seed in enumerate(frozen['seeds'][cohort]):
            opaque=f'policy-{ids[index]:03d}';index+=1
            p=sampled_cfr(solver,seed) if cohort=='A' else conventional_policy(solver,seed)
            validate_policy(solver,p)
            policies.append(dict(id=opaque,cohort=cohort,replicate=rep,probabilities=pack(p)))
            mapping[opaque]=dict(family='sampled-chance CFR' if cohort=='A' else 'equity pot-odds logit',seed=seed)
        print('generated opaque cohort',cohort,flush=True)
    # Mapping is committed separately and is not passed to measurement/prediction.
    write_new(directory/'sealed_identity.json',mapping)
    write_new(directory/'policies.json',policies)
    event(directory,'opaque_policies_generated',identity_commitment=digest(directory/'sealed_identity.json'),policies_sha256=digest(directory/'policies.json'))
    reference=compile_measurements(solver)
    opponents=[conventional_policy(solver,parameters=p) for p in ((5.,-1.5,.5,0.),(2.,1.5,.8,0.))]
    records=[]
    for item in policies:
        p=unpack(item['probabilities']);b,c=measure(solver,p,reference)
        y=[cross_value(solver,p,o) for o in opponents]+[matched_response(solver,p)]
        records.append(dict(id=item['id'],cohort=item['cohort'],replicate=item['replicate'],baseline=b.tolist(),candidate=c.tolist(),outcomes=y))
    write_new(directory/'measurements.json',records)
    event(directory,'measurement_frozen',measurements_sha256=digest(directory/'measurements.json'))
    check_freeze(frozen)
    results={}
    for d,(train,test) in enumerate((('A','B'),('B','A'))):
        a=[r for r in records if r['cohort']==train];b=[r for r in records if r['cohort']==test]
        arrays=lambda rows,key:np.array([r[key] for r in rows])
        results[f'{train}_to_{test}']=paired_inference(arrays(a,'baseline'),arrays(a,'candidate'),arrays(a,'outcomes'),arrays(b,'baseline'),arrays(b,'candidate'),arrays(b,'outcomes'),CONFIG['inference_seed']+d)
    ps=[]
    for r in results.values(): ps.extend(r['p_superiority']+r['p_no_meaningful_benefit'])
    adj=holm(ps); offset=0
    for r in results.values():
        r['holm_superiority']=adj[offset:offset+3].tolist();r['holm_no_meaningful_benefit']=adj[offset+3:offset+6].tolist();offset+=6
    write_new(directory/'predictions_frozen.json',results)
    event(directory,'predictions_frozen',predictions_sha256=digest(directory/'predictions_frozen.json'))
    # The first and only identity-file read is after measurement and predictions.
    check_freeze(frozen)
    revealed=reveal(directory,frozen)
    reveal_order=sorted(revealed)
    event(directory,'identities_revealed',reveal_order=reveal_order,identity_sha256=digest(directory/'sealed_identity.json'))
    chaos=evaluate_chaos(solver);write_new(directory/'chaos_best_response.json',chaos)
    after={k:digest(directory.parent/k) for k in historical}
    if after!=historical: raise AssertionError('historical validation artifacts changed')
    report=dict(status='completed',config=CONFIG,endpoints=ENDPOINTS,results=results,
                cohort_identity={c:sorted({revealed[r['id']]['family'] for r in records if r['cohort']==c}) for c in ('A','B')},
                historical_files_verified=len(historical),all_historical_unchanged=True,
                baseline_features=len(records[0]['baseline']),candidate_features=['pressure reference index','control uniform-reference efficiency','adequacy-weighted normalized surprisal'],
                panel_counts={'measurement':sum(panel(k)==0 for k in solver.infos),'heldout':sum(panel(k)==1 for k in solver.infos)},
                blindness='procedural only; same author implements generators and measurement; cohort identity may be inferred',
                target_scope=chaos['scope'],chaos_exploitability=chaos['exploitability'],
                limitations=['No independent investigator or hardware access barrier.',
                 'New fixed reference PCC candidate metrics; not exhaustive validation of PCC or historical estimators.',
                 'Policy populations are restricted; inference conditional on fixed reference opponents and game.',
                 'Bootstrap tests are approximate; intervals are marginal, Holm p values control prespecified 12-test family.',
                 'Exact zero benefit cannot be established; no-benefit test uses frozen 0.01 variance-normalized margin.',
                 'No external OpenSpiel replication; independent BR algorithm shares local tree.',
                 'Historical policy evolution and cross-game claims remain untested.'])
    write_new(directory/'report.json',report)
    event(directory,'completed',report_sha256=digest(directory/'report.json'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.output)
    print(json.dumps({k:report[k] for k in ('status','chaos_exploitability','historical_files_verified')},indent=2))

if __name__=='__main__': main()
