"""Run a fixed paired topology follow-up without changing the original study."""
import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import run_grid as g


def holm(pvalues):
    pvalues=np.asarray(pvalues,dtype=float)
    if pvalues.ndim!=1 or not np.isfinite(pvalues).all() or ((pvalues<0)|(pvalues>1)).any():
        raise ValueError('Finite p values in [0,1] required')
    order=np.argsort(pvalues); answer=np.empty(len(order)); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*pvalues[index]));answer[index]=previous
    return answer


def paired_summary(values,seed=20261017,draws=200000):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite paired differences required')
    rng=np.random.default_rng(seed); observed=abs(x.mean()); extreme=0; used=0
    if len(x)<=16:
        null=np.asarray(list(itertools.product((-1.,1.),repeat=len(x))))@x/len(x)
        p=float(np.mean(np.abs(null)>=observed-1e-15));method='exact paired sign flip'
    else:
        while used<draws:
            n=min(10000,draws-used); signs=rng.integers(0,2,(n,len(x)),dtype=np.int8)*2-1
            extreme+=int(np.sum(np.abs(signs@x/len(x))>=observed-1e-15));used+=n
        p=(extreme+1)/(draws+1);method='Monte Carlo paired sign flip, add-one'
    boot=np.mean(x[rng.integers(0,len(x),(20000,len(x)))],axis=1)
    return {'n_pairs':len(x),'mean_difference':float(x.mean()),'median_difference':float(np.median(x)),
            'sample_sd':float(x.std(ddof=1)),'descriptive_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),
            'positive_pairs':int((x>0).sum()),'ties':int((x==0).sum()),'negative_pairs':int((x<0).sum()),
            'p_value':p,'test':method,'monte_carlo_draws':used}


def analyze(rows,protocol):
    lookup={g.run_id(r):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate run condition')
    expected=[dict(dataset=d,ordering=o,width=protocol['width'],seed=s,model=m)
              for d in protocol['datasets'] for s in protocol['seeds']
              for o in protocol['orderings'] for m in protocol['models']]
    if set(lookup)!={g.run_id(j) for j in expected}:raise ValueError('Incomplete or unexpected paired grid')
    metrics={};paired=[]
    for d in protocol['datasets']:
        differences={}
        for o in protocol['orderings']:
            differences[o]=[]
            for s in protocol['seeds']:
                def row(m):return lookup[g.run_id(dict(dataset=d,ordering=o,width=protocol['width'],seed=s,model=m))]
                b,r=row('balanced'),row('random')
                if b['params']!=r['params'] or b['forward_macs']!=r['forward_macs']:
                    raise ValueError('Topology comparison must preserve parameters and linear MACs')
                diff=b['test_accuracy']-r['test_accuracy'];differences[o].append(diff)
                paired.append({'dataset':d,'ordering':o,'seed':s,'balanced_accuracy':b['test_accuracy'],
                               'random_accuracy':r['test_accuracy'],'difference':diff})
        metrics[d+'_native']=paired_summary(differences['native'],protocol['analysis_seed'])
        interaction=np.asarray(differences['native'])-np.asarray(differences['permuted'])
        metrics[d+'_order_interaction']=paired_summary(interaction,protocol['analysis_seed'])
    family=protocol['primary_family'];adjusted=holm([metrics[n]['p_value'] for n in family])
    for name,p in zip(family,adjusted):metrics[name]['holm_p_value']=float(p)
    return {'primary_family':metrics,'scope':protocol['limits']},paired


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    protocol_path=g.ROOT/'FOLLOWUP_PROTOCOL.json';protocol=json.loads(protocol_path.read_text())
    hashes={str(f.relative_to(g.ROOT)):g.sha(f) for f in [Path(__file__),g.ROOT/'run_grid.py',protocol_path,*sorted((g.ROOT/'data').iterdir())] if f.is_file()}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.out.exists():
        if not a.resume:raise ValueError('Choose a new output directory or --resume')
        manifest=json.loads((a.out/'manifest.json').read_text())
        if manifest['source_sha256']!=hashes:raise ValueError('Resume source/protocol/data mismatch')
    else:
        a.out.mkdir(parents=True)
        manifest={'protocol':protocol,'source_sha256':hashes,'python':platform.python_version(),
                  'torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),
                  'started_unix':time.time(),'training_complete':False,'evaluation_complete':False}
        g.dump(a.out/'manifest.json',manifest)
    for directory in ['runs','history','checkpoints','predictions','preprocessing']:(a.out/directory).mkdir(exist_ok=True)
    jobs=[dict(dataset=d,ordering=o,width=protocol['width'],seed=s,model=m)
          for d in protocol['datasets'] for s in protocol['seeds'] for o in protocol['orderings'] for m in protocol['models']]
    for i,job in enumerate(jobs):
        jid=g.run_id(job);path=a.out/'runs'/f'{jid}.json'
        if path.exists():continue
        model,row,history,meta=g.train_one(job,protocol['epochs'])
        torch.save(model.state_dict(),a.out/'checkpoints'/f'{jid}.pt')
        g.write_csv(a.out/'history'/f'{jid}.csv',history)
        np.savez_compressed(a.out/'preprocessing'/f"{job['dataset']}_{job['ordering']}_{job['seed']}.npz",**meta)
        g.dump(path,row)
        if (i+1)%16==0:print(f'Trained {i+1}/{len(jobs)}; held-out evaluation waits for all training.',flush=True)
    manifest['training_complete']=True;g.dump(a.out/'manifest.json',manifest)
    rows=[]
    for job in jobs:
        jid=g.run_id(job);row=json.loads((a.out/'runs'/f'{jid}.json').read_text())
        data,dims,k,_=g.load_data(job['dataset'],job['seed'],job['ordering'])
        model=g.make_model(job['dataset'],job['model'],job['width'],job['seed'],dims,k)
        model.load_state_dict(torch.load(a.out/'checkpoints'/f'{jid}.pt',map_location='cpu',weights_only=True),strict=True)
        metrics,logits=g.evaluate(model,data['test']);row.update({'test_'+n:v for n,v in metrics.items()})
        np.savez_compressed(a.out/'predictions'/f'{jid}.npz',logits=logits,targets=data['test'][1].numpy())
        row['checkpoint_sha256']=g.sha(a.out/'checkpoints'/f'{jid}.pt');rows.append(row)
    summary,paired=analyze(rows,protocol);g.dump(a.out/'summary.json',summary)
    g.write_csv(a.out/'runs.csv',rows);g.write_csv(a.out/'paired.csv',paired)
    manifest.update(evaluation_complete=True,completed_runs=len(rows),finished_unix=time.time())
    g.dump(a.out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
