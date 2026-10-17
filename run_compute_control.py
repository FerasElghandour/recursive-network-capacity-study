#!/usr/bin/env python3
"""Supplementary dense baseline matched on corrected training linear MACs.

python run_compute_control.py --main results --out compute_control
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_grid as g


def train(job):
    name,h,seed=job['dataset'],job['width'],job['seed']
    data,dims,k,meta=g.load_data(name,seed,'native')
    model=g.make_model(name,'mlp',h,seed,dims,k)
    arch=g.counts(model)
    pt,mt=g.tree_counts(dims,h,k)
    a,b=g.dense_widths(dims,h,k)
    tree_per_example=3*mt-sum(dims)*h
    dense_per_example=3*arch['forward_macs']-sum(dims)*a
    x,y=data['train']
    target_samples=round(len(x)*60*tree_per_example/dense_per_example)
    target_macs=len(x)*60*tree_per_example
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    gen=torch.Generator().manual_seed(seed+5000)
    history,best_state,best_loss,best_epoch=[],None,float('inf'),None
    training_seconds,seen,updates,epoch=0.,0,0,0
    run_start=time.perf_counter()
    while seen<target_samples:
        epoch+=1
        model.train()
        start=time.perf_counter()
        order=torch.randperm(len(x),generator=gen)
        for ix in order.split(256):
            ix=ix[:target_samples-seen]
            if not len(ix):
                break
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[ix]),y[ix]).backward()
            optimizer.step()
            seen+=len(ix)
            updates+=1
        training_seconds+=time.perf_counter()-start
        vm,_=g.evaluate(model,data['val'])
        tm,_=g.evaluate(model,data['train'])
        history.append({**job,'pass':epoch,'samples_seen':seen,'epochs_equivalent':seen/len(x),
                        'train_loss':tm['loss'],'train_accuracy':tm['accuracy'],
                        'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],
                        'train_seconds':training_seconds,'linear_train_macs':seen*dense_per_example})
        if vm['loss']<best_loss:
            best_loss,best_epoch=vm['loss'],seen/len(x)
            best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    tm,_=g.evaluate(model,data['train'])
    vm,_=g.evaluate(model,data['val'])
    mismatch=seen*dense_per_example/target_macs-1
    assert abs(mismatch)<=.5*dense_per_example/target_macs+1e-12
    row={**job,**arch,'tree_target_params':pt,'parameter_mismatch':arch['params']/pt-1,
         'tree_training_macs_per_example':tree_per_example,'dense_training_macs_per_example':dense_per_example,
         'tree_training_macs':target_macs,'linear_training_macs':seen*dense_per_example,
         'training_mac_mismatch':mismatch,'samples_seen':seen,'train_samples':len(x),
         'epochs_equivalent':seen/len(x),'updates':updates,'selected_epoch':best_epoch,
         'train_loss':tm['loss'],'train_accuracy':tm['accuracy'],'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],
         'train_seconds':training_seconds,'run_seconds':time.perf_counter()-run_start}
    return model,row,history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,default=g.ROOT/'results')
    p.add_argument('--out',type=Path,default=g.ROOT/'compute_control')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    main_manifest=json.loads((args.main/'manifest.json').read_text())
    assert 'finished_unix' in main_manifest,'Run only after main grid finishes to preserve serial timings.'
    assert main_manifest['epochs']==60
    out=args.out.resolve()
    jobs=[{'dataset':d,'ordering':'native','width':h,'seed':s,'model':'mlp_compute_matched'}
          for d in g.DATASETS for h in g.WIDTHS for s in g.SEEDS]
    random.Random(20261016).shuffle(jobs)
    hashes={f.name:g.sha(f) for f in [Path(__file__),g.ROOT/'run_grid.py',g.ROOT/'COMPUTE_CONTROL_PROTOCOL.md']}
    if (out/'manifest.json').exists():
        assert args.resume
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['hashes']==hashes
    else:
        out.mkdir(parents=True,exist_ok=True)
        manifest={'hashes':hashes,'jobs':jobs,'started_unix':time.time(),'main_manifest_sha':g.sha(args.main/'manifest.json'),
                  'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':1}
        g.dump(out/'manifest.json',manifest)
    for folder in ['runs','history','checkpoints','predictions']:
        (out/folder).mkdir(exist_ok=True)
    start=time.perf_counter()
    for i,job in enumerate(jobs):
        jid=g.run_id(job)
        if (out/'runs'/f'{jid}.json').exists():
            continue
        model,row,history=train(job)
        torch.save(model.state_dict(),out/'checkpoints'/f'{jid}.pt')
        g.write_csv(out/'history'/f'{jid}.csv',history)
        g.dump(out/'runs'/f'{jid}.json',row)
        if (i+1)%16==0:
            print(f'compute control trained {i+1}/{len(jobs)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    print('Supplementary training complete; beginning test evaluation.',flush=True)
    rows=[]
    for job in jobs:
        jid=g.run_id(job)
        row=json.loads((out/'runs'/f'{jid}.json').read_text())
        data,dims,k,_=g.load_data(job['dataset'],job['seed'],'native')
        model=g.make_model(job['dataset'],'mlp',job['width'],job['seed'],dims,k)
        model.load_state_dict(torch.load(out/'checkpoints'/f'{jid}.pt',weights_only=True))
        metrics,logits=g.evaluate(model,data['test'])
        row.update({'test_'+k:v for k,v in metrics.items()})
        g.dump(out/'runs'/f'{jid}.json',row)
        np.savez_compressed(out/'predictions'/f'{jid}.npz',logits=logits,targets=data['test'][1].numpy())
        rows.append(row)
    g.dump(out/'runs.json',rows)
    g.write_csv(out/'runs.csv',rows)
    manifest['finished_unix']=time.time()
    manifest['total_elapsed_seconds']=manifest['finished_unix']-manifest['started_unix']
    g.dump(out/'manifest.json',manifest)
    print(f'COMPLETE {len(rows)} supplementary runs',flush=True)


if __name__=='__main__':
    main()
