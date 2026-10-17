#!/usr/bin/env python3
"""Reproducible expanded topology study. Training/evaluation require torch+numpy.

python run_grid.py --out results
python run_grid.py --out results --resume
python run_grid.py --verify results
python run_grid.py --self-check
"""
import argparse
import copy
import csv
import functools
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATASETS = ['optdigits', 'pendigits', 'spambase', 'satimage']
KINDS = ['balanced', 'random', 'skewed', 'mlp']
WIDTHS = [8, 16, 32, 64]
SEEDS = list(range(10))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def split_indices(y, seed, fraction=.2):
    rng = np.random.default_rng(seed)
    keep, hold = [], []
    for c in np.unique(y):
        ix = rng.permutation(np.flatnonzero(y == c))
        n = round(len(ix)*fraction)
        hold.extend(ix[:n])
        keep.extend(ix[n:])
    keep, hold = np.sort(keep), np.sort(hold)
    assert len(np.intersect1d(keep, hold)) == 0
    assert len(keep) + len(hold) == len(y)
    return keep, hold


@functools.lru_cache(None)
def raw_data(name):
    base = ROOT/'data'
    meta = {}
    if name == 'spambase':
        a = np.loadtxt(base/'spambase.data', delimiter=',', dtype=np.float32)
        assert a.shape == (4601, 58)
        _, group = np.unique(a[:, :-1], axis=0, return_inverse=True)
        conflict = []
        for g in np.unique(group):
            if len(np.unique(a[group == g, -1])) > 1:
                conflict.append(g)
        valid = np.flatnonzero(~np.isin(group, conflict))
        _, unique_ix = np.unique(a[valid], axis=0, return_index=True)
        retained = np.sort(valid[unique_ix])
        clean = a[retained]
        assert len(clean) == 4204
        tr, te = split_indices(clean[:, -1], 20261015)
        a, b = clean[tr], clean[te]
        meta = {'raw_records':4601, 'retained_records':4204,
                'conflicting_groups':len(conflict), 'retained_raw_indices':retained,
                'fixed_pool_raw_indices':retained[tr], 'fixed_test_raw_indices':retained[te]}
    elif name == 'satimage':
        a = np.loadtxt(base/'sat.trn', dtype=np.float32)
        b = np.loadtxt(base/'sat.tst', dtype=np.float32)
        assert a.shape == (4435, 37) and b.shape == (2000, 37)
    else:
        a = np.loadtxt(base/(name+'.tra'), delimiter=',', dtype=np.float32)
        b = np.loadtxt(base/(name+'.tes'), delimiter=',', dtype=np.float32)
        assert (len(a), len(b)) == ((3823,1797) if name == 'optdigits' else (7494,3498))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    labels = np.unique(np.concatenate((a[:, -1], b[:, -1])))
    ya, yb = np.searchsorted(labels, a[:, -1]), np.searchsorted(labels, b[:, -1])
    return a[:, :-1].copy(), ya, b[:, :-1].copy(), yb, labels, meta


@functools.lru_cache(None)
def load_data(name, seed, ordering):
    xa, ya, xb, yb, labels, raw_meta = raw_data(name)
    ti, vi = split_indices(ya, 20261015+seed)
    if name in ['optdigits','pendigits']:
        scale = 16 if name == 'optdigits' else 100
        xa, xb = xa/scale, xb/scale
        mean, std = np.zeros(xa.shape[1], dtype=np.float32), np.ones(xa.shape[1], dtype=np.float32)*scale
    else:
        if name == 'spambase':
            assert xa.min() >= 0 and xb.min() >= 0
            xa, xb = np.log1p(xa), np.log1p(xb)
        mean, std = xa[ti].mean(0), xa[ti].std(0)
        std[std < 1e-8] = 1.
        xa, xb = (xa-mean)/std, (xb-mean)/std
    if name == 'optdigits':
        groups = [[rr*8+cc for rr in range(r,r+2) for cc in range(c,c+4)]
                  for r in range(0,8,2) for c in range(0,8,4)]
    else:
        groups = [ix.tolist() for ix in np.array_split(np.arange(xa.shape[1]), 9 if name=='satimage' else 8)]
    dims = tuple(map(len, groups))
    order = np.array(sum(groups, []))
    if ordering == 'permuted':
        order = np.random.default_rng(70000+100*DATASETS.index(name)+seed).permutation(xa.shape[1])
    assert sorted(order.tolist()) == list(range(xa.shape[1]))
    def xy(x,y):
        return torch.tensor(x[:, order].copy(), dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    data = {'train':xy(xa[ti],ya[ti]), 'val':xy(xa[vi],ya[vi]), 'test':xy(xb,yb)}
    for x, y in data.values():
        assert torch.isfinite(x).all() and y.min() >= 0 and y.max() < len(labels)
    meta = {'train_indices':ti, 'validation_indices':vi, 'feature_order':order,
            'mean':mean, 'scale':std, 'class_labels':labels,
            **{k:v for k,v in raw_meta.items() if isinstance(v,np.ndarray)}}
    return data, dims, len(labels), meta


def topology(n, kind, seed):
    rng = random.Random(107+seed)
    operations, depths = [], []
    def rec(ids, depth=0):
        if len(ids)==1:
            depths.append(depth)
            return ids[0]
        k = (len(ids)//2 if kind=='balanced' else
             (1 if seed%2==0 else len(ids)-1) if kind=='skewed' else rng.randrange(1,len(ids)))
        l, r = rec(ids[:k],depth+1), rec(ids[k:],depth+1)
        index = n+len(operations)
        operations.append((l,r))
        return index
    assert rec(list(range(n))) == 2*n-2
    return operations, depths


class Tree(nn.Module):
    def __init__(self, dims, width, classes, kind, seed):
        super().__init__()
        self.dims = dims
        self.leaves = nn.ModuleList([nn.Linear(d,width) for d in dims])
        self.merges = nn.ModuleList([nn.Linear(2*width,width) for _ in range(len(dims)-1)])
        self.head = nn.Linear(width,classes)
        self.operations, self.depths = topology(len(dims),kind,seed)

    def forward(self,x):
        nodes = [F.relu(layer(patch)) for layer,patch in zip(self.leaves,x.split(self.dims,dim=1))]
        for layer,(a,b) in zip(self.merges,self.operations):
            nodes.append(F.relu(layer(torch.cat((nodes[a],nodes[b]),dim=1))))
        return self.head(nodes[-1])


def tree_counts(dims,h,k):
    n, d = len(dims), sum(dims)
    macs = d*h + (n-1)*2*h*h + h*k
    return macs+(2*n-1)*h+k, macs


@functools.lru_cache(None)
def dense_widths(dims,h,k):
    pt, mt = tree_counts(dims,h,k)
    d = sum(dims)
    choices = []
    for a in range(2,1025):
        # Convex minimax objective in b: crossings and feasible boundaries.
        bp = (pt-(d+1)*a-k)/(a+k+1)
        bm = (mt-d*a)/(a+k)
        cross = (2*pt*mt-mt*((d+1)*a+k)-pt*d*a)/(mt*(a+k+1)+pt*(a+k))
        lower = max(8,math.ceil(a/4))
        bs = {lower,a}
        for candidate in (bp,bm,cross):
            bs.update(range(math.floor(candidate)-1,math.floor(candidate)+3))
        for b in bs:
            if lower<=b<=a:
                p, m = (d+1)*a+(a+k+1)*b+k, d*a+(a+k)*b
                choices.append((max(abs(p/pt-1),abs(m/mt-1)),a,b))
    return min(choices)[1:]


def make_model(name,kind,width,seed,dims,classes):
    torch.manual_seed(seed+10000*DATASETS.index(name))
    if kind=='mlp':
        a,b = dense_widths(dims,width,classes)
        return nn.Sequential(nn.Linear(sum(dims),a),nn.ReLU(),nn.Linear(a,b),nn.ReLU(),nn.Linear(b,classes))
    return Tree(dims,width,classes,kind,seed)


def counts(model):
    return {'params':sum(p.numel() for p in model.parameters()),
            'forward_macs':sum(m.in_features*m.out_features for m in model.modules() if isinstance(m,nn.Linear)),
            'max_merge_depth':max(model.depths) if isinstance(model,Tree) else None,
            'mean_merge_depth':float(np.mean(model.depths)) if isinstance(model,Tree) else None}


@torch.no_grad()
def evaluate(model,xy):
    model.eval()
    x,y = xy
    logits = model(x)
    return {'loss':F.cross_entropy(logits,y).item(),
            'accuracy':(logits.argmax(1)==y).float().mean().item()},logits.numpy()


def run_id(job):
    return '_'.join(str(job[k]) for k in ['dataset','ordering','width','seed','model'])


def train_one(job,epochs):
    name,seed,ordering,h,kind = [job[k] for k in ['dataset','seed','ordering','width','model']]
    data,dims,k,meta = load_data(name,seed,ordering)
    model = make_model(name,kind,h,seed,dims,k)
    arch = counts(model)
    pt,mt = tree_counts(dims,h,k)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    gen = torch.Generator().manual_seed(5000+seed)
    x,y = data['train']
    history,best_state,best_loss,best_epoch = [],None,float('inf'),None
    training_seconds = 0.
    run_start = time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(x),generator=gen)
        for ix in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ix]),y[ix])
            loss.backward()
            optimizer.step()
        training_seconds += time.perf_counter()-start
        vm,_ = evaluate(model,data['val'])
        tm,_ = evaluate(model,data['train'])
        history.append({**job,'epoch':epoch,'train_loss':tm['loss'],'train_accuracy':tm['accuracy'],
                        'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],
                        'train_seconds':training_seconds,'est_train_macs':3*arch['forward_macs']*len(x)*epoch})
        if vm['loss']<best_loss:
            best_loss,best_epoch = vm['loss'],epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    tm,_ = evaluate(model,data['train'])
    vm,_ = evaluate(model,data['val'])
    row = {**job,**arch,'tree_target_params':pt,'tree_target_macs':mt,
           'parameter_mismatch':arch['params']/pt-1,'mac_mismatch':arch['forward_macs']/mt-1,
           'epochs':epochs,'selected_epoch':best_epoch,'train_samples':len(x),'val_samples':len(data['val'][0]),
           'test_samples':len(data['test'][0]),'updates':math.ceil(len(x)/256)*epochs,
           'train_loss':tm['loss'],'train_accuracy':tm['accuracy'],'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],
           'train_seconds':training_seconds,'run_seconds':time.perf_counter()-run_start,
           'est_train_macs':3*arch['forward_macs']*len(x)*epochs}
    return model,row,history,meta


def self_check():
    checks = []
    for name in DATASETS:
        _,dims,k,_ = load_data(name,0,'native')
        for h in WIDTHS:
            ms = [make_model(name,kind,h,0,dims,k) for kind in KINDS]
            x = torch.randn(5,sum(dims))
            for m in ms:
                assert m(x).shape == (5,k)
                m(x).square().mean().backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            pt,mt = tree_counts(dims,h,k)
            for m in ms[:3]:
                assert (counts(m)['params'],counts(m)['forward_macs'])==(pt,mt)
            assert all(torch.equal(p,q) for m in ms[1:3] for p,q in zip(ms[0].parameters(),m.parameters()))
            assert max(ms[0].depths)==math.ceil(math.log2(len(dims)))
            assert max(ms[2].depths)==len(dims)-1
            assert not torch.equal(ms[0](x),ms[2](x))
            checks.append({'dataset':name,'width':h,'tree_params':pt,'tree_macs':mt,
                           'dense_widths':dense_widths(dims,h,k),'dense_counts':counts(ms[-1])})
    # Synthetic-only optimization smoke check. No real held-out evaluation here.
    toy=Tree((2,)*8,8,3,'balanced',0)
    opt=torch.optim.AdamW(toy.parameters(),lr=.003)
    for _ in range(2):
        opt.zero_grad()
        F.cross_entropy(toy(torch.randn(20,16)),torch.arange(20)%3).backward()
        opt.step()
    return checks


def verify(out):
    manifest=json.loads((out/'manifest.json').read_text())
    answers=[]
    for name in DATASETS:
        job={'dataset':name,'ordering':'native','width':16,'seed':0,'model':'balanced'}
        model,row,_,_=train_one(job,manifest['epochs'])
        saved=json.loads((out/'runs'/f'{run_id(job)}.json').read_text())
        data,*_=load_data(name,0,'native')
        _,logits=evaluate(model,data['test'])
        prior=np.load(out/'predictions'/f'{run_id(job)}.npz')['logits']
        old_state=torch.load(out/'checkpoints'/f'{run_id(job)}.pt',weights_only=True)
        keys=['selected_epoch','train_loss','train_accuracy','val_loss','val_accuracy']
        result={'dataset':name,'metrics_exact':all(row[k]==saved[k] for k in keys),
                'checkpoint_exact':all(torch.equal(v,old_state[k]) for k,v in model.state_dict().items()),
                'test_logits_exact':bool(np.array_equal(logits,prior)),
                'max_abs_logit_difference':float(np.max(np.abs(logits-prior)))}
        assert result['metrics_exact'] and result['checkpoint_exact'] and result['test_logits_exact']
        answers.append(result)
        print(result,flush=True)
    dump(out/'reproducibility.json',answers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify',type=Path)
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--epochs',type=int,default=60)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checks=self_check()
    if args.self_check:
        print(json.dumps(checks,indent=2))
        return
    if args.verify:
        verify(args.verify.resolve())
        return
    out=args.out.resolve()
    jobs=[{'dataset':d,'ordering':o,'width':h,'seed':s,'model':k}
          for d in DATASETS for o in ['native','permuted']
          for h in (WIDTHS if o=='native' else [16,64]) for s in SEEDS for k in KINDS]
    random.Random(20261015).shuffle(jobs)
    hashes={str(f.relative_to(ROOT)):sha(f) for f in
            [ROOT/'run_grid.py',ROOT/'PROTOCOL.md',*sorted((ROOT/'data').iterdir())] if f.is_file()}
    if (out/'manifest.json').exists():
        assert args.resume,'Use --resume or select a new output directory.'
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['hashes']==hashes and manifest['epochs']==args.epochs
    else:
        out.mkdir(parents=True,exist_ok=True)
        manifest={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                  'platform':platform.platform(),'processor':platform.processor(),'device':'cpu',
                  'threads':1,'deterministic':True,'dtype':'float32','epochs':args.epochs,
                  'seeds':SEEDS,'widths':WIDTHS,'batch_size':256,'lr':.003,'weight_decay':.0001,
                  'optimizer':'AdamW','hashes':hashes,'jobs':jobs,'started_unix':time.time()}
        dump(out/'manifest.json',manifest)
    for folder in ['runs','history','checkpoints','predictions','preprocessing']:
        (out/folder).mkdir(exist_ok=True)
    dump(out/'architecture_checks.json',checks)
    start=time.perf_counter()
    for i,job in enumerate(jobs):
        jid=run_id(job)
        path=out/'runs'/f'{jid}.json'
        if path.exists():
            continue
        model,row,history,meta=train_one(job,args.epochs)
        torch.save(model.state_dict(),out/'checkpoints'/f'{jid}.pt')
        write_csv(out/'history'/f'{jid}.csv',history)
        np.savez_compressed(out/'preprocessing'/f"{job['dataset']}_{job['ordering']}_{job['seed']}.npz",**meta)
        if isinstance(model,Tree):
            dump(out/'checkpoints'/f'{jid}_topology.json',{'operations':model.operations,'depths':model.depths})
        dump(path,row) # Commit the completed run last, so --resume is safe.
        if (i+1)%8==0 or i==0:
            print(f"trained {i+1}/{len(jobs)} elapsed={time.perf_counter()-start:.1f}s last={jid} "
                  f"val_loss={row['val_loss']:.4f} time={row['train_seconds']:.2f}s",flush=True)
    print('TRAINING COMPLETE. Beginning held-out evaluation.',flush=True)
    rows=[]
    for i,job in enumerate(jobs):
        jid=run_id(job)
        path=out/'runs'/f'{jid}.json'
        row=json.loads(path.read_text())
        data,dims,k,_=load_data(job['dataset'],job['seed'],job['ordering'])
        model=make_model(job['dataset'],job['model'],job['width'],job['seed'],dims,k)
        model.load_state_dict(torch.load(out/'checkpoints'/f'{jid}.pt',weights_only=True))
        metrics,logits=evaluate(model,data['test'])
        row.update({'test_'+k:v for k,v in metrics.items()})
        times=[]
        for _ in range(5):
            st=time.perf_counter()
            evaluate(model,data['test'])
            times.append(time.perf_counter()-st)
        row['inference_seconds']=float(np.median(times))
        np.savez_compressed(out/'predictions'/f'{jid}.npz',logits=logits,targets=data['test'][1].numpy())
        dump(path,row)
        rows.append(row)
        if (i+1)%160==0:
            print(f'evaluated {i+1}/{len(jobs)}',flush=True)
    write_csv(out/'runs.csv',rows)
    dump(out/'runs.json',rows)
    manifest['finished_unix']=time.time()
    manifest['total_elapsed_seconds']=manifest['finished_unix']-manifest['started_unix']
    dump(out/'manifest.json',manifest)
    print(f"COMPLETE {len(rows)} runs; elapsed {manifest['total_elapsed_seconds']:.1f}s",flush=True)


if __name__=='__main__':
    main()
