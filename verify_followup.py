"""Replay every published follow-up model; optional exact reference-logit check."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import run_grid as g

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=g.ROOT/'results/followup-v1');p.add_argument('--strict-logits',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=json.loads((a.results/'replay-manifest.json').read_text());rows=list(csv.DictReader((a.results/'runs.csv').open()))
    if len(rows)!=256 or len({g.run_id(r) for r in rows})!=256:raise ValueError('Expected all 256 unique conditions')
    archives={};exact=0
    for name,digest in manifest['weight_archives'].items():
        if g.sha(a.results/name)!=digest:raise ValueError('Weight archive fingerprint mismatch')
        archives[name.split('-')[0]]=np.load(a.results/name,allow_pickle=False)
    try:
        for row in rows:
            job={k:row[k] for k in ['dataset','ordering','model']};job.update(width=int(row['width']),seed=int(row['seed']))
            jid=g.run_id(job);data,dims,k,_=g.load_data(job['dataset'],job['seed'],job['ordering'])
            model=g.make_model(job['dataset'],job['model'],job['width'],job['seed'],dims,k)
            archive=archives[job['dataset']];model.load_state_dict({key:torch.from_numpy(archive[jid+'|'+key].copy()) for key in model.state_dict()},strict=True)
            metrics,logits=g.evaluate(model,data['test'])
            if not math.isclose(metrics['accuracy'],float(row['test_accuracy']),abs_tol=1e-7):raise ValueError('Prediction accuracy mismatch: '+jid)
            if not math.isclose(metrics['loss'],float(row['test_loss']),rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Prediction loss mismatch: '+jid)
            original=manifest['predictions'][jid]
            if not np.allclose(logits[:8],original['first_eight_logits'],atol=1e-5,rtol=1e-5):raise ValueError('Representative logits changed: '+jid)
            match=hashlib.sha256(logits.tobytes()).hexdigest()==original['logits_sha256'];exact+=int(match)
            if a.strict_logits and not match:raise ValueError('Exact reference-logit fingerprint changed: '+jid)
    finally:
        for archive in archives.values():archive.close()
    print(json.dumps({'models_replayed':len(rows),'accuracy_and_loss_match':True,'exact_logit_fingerprints':exact,
                      'strict_reference_mode':a.strict_logits,'scope':'Exact hashes require the same numerical runtime; bounded tolerance is used for cross-runtime logits/loss.'},indent=2))

if __name__=='__main__':main()
