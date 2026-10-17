#!/usr/bin/env python3
"""Verify analytical linear forward/backward MAC counts against PyTorch profiler."""
import json
import torch
import run_grid as g


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows=[]
    for name in g.DATASETS:
        _,dims,k,_=g.load_data(name,0,'native')
        for h in [8,64]:
            for kind in ['balanced','skewed','mlp']:
                model=g.make_model(name,kind,h,0,dims,k)
                x=torch.randn(3,sum(dims))
                c=g.counts(model)
                first=sum(dims)*(g.dense_widths(dims,h,k)[0] if kind=='mlp' else h)
                expected=3*(3*c['forward_macs']-first)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
                    model(x).sum().backward()
                measured=sum(e.flops for e in prof.key_averages() if e.key in ['aten::mm','aten::addmm'])//2
                row={'dataset':name,'width':h,'model':kind,'batch':3,
                     'analytical_training_linear_macs':expected,'profiler_linear_macs':measured,
                     'match':expected==measured}
                rows.append(row)
                assert row['match'],row
    g.dump(g.ROOT/'results/compute_profiler_audit.json',rows)
    print(json.dumps({'linear_forward_backward_accounting_checks':len(rows),'all_match':all(r['match'] for r in rows)}))


if __name__=='__main__':
    main()
