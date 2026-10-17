#!/usr/bin/env python3
"""Analyze completed run_grid.py outputs; save tables, plots and report.

python analyze_results.py --results results
Requires numpy and matplotlib, in addition to training dependencies for run_grid.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
DATASETS=['optdigits','pendigits','spambase','satimage']
TITLES={'optdigits':'Scanned digits','pendigits':'Pen trajectories','spambase':'Email spam','satimage':'Satellite imagery'}
KINDS=['balanced','random','skewed','mlp']
WIDTHS=[8,16,32,64]
METRICS=['test_accuracy','test_loss','train_seconds']


def write_csv(path,rows):
    if rows:
        with Path(path).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def estimate(values):
    d=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261015)
    b=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(d))))
    p=float(np.mean(np.abs(signs@d/len(d)) >= abs(d.mean())-1e-12))
    q=np.quantile(b,[.025,.975])
    return {'mean':float(d.mean()),'sd':float(d.std(ddof=1)),
            'median':float(np.median(d)),'minimum':float(d.min()),'maximum':float(d.max()),
            'ci95_low':float(q[0]),'ci95_high':float(q[1]),'p_signflip':p}


def holm(rows,key='p_signflip'):
    order=sorted(range(len(rows)),key=lambda i:rows[i][key])
    previous=0.
    for j,i in enumerate(order):
        previous=max(previous,min(1.,(len(rows)-j)*rows[i][key]))
        rows[i]['p_holm']=previous


def analyze(rows,out):
    assert len(rows)==960
    lookup={(r['dataset'],r['ordering'],r['width'],r['seed'],r['model']):r for r in rows}
    assert len(lookup)==960
    summary=[]
    for dataset in DATASETS:
        for ordering in ['native','permuted']:
            for width in (WIDTHS if ordering=='native' else [16,64]):
                for kind in KINDS:
                    runs=[lookup[dataset,ordering,width,seed,kind] for seed in range(10)]
                    s={k:runs[0][k] for k in ['dataset','ordering','width','model','params','forward_macs',
                        'train_samples','val_samples','test_samples','parameter_mismatch','mac_mismatch','est_train_macs']}
                    s['seeds']=len(runs)
                    for metric in ['test_accuracy','test_loss','train_accuracy','train_loss','val_accuracy',
                                   'val_loss','train_seconds','run_seconds','inference_seconds','selected_epoch']:
                        vals=np.array([r[metric] for r in runs])
                        s[metric+'_mean']=float(vals.mean())
                        s[metric+'_sd']=float(vals.std(ddof=1))
                        s[metric+'_median']=float(np.median(vals))
                    summary.append(s)
    comparisons=[]
    for dataset in DATASETS:
        for ordering in ['native','permuted']:
            for width in (WIDTHS if ordering=='native' else [16,64]):
                for baseline in KINDS[1:]:
                    for metric in METRICS:
                        d=np.array([lookup[dataset,ordering,width,seed,'balanced'][metric]
                                   -lookup[dataset,ordering,width,seed,baseline][metric] for seed in range(10)])
                        c={'dataset':dataset,'ordering':ordering,'width':width,'baseline':baseline,'metric':metric,
                           **estimate(d),'balanced_wins':int(sum(d>0) if metric=='test_accuracy' else sum(d<0)),
                           'pairs':len(d),'p_holm':None}
                        comparisons.append(c)
    for metric in METRICS:
        holm([c for c in comparisons if c['ordering']=='native' and c['metric']==metric])
    slopes=[]
    for dataset in DATASETS:
        for baseline in KINDS[1:]:
            x=np.log2([lookup[dataset,'native',h,0,'balanced']['params'] for h in WIDTHS])
            x=x-x.mean()
            for metric in METRICS:
                delta=np.array([[lookup[dataset,'native',h,s,'balanced'][metric]
                                 -lookup[dataset,'native',h,s,baseline][metric] for h in WIDTHS] for s in range(10)])
                per_seed_slope=delta@x/(x@x)
                endpoint=estimate(delta[:,-1]-delta[:,0])
                slopes.append({'dataset':dataset,'baseline':baseline,'metric':metric,
                               **estimate(per_seed_slope),'endpoint_change':endpoint['mean'],
                               'endpoint_ci95_low':endpoint['ci95_low'],'endpoint_ci95_high':endpoint['ci95_high']})
    for metric in METRICS:
        holm([s for s in slopes if s['metric']==metric])
    controls=[]
    for dataset in DATASETS:
        for width in [16,64]:
            for baseline in KINDS[1:]:
                for metric in METRICS:
                    diffs=[]
                    for s in range(10):
                        adv=lambda o:lookup[dataset,o,width,s,'balanced'][metric]-lookup[dataset,o,width,s,baseline][metric]
                        diffs.append(adv('permuted')-adv('native'))
                    controls.append({'dataset':dataset,'width':width,'baseline':baseline,'metric':metric,**estimate(diffs)})
    write_csv(out/'summary.csv',summary)
    write_csv(out/'paired_comparisons.csv',comparisons)
    write_csv(out/'size_interactions.csv',slopes)
    write_csv(out/'feature_order_interactions.csv',controls)
    return summary,comparisons,slopes,controls


def figures(summary,comparisons,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    colors={'balanced':'#0072B2','random':'#E69F00','skewed':'#CC79A7','mlp':'#009E73'}
    labels={'balanced':'Balanced tree','random':'Random tree','skewed':'Skewed tree','mlp':'Dense MLP'}
    for metric,ylabel,name in [('test_accuracy','Test accuracy (%)','accuracy_by_size'),
                               ('test_loss','Test cross-entropy','loss_by_size'),
                               ('train_seconds','Training time (seconds)','time_by_size')]:
        fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharex=False,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for kind in KINDS:
                cells=sorted([r for r in summary if r['dataset']==dataset and r['ordering']=='native' and r['model']==kind],key=lambda r:r['width'])
                x=np.array([r['params'] for r in cells])
                y=np.array([r[metric+'_mean'] for r in cells])
                sd=np.array([r[metric+'_sd'] for r in cells])
                factor=100 if metric=='test_accuracy' else 1
                ax.plot(x,y*factor,'o-',label=labels[kind],color=colors[kind],lw=1.8,ms=4)
                ax.fill_between(x,(y-sd)*factor,(y+sd)*factor,color=colors[kind],alpha=.10)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3), numticks=8))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:g}K'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if metric=='train_seconds':
                ax.set_yscale('log')
            ax.set_title(TITLES[dataset],loc='left',weight='bold')
            ax.set_xlabel('Trainable parameters (log scale)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
        handles,legend=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='lower center',ncol=4,frameon=False)
        fig.suptitle({'test_accuracy':'Accuracy changes with both model size and task',
                      'test_loss':'Larger capacity does not guarantee lower held-out loss',
                      'train_seconds':'Equal arithmetic does not imply equal elapsed time'}[metric],fontsize=15,weight='bold')
        fig.text(.5,.045,'Mean ± one seed SD; 10 seeds; native feature order; 60 epochs per run',ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=[0,.065,1,.95])
        fig.savefig(out/(name+'.png'),dpi=170)
        fig.savefig(out/(name+'.svg'))
        plt.close(fig)
    for metric,ylabel,name in [('test_accuracy','Balanced − comparator accuracy (pp)','balance_advantage'),
                               ('test_loss','Balanced − comparator cross-entropy','balance_loss_advantage')]:
        fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharex=True,sharey=True)
        for ax,dataset in zip(axes.flat,DATASETS):
            for baseline in KINDS[1:]:
                cells=sorted([r for r in comparisons if r['dataset']==dataset and r['ordering']=='native'
                              and r['baseline']==baseline and r['metric']==metric],key=lambda r:r['width'])
                factor=100 if metric=='test_accuracy' else 1
                x=np.array([r['width'] for r in cells])
                y=np.array([r['mean'] for r in cells])*factor
                low=np.array([r['ci95_low'] for r in cells])*factor
                high=np.array([r['ci95_high'] for r in cells])*factor
                ax.plot(x,y,'o-',color=colors[baseline],label=labels[baseline])
                ax.fill_between(x,low,high,color=colors[baseline],alpha=.13)
            ax.axhline(0,color='#555555',lw=.8,ls='--')
            ax.set_xscale('log',base=2)
            ax.set_xticks(WIDTHS,labels=WIDTHS)
            ax.set_title(TITLES[dataset],loc='left',weight='bold')
            ax.set_xlabel('Tree hidden width (capacity setting)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
        handles,legend=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False)
        fig.suptitle('The effect of balance depends on size and comparator',fontsize=15,weight='bold')
        fig.text(.5,.045,'Paired seed means and pointwise bootstrap 95% intervals; intervals are not multiplicity-adjusted',ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=[0,.065,1,.95])
        fig.savefig(out/(name+'.png'),dpi=170)
        fig.savefig(out/(name+'.svg'))
        plt.close(fig)


def report(rows,summary,comparisons,slopes,controls,out):
    # This generated report deliberately leaves interpretation to the measured
    # tables; a readable FINDINGS.md can add a concise synthesis afterward.
    manifest=json.loads((out/'manifest.json').read_text())
    lines=['# Expanded balanced-recursion experiment: measured tables','',
           f"Completed {len(rows):,} runs; 4 datasets, 4 sizes, 10 paired seeds; native grid plus feature-order controls.",
           '',f"Total recorded elapsed time: {manifest['total_elapsed_seconds']/60:.1f} minutes.",
           '', 'All means below use validation-selected checkpoints. Main-grid models trained for 60 epochs; the supplementary control adjusts sample exposure.',
           'Accuracy ± values are sample SD across seeds, in percentage points.',
           'Paired intervals are pointwise percentile bootstrap 95% intervals over ten seed pairs.',
           'They are conditional on fixed test sets; validation splits overlap.', '',
           '## Native feature order: accuracy, loss and time','',
           '| Dataset | Width | Model | Parameters | Test accuracy (%) ± SD | Test loss ± SD | Train seconds ± SD |',
           '|---|---:|---|---:|---:|---:|---:|']
    for s in summary:
        if s['ordering']=='native':
            lines.append(f"| {s['dataset']} | {s['width']} | {s['model']} | {s['params']:,} | "
                         f"{s['test_accuracy_mean']*100:.2f} ± {s['test_accuracy_sd']*100:.2f} | "
                         f"{s['test_loss_mean']:.4f} ± {s['test_loss_sd']:.4f} | "
                         f"{s['train_seconds_mean']:.3f} ± {s['train_seconds_sd']:.3f} |")
    lines+=['','## Native paired accuracy effects','',
            'Positive differences favor balanced. Holm correction covers all 48 native cell comparisons.',
            '', '| Dataset | Width | Comparator | Balanced advantage (pp) | 95% interval (pp) | Wins/10 | Holm p |',
            '|---|---:|---|---:|---:|---:|---:|']
    for c in comparisons:
        if c['ordering']=='native' and c['metric']=='test_accuracy':
            lines.append(f"| {c['dataset']} | {c['width']} | {c['baseline']} | {100*c['mean']:+.2f} | "
                         f"[{100*c['ci95_low']:+.2f}, {100*c['ci95_high']:+.2f}] | {c['balanced_wins']} | {c['p_holm']:.4f} |")
    lines+=['','## Change of accuracy advantage with size','',
            'Slope is percentage points per doubling of tree parameter count; this is not a scaling exponent.',
            'Endpoint change is the width64 advantage minus the width8 advantage. Positive means balance gains relative accuracy with width.',
            'Holm correction covers 12 slope tests per metric.', '',
            '| Dataset | Comparator | Slope (pp/doubling) | Slope 95% interval | Width64−width8 change (pp) | Change 95% interval | Holm slope p |',
            '|---|---|---:|---:|---:|---:|---:|']
    for s in slopes:
        if s['metric']=='test_accuracy':
            lines.append(f"| {s['dataset']} | {s['baseline']} | {100*s['mean']:+.2f} | "
                         f"[{100*s['ci95_low']:+.2f}, {100*s['ci95_high']:+.2f}] | {100*s['endpoint_change']:+.2f} | "
                         f"[{100*s['endpoint_ci95_low']:+.2f}, {100*s['endpoint_ci95_high']:+.2f}] | {s['p_holm']:.4f} |")
    lines+=['','## Shuffled-feature control: paired accuracy effects','',
            'Secondary sensitivity analysis; these intervals and tests are not corrected for multiplicity.', '',
            '| Dataset | Width | Comparator | Balanced advantage (pp) | 95% interval (pp) | Wins/10 |',
            '|---|---:|---|---:|---:|---:|']
    for c in comparisons:
        if c['ordering']=='permuted' and c['metric']=='test_accuracy':
            lines.append(f"| {c['dataset']} | {c['width']} | {c['baseline']} | {100*c['mean']:+.2f} | "
                         f"[{100*c['ci95_low']:+.2f}, {100*c['ci95_high']:+.2f}] | {c['balanced_wins']} |")
    lines+=['','## Budget matching','',
            '| Dataset | Width | Tree parameters | Tree MACs/example | MLP parameters | MLP MACs/example | Parameter mismatch | MAC mismatch |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for h in WIDTHS:
            tree=next(s for s in summary if (s['dataset'],s['ordering'],s['width'],s['model'])==(dataset,'native',h,'balanced'))
            dense=next(s for s in summary if (s['dataset'],s['ordering'],s['width'],s['model'])==(dataset,'native',h,'mlp'))
            lines.append(f"| {dataset} | {h} | {tree['params']:,} | {tree['forward_macs']:,} | {dense['params']:,} | "
                         f"{dense['forward_macs']:,} | {dense['parameter_mismatch']*100:+.2f}% | {dense['mac_mismatch']*100:+.2f}% |")
    lines+=['','## Inference limits','',
            'The smallest attainable two-sided exact sign-flip p-value with ten pairs is 2/1024 = 0.001953.',
            'Across a family of 48 tests, Holm cannot reject even its first test at 0.05 with this minimum p-value.',
            'Consequently, cell-level Holm non-rejection here is not evidence that effects are absent.',
            'Pointwise intervals and consistent effect directions remain descriptive, not simultaneous confidence claims.',
            'The 12-test interaction family can resolve the smallest exact p-values; its power is still limited.',
            '', 'The sweep changes width and arithmetic together at fixed epochs. It does not estimate a compute-optimal scaling law.',
            'Four legacy datasets and fixed optimizer settings cannot establish general superiority across tasks.',
            'Topology changes path depths and receptive fields; a feature-order control cannot isolate these mechanisms completely.',
            'Native and shuffled conditions share test examples and seed splits and are not independent replications.',
            '', 'See PROTOCOL.md, README.md, FINDINGS.md, raw CSVs, checkpoints and predictions for the complete record.']
    (out/'MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')


def corrected_compute_audit(rows,out):
    checks=json.loads((out/'architecture_checks.json').read_text())
    dsizes={'optdigits':64,'pendigits':16,'spambase':57,'satimage':36}
    audit=[]
    for r in rows:
        c=next(c for c in checks if c['dataset']==r['dataset'] and c['width']==r['width'])
        d=dsizes[r['dataset']]
        tree_cost=3*c['tree_macs']-d*r['width']
        cost=3*r['forward_macs']-d*(c['dense_widths'][0] if r['model']=='mlp' else r['width'])
        audit.append({k:r[k] for k in ['dataset','ordering','width','seed','model']}|
                     {'linear_training_macs_per_example':cost,'linear_training_macs_per_run':cost*r['train_samples']*r['epochs'],
                      'tree_linear_training_macs_per_run':tree_cost*r['train_samples']*r['epochs'],
                      'corrected_linear_compute_mismatch':cost/tree_cost-1})
    write_csv(out/'corrected_training_budget_audit.csv',audit)
    return audit


def supplementary(rows,summary,folder,out):
    extra=json.loads((folder/'runs.json').read_text())
    assert len(extra)==160
    original={(r['dataset'],r['width'],r['seed'],r['model']):r for r in rows if r['ordering']=='native'}
    additional={(r['dataset'],r['width'],r['seed']):r for r in extra}
    cells,effects,slopes=[],[],[]
    for dataset in DATASETS:
        for h in WIDTHS:
            runs=[additional[dataset,h,s] for s in range(10)]
            cell={'dataset':dataset,'width':h,'params':runs[0]['params'],
                  'corrected_training_mac_mismatch_max':max(abs(r['training_mac_mismatch']) for r in runs)}
            for metric in ['test_accuracy','test_loss','train_seconds','epochs_equivalent','updates','selected_epoch']:
                vals=np.array([r[metric] for r in runs])
                cell[metric+'_mean']=float(vals.mean())
                cell[metric+'_sd']=float(vals.std(ddof=1))
            cells.append(cell)
            for metric in METRICS:
                d=np.array([original[dataset,h,s,'balanced'][metric]-additional[dataset,h,s][metric] for s in range(10)])
                effects.append({'dataset':dataset,'width':h,'baseline':'mlp_compute_matched','metric':metric,
                                **estimate(d),'balanced_wins':int(sum(d>0) if metric=='test_accuracy' else sum(d<0))})
        x=np.log2([original[dataset,h,0,'balanced']['params'] for h in WIDTHS])
        x=x-x.mean()
        for metric in METRICS:
            delta=np.array([[original[dataset,h,s,'balanced'][metric]-additional[dataset,h,s][metric]
                             for h in WIDTHS] for s in range(10)])
            e=estimate(delta[:,-1]-delta[:,0])
            slopes.append({'dataset':dataset,'metric':metric,**estimate(delta@x/(x@x)),
                           'endpoint_change':e['mean'],'endpoint_ci95_low':e['ci95_low'],'endpoint_ci95_high':e['ci95_high']})
    write_csv(folder/'summary.csv',cells)
    write_csv(folder/'paired_balanced_comparisons.csv',effects)
    write_csv(folder/'size_interactions.csv',slopes)
    lines=['','## Supplementary dense baseline matched on corrected training arithmetic','',
           '160 additional runs, specified before main-grid test evaluation. Total study: 1,120 training runs.',
           'Dense input layers have no input-gradient calculation, so corrected training MACs are 3×forward minus input-layer MACs.',
           'Dense sample counts are adjusted to match the tree total to the nearest example; data exposure and update counts differ.',
           'This accounts for linear forward/backward products, not optimizer, bias, activation or memory costs.',
           'See COMPUTE_CONTROL_PROTOCOL.md. Intervals below are pointwise/descriptive without multiplicity correction.','',
           '| Dataset | Width | Corrected-compute dense accuracy (%) | Dense loss | Dense equivalent epochs | Balanced−dense accuracy (pp) | 95% interval (pp) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        e=next(e for e in effects if e['dataset']==c['dataset'] and e['width']==c['width'] and e['metric']=='test_accuracy')
        lines.append(f"| {c['dataset']} | {c['width']} | {c['test_accuracy_mean']*100:.2f} ± {c['test_accuracy_sd']*100:.2f} | "
                     f"{c['test_loss_mean']:.4f} | {c['epochs_equivalent_mean']:.2f} | {100*e['mean']:+.2f} | "
                     f"[{100*e['ci95_low']:+.2f}, {100*e['ci95_high']:+.2f}] |")
    with (out/'MEASURED_RESULTS.md').open('a') as f:
        f.write('\n'.join(lines)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for kind,label,color in [('balanced','Balanced tree','#0072B2'),('mlp','Dense, fixed epochs','#009E73'),
                                 ('extra','Dense, matched training arithmetic','#555555')]:
            data=([c for c in cells if c['dataset']==dataset] if kind=='extra' else
                  [c for c in summary if c['dataset']==dataset and c['ordering']=='native' and c['model']==kind])
            data=sorted(data,key=lambda c:c['width'])
            ax.plot(WIDTHS,[100*c['test_accuracy_mean'] for c in data],'o-',color=color,label=label)
        ax.set_title(TITLES[dataset],loc='left',weight='bold')
        ax.set_xscale('log',base=2)
        ax.set_xticks(WIDTHS,labels=WIDTHS)
        ax.set_xlabel('Tree hidden width (capacity setting)')
        ax.set_ylabel('Test accuracy (%)')
        ax.grid(alpha=.18)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Dense comparison with a corrected training arithmetic budget',fontsize=15,weight='bold')
    fig.tight_layout(rect=[0,.04,1,.95])
    fig.savefig(out/'corrected_compute_comparison.png',dpi=170)
    fig.savefig(out/'corrected_compute_comparison.svg')
    plt.close(fig)
    return cells,effects,slopes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--compute-control',type=Path,default=ROOT/'compute_control')
    args=p.parse_args()
    out=args.results.resolve()
    rows=json.loads((out/'runs.json').read_text())
    summary,comparisons,slopes,controls=analyze(rows,out)
    figures(summary,comparisons,out)
    report(rows,summary,comparisons,slopes,controls,out)
    corrected_compute_audit(rows,out)
    if (args.compute_control/'runs.json').exists():
        supplementary(rows,summary,args.compute_control,out)
    if out==(ROOT/'results').resolve():
        (ROOT/'MEASURED_RESULTS.md').write_text((out/'MEASURED_RESULTS.md').read_text())
    (out/'analysis_metadata.json').write_text(json.dumps({'bootstrap_resamples':20000,'bootstrap_seed':20261015,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_cell_tests_per_metric':48,'native_slope_tests_per_metric':12},indent=2)+'\n')
    print('Saved summaries, paired effects, interactions, five figures and MEASURED_RESULTS.md.')


if __name__=='__main__':
    main()
