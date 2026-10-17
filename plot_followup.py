"""Plot all paired follow-up seed differences, means and descriptive intervals."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader((a.results/'paired.csv').open()));summary=json.loads((a.results/'summary.json').read_text())['primary_family']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(7,6.2),sharex=True,layout='constrained')
    for ax,name in zip(axes,['optdigits','pendigits']):
        x=np.array([float(r['difference'])*100 for r in rows if r['dataset']==name and r['ordering']=='native'])
        y=np.linspace(-.16,.16,len(x));m=summary[name+'_native'];lo,hi=np.array(m['descriptive_95pct_interval'])*100;mean=m['mean_difference']*100
        ax.axvline(0,color='#8a8a8a',lw=1);ax.scatter(x,y,color='#426c8d',s=28,alpha=.8,label='32 paired seeds')
        ax.plot([lo,hi],[-.3,-.3],color='#1a384f',lw=3);ax.scatter([mean],[-.3],s=75,marker='D',color='#1a384f')
        ax.set_title(f'{name.capitalize()} · mean {mean:+.2f} pp · Holm p={m["holm_p_value"]:.4f}',loc='left',fontsize=12)
        ax.set(yticks=[],ylim=(-.45,.28));ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    axes[-1].set_xlabel('Balanced − random test accuracy (percentage points)')
    fig.suptitle('A narrow topology effect across new training seeds',x=.02,ha='left',fontsize=15,fontweight='bold')
    fig.supxlabel('Dots: individual pairs. Diamond: mean. Line: descriptive 95% bootstrap interval.\nIntervals are pointwise; Holm correction applies to the four primary p values.\nPreviously inspected datasets · width 16 · 60 epochs · CPU reference',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(a.results/('paired-effects.'+suffix),dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
