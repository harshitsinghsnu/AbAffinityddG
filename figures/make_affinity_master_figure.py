"""
make_affinity_master_figure.py — consolidated panel of AbAffinity absolute-affinity (pKd) results.
==================================================================================================
  a  Architecture: AbAffinity vs fused two-stream vs concat+MLP, random & antigen-cold splits (SAAINT-DB)
  b  Backbone ablation: ESM-2 / ProtBERT / AntiBERTy / ProGen2 (Pearson + Spearman, error bars)
  c  External benchmarks: Pearson vs MVSF-AB (SAbDab, AB-Bind, SKEMPI 2.0, held-out)
  d  External benchmarks: RMSE vs MVSF-AB
  e  Integrated-gradients attribution + structural mapping, 1VFB (D1.3-lysozyme)
  f  Few-shot fine-tuning curves (Spearman vs label fraction) on 1mlc / trastuzumab / 4fqi
All values from the 3-stream results dir. Butter (AbAffinity/chosen) + lilac (alternatives).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
rcParams['font.family'] = 'DejaVu Sans'; rcParams['pdf.fonttype'] = 42; rcParams['ps.fonttype'] = 42
HERE = os.path.dirname(os.path.abspath(__file__))
BUTTER, LILAC, LILAC_L, LILAC_D, INK, MUTE, GRID = '#E8C547', '#9575CD', '#C6B3E3', '#5E3F9E', '#1a1a1a', '#666666', '#DDDDDD'
FS_T, FS_A, FS_TK, FS_V = 13.5, 12, 10, 9.5
EB = dict(ecolor='#333', capsize=3, elinewidth=1.1, capthick=1.1)

def style(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=FS_TK)
def plab(ax, l, s): ax.set_title(f'{l}  {s}', loc='left', fontsize=FS_T, fontweight='bold', color=INK, pad=5)

fig = plt.figure(figsize=(18, 9.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.26)

# ---- a: architecture across SAAINT-DB splits (Pearson; error bars on all three) ----
ax = fig.add_subplot(gs[0, 0]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
AB = [0.842, 0.600]; AB_SD = [0.022, 0.072]
TS = [0.815, 0.550]; TS_SD = [0.028, 0.112]
CC = [0.817, 0.549]; CC_SD = [0.033, 0.089]
x = np.arange(2); w = 0.26
b = ax.bar(x - w, AB, w, yerr=AB_SD, error_kw=EB, color=BUTTER, zorder=3, label='AbAffinity (tri-stream)')
for bb in b: bb.set_edgecolor(INK); bb.set_linewidth(1.4)
ax.bar(x, TS, w, yerr=TS_SD, error_kw=EB, color=LILAC, zorder=3, label='fused two-stream')
ax.bar(x + w, CC, w, yerr=CC_SD, error_kw=EB, color=LILAC_L, zorder=3, label='concat+MLP')
for xi, v in zip(x - w, AB): ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=FS_V - 1, color=INK)
ax.set_xticks(x); ax.set_xticklabels(['random', 'antigen-cold'], fontsize=FS_TK, color=MUTE)
ax.set_ylim(0, 1.0); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 2, loc='upper right', ncol=1)
plab(ax, 'a', 'Architecture comparison (SAAINT-DB)')

# ---- b: backbone / PLM ablation (Pearson + Spearman, error bars) ----
ax = fig.add_subplot(gs[0, 1]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
PLM = ['ESM-2', 'ProtBERT', 'AntiBERTy', 'ProGen2']
BP = [0.838, 0.819, 0.532, 0.412]; BP_SD = [0.028, 0.025, 0.111, 0.036]
BS = [0.822, 0.809, 0.517, 0.393]; BS_SD = [0.033, 0.027, 0.117, 0.030]
x = np.arange(4); w = 0.38
b = ax.bar(x - w/2, BP, w, yerr=BP_SD, error_kw=EB, color=BUTTER, zorder=3, label='Pearson')
b[0].set_edgecolor(INK); b[0].set_linewidth(1.5)
b2 = ax.bar(x + w/2, BS, w, yerr=BS_SD, error_kw=EB, color=LILAC, zorder=3, label='Spearman')
b2[0].set_edgecolor(INK); b2[0].set_linewidth(1.5)
ax.set_xticks(x); ax.set_xticklabels(PLM, fontsize=FS_TK - 1.5, color=MUTE, rotation=15, ha='right')
ax.set_ylim(0, 1.0); ax.set_ylabel('correlation', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper right', ncol=1)
plab(ax, 'b', 'Backbone ablation (ESM-2 best)')

# ---- c: external Pearson vs MVSF ----
EXT = ['SAbDab', 'AB-Bind', 'SKEMPI', 'held-out']
PCC_MVSF = [0.491, 0.739, 0.671, 0.467]; PCC_AB = [0.593, 0.781, 0.722, 0.551]; PCC_SD = [0.015, 0.006, 0.010, 0.056]
ax = fig.add_subplot(gs[0, 2]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
x = np.arange(4); w = 0.38
ax.bar(x - w/2, PCC_MVSF, w, color=LILAC, zorder=3, label='MVSF-AB')
b = ax.bar(x + w/2, PCC_AB, w, yerr=PCC_SD, error_kw=EB, color=BUTTER, zorder=3, label='AbAffinity')
for bb in b: bb.set_edgecolor(INK); bb.set_linewidth(1.3)
ax.set_xticks(x); ax.set_xticklabels(EXT, fontsize=FS_TK - 1, color=MUTE)
ax.set_ylim(0, 1.0); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper left', ncol=2)
plab(ax, 'c', 'External benchmarks vs MVSF-AB: Pearson')

# ---- d: external RMSE vs MVSF ----
RMSE_MVSF = [1.839, 1.905, 1.513, 1.447]; RMSE_AB = [1.326, 1.288, 1.021, 1.339]; RMSE_SD = [0.018, 0.009, 0.015, 0.075]
ax = fig.add_subplot(gs[1, 0]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
ax.bar(x - w/2, RMSE_MVSF, w, color=LILAC, zorder=3, label='MVSF-AB')
b = ax.bar(x + w/2, RMSE_AB, w, yerr=RMSE_SD, error_kw=EB, color=BUTTER, zorder=3, label='AbAffinity')
for bb in b: bb.set_edgecolor(INK); bb.set_linewidth(1.3)
ax.set_xticks(x); ax.set_xticklabels(EXT, fontsize=FS_TK - 1, color=MUTE)
ax.set_ylim(0, 2.1); ax.set_ylabel('RMSE ($pK_d$, lower better)', fontsize=FS_A - 1, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper right', ncol=2)
plab(ax, 'd', 'External benchmarks vs MVSF-AB: RMSE')

# ---- e: 1VFB IG attribution + structural mapping (image) ----
axe = fig.add_subplot(gs[1, 1]); axe.axis('off')
axe.imshow(mpimg.imread(os.path.join(HERE, 'ig_1vfb_composite.png')))
axe.set_title('e  IG attribution + structural mapping (1VFB)',
              loc='left', fontsize=FS_T, fontweight='bold', color=INK, pad=6)

# ---- f: few-shot fine-tuning curves (Spearman vs fraction) ----
ax = fig.add_subplot(gs[1, 2]); style(ax); ax.grid(True, color=GRID, lw=0.8, zorder=0)
fr = [0, 10, 20, 30]
curves = {'1mlc': ([0.166, 0.049, 0.296, 0.415], [0, 0.184, 0.112, 0.036], BUTTER),
          'trastuzumab': ([0.138, 0.207, 0.245, 0.304], [0, 0.160, 0.054, 0.041], LILAC_D),
          '4fqi': ([0.469, 0.871, 0.917, 0.950], [0, 0.019, 0.024, 0.008], LILAC)}
for name, (y, e, c) in curves.items():
    ax.errorbar(fr, y, yerr=e, fmt='-o', color=c, lw=2.4, ms=8, capsize=4, mec=INK, mew=1.0, label=name, zorder=3)
ax.set_xticks(fr); ax.set_xticklabels(['0', '10%', '20%', '30%'], fontsize=FS_TK, color=MUTE)
ax.set_ylim(-0.05, 1.0); ax.set_xlabel('fine-tuning label fraction', fontsize=FS_A - 1, color=INK)
ax.set_ylabel('within-target Spearman', fontsize=FS_A - 1, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 0.5, loc='center right')
plab(ax, 'f', 'Adaptation (few-shot fine-tuning)')

fig.suptitle('AbAffinity: absolute affinity ($pK_d$) results', fontsize=15.5, fontweight='bold', color=INK, x=0.015, ha='left', y=0.997)
for e in ('png', 'pdf'): fig.savefig(os.path.join(HERE, f'figure_affinity_results.{e}'), dpi=250, bbox_inches='tight')
print('wrote figure_affinity_results.png')
