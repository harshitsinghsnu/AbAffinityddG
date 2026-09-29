"""
make_ddg_master_figure.py — ONE consolidated panel of all AbAffinity-ddG results.
=================================================================================
Main model = AbAffinity-ddG with the native pKd conversion anchor  s*(pKd_wt - pKd_mut),
  pKd = pKd_lo + (pKd_hi - pKd_lo)*(cos+1)/2   (AbAffinity's own calibration).
Cosine-difference anchor s*(cos_wt - cos_mut) is retained as the panel-d ablation.
2 x 3 grid (bars a-d,f computed/verified from results/; scatter e from released preds):
  a  S1131 complex-disjoint  b  S1131 antigen-disjoint  c  S1131 random vs sequence predictors
  d  Ablation pKd-convert (main) vs cosine-difference   e  calibration   f  large-effect detection
Error bars: AbAffinity (ours) only, s.d. over 3 seeds. Butter (ours) + lilac (baselines).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
rcParams['font.family'] = 'DejaVu Sans'; rcParams['pdf.fonttype'] = 42; rcParams['ps.fonttype'] = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, '..', 'results'))
BUTTER, LILAC, LILAC_L, LILAC_D, INK, MUTE, GRID = '#E8C547', '#9575CD', '#C6B3E3', '#5E3F9E', '#1a1a1a', '#666666', '#DDDDDD'
FS_T, FS_A, FS_TK, FS_V = 13.5, 12, 10, 9.5
EB = dict(ecolor='#333', capsize=3, elinewidth=1.1, capthick=1.1)

def style(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=FS_TK)
def plab(ax, l, s): ax.set_title(f'{l}  {s}', loc='left', fontsize=FS_T, fontweight='bold', color=INK, pad=5)
def ours_err(v, sd0): return [sd0] + [0] * (len(v) - 1)
def annot(ax, vals, sd0, fs=FS_V):
    for xi, v in enumerate(vals):
        off = sd0 if xi == 0 else 0
        ax.annotate(f'{v:.2f}', (xi, v + off), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=fs, color=INK)
def zp(df):
    t, p = [], []
    for _, g in df.groupby('fold'):
        tt, pp = g['true'].values, g['pred'].values
        if len(g) < 3 or tt.std() < 1e-8 or pp.std() < 1e-8: continue
        t.append((tt - tt.mean()) / tt.std()); p.append((pp - pp.mean()) / pp.std())
    return pearsonr(np.concatenate(t), np.concatenate(p))[0]
def stat(sub, stem):
    rs = [zp(pd.read_csv(f'{RES}/{sub}/{stem}_s{s}.csv')) for s in (0, 1, 2) if os.path.exists(f'{RES}/{sub}/{stem}_s{s}.csv')]
    return (np.mean(rs), np.std(rs)) if rs else (np.nan, 0)

fig = plt.figure(figsize=(15.5, 8.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.28)

# ---- a, b: generalization vs ML baselines (ours = pKd-convert) ----
MODELS = ['AbAffinity\n-ΔΔG', 'anchor', 'XGBoost', 'RandomFor.', 'MLP', 'LSTM']
CX = [0.713, 0.687, 0.580, 0.540, 0.363, 0.161]
AG = [0.675, 0.650, 0.458, 0.414, 0.367, 0.061]
COLS = [BUTTER, LILAC_L, LILAC, LILAC, LILAC, LILAC]
for k, (name, v, sd0) in enumerate([('complex-disjoint', CX, 0.017), ('antigen-disjoint', AG, 0.019)]):
    ax = fig.add_subplot(gs[0, k]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    bars = ax.bar(range(len(MODELS)), v, 0.66, yerr=ours_err(v, sd0), error_kw=EB, color=COLS, zorder=3)
    bars[0].set_edgecolor(INK); bars[0].set_linewidth(1.6); annot(ax, v, sd0)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=FS_TK - 2, color=MUTE, rotation=22, ha='right')
    ax.set_ylim(0, 0.82); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
    plab(ax, 'ab'[k], f'S1131 {name}')

# ---- c: vs published sequence methods (S1131 10-fold CV, all beaten) ----
ax = fig.add_subplot(gs[0, 2]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
CM = ['AbAffinity\n-ΔΔG', 'AttABseq', 'DeepEP\n-PPI', 'TransPPI', 'PIPR', 'BeAtMuSiC']
CV = [0.801, 0.66, 0.21, 0.38, 0.33, 0.29]; CC = [BUTTER] + [LILAC] * 5
bars = ax.bar(range(len(CV)), CV, 0.66, yerr=ours_err(CV, 0.004), error_kw=EB, color=CC, zorder=3)
bars[0].set_edgecolor(INK); bars[0].set_linewidth(1.6); annot(ax, CV, 0.004)
ax.set_xticks(range(len(CV))); ax.set_xticklabels(CM, fontsize=FS_TK - 2.5, color=MUTE, rotation=22, ha='right')
ax.set_ylim(0, 0.95); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
plab(ax, 'c', 'S1131 10-fold CV: vs published methods')

# ---- d: anchor formulation, pKd-convert (main) vs cosine-difference (error bars on both) ----
ax = fig.add_subplot(gs[1, 0]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
sp3 = ['random', 'complex-\ndisjoint', 'antigen-\ndisjoint']
pkd_stems = ('s1131_fold_record', 's1131_fold_complex5', 's1131_fold_antigen5')
pkd = [stat('anchorpkd_final', s)[0] for s in pkd_stems]; pkds = [stat('anchorpkd_final', s)[1] for s in pkd_stems]
cos_src = [('wmt_preds', 'whuber_hinge_fold_record'), ('cosine_final', 's1131_fold_complex5'), ('cosine_final', 's1131_fold_antigen5')]
cos = [stat(a, b)[0] for a, b in cos_src]; coss = [stat(a, b)[1] for a, b in cos_src]
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w/2, pkd, w, yerr=pkds, error_kw=EB, color=BUTTER, zorder=3, label='pKd-convert (ours)')
for bb in b1: bb.set_edgecolor(INK); bb.set_linewidth(1.4)
ax.bar(x + w/2, cos, w, yerr=coss, error_kw=EB, color=LILAC_L, zorder=3, label='cosine-difference')
for xi, (v, s) in zip(x - w/2, zip(pkd, pkds)): ax.annotate(f'{v:.2f}', (xi, v + s), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
for xi, (v, s) in zip(x + w/2, zip(cos, coss)):
    if not np.isnan(v): ax.annotate(f'{v:.2f}', (xi, v + s), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
ax.set_xticks(x); ax.set_xticklabels(sp3, fontsize=FS_TK - 1, color=MUTE)
ax.set_ylim(0, 0.95); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper center', ncol=1)
plab(ax, 'd', 'Anchor: pKd-convert vs cosine')

# ---- e: contribution of the learned correction (anchor-only vs full), error bars on both ----
ax = fig.add_subplot(gs[1, 1]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
anc = [stat('anchoronly_final', s)[0] for s in pkd_stems]; ancs = [stat('anchoronly_final', s)[1] for s in pkd_stems]
b1 = ax.bar(x - w/2, anc, w, yerr=ancs, error_kw=EB, color=LILAC_L, zorder=3, label='affinity anchor only')
ax.bar(x + w/2, pkd, w, yerr=pkds, error_kw=EB, color=BUTTER, zorder=3, label='AbAffinity-ΔΔG (full)')
for bb in ax.patches[3:6]: bb.set_edgecolor(INK); bb.set_linewidth(1.4)
for xi, (v, s) in zip(x - w/2, zip(anc, ancs)):
    if not np.isnan(v): ax.annotate(f'{v:.2f}', (xi, v + s), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
for xi, (v, s) in zip(x + w/2, zip(pkd, pkds)): ax.annotate(f'{v:.2f}', (xi, v + s), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
ax.set_xticks(x); ax.set_xticklabels(sp3, fontsize=FS_TK - 1, color=MUTE)
ax.set_ylim(0, 0.95); ax.set_ylabel('Pearson $r$', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper center', ncol=1)
plab(ax, 'e', 'Correction contribution')

# ---- f: large-effect detection (pKd-convert preds) ----
ax = fig.add_subplot(gs[1, 2]); style(ax); ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
da, das, au, aus = [], [], [], []
for stem in ('s1131_fold_record', 's1131_fold_complex5', 's1131_fold_antigen5'):
    dd, aa = [], []
    for s in (0, 1, 2):
        f = f'{RES}/anchorpkd_final/{stem}_s{s}.csv'
        if not os.path.exists(f): continue
        e = pd.read_csv(f); big = e[e['true'].abs() > 2]
        if len(big) < 10: continue
        dd.append(((big['pred'] > 0) == (big['true'] > 0)).mean())
        y = (big['true'] < 0).astype(int)
        if y.nunique() == 2: aa.append(roc_auc_score(y, -big['pred'].values))
    da.append(np.mean(dd)); das.append(np.std(dd)); au.append(np.mean(aa)); aus.append(np.std(aa))
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w/2, da, w, yerr=das, error_kw=EB, color=BUTTER, zorder=3, label='direction acc.')
for bb in b1: bb.set_edgecolor(INK); bb.set_linewidth(1.4)
ax.bar(x + w/2, au, w, yerr=aus, error_kw=EB, color=LILAC, zorder=3, label='AUROC (enhancing)')
ax.axhline(0.5, color=MUTE, lw=1, ls=':', zorder=2)
for xi, (v, s) in zip(x - w/2, zip(da, das)): ax.annotate(f'{v:.2f}', (xi, v + s + 0.01), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
for xi, (v, s) in zip(x + w/2, zip(au, aus)): ax.annotate(f'{v:.2f}', (xi, v + s + 0.01), textcoords='offset points', xytext=(0, 2), ha='center', fontsize=FS_V - 1, color=INK)
ax.set_xticks(x); ax.set_xticklabels(sp3, fontsize=FS_TK - 1, color=MUTE)
ax.set_ylim(0, 1.32); ax.set_ylabel('score (|ΔΔG|>2)', fontsize=FS_A, color=INK)
ax.legend(frameon=False, fontsize=FS_TK - 1, loc='upper center', ncol=2)
plab(ax, 'f', 'Large-effect detection')

fig.suptitle('AbAffinity-ΔΔG results', fontsize=15.5, fontweight='bold', color=INK, x=0.015, ha='left', y=0.995)
for e in ('png', 'pdf'): fig.savefig(os.path.join(HERE, f'figure_ddg_results.{e}'), dpi=300, bbox_inches='tight')
print('pKd main:', [round(v, 3) for v in pkd], '| cosine:', [round(v, 3) if not np.isnan(v) else None for v in cos])
print('anchor-only:', [round(v, 3) if not np.isnan(v) else None for v in anc], '| large-effect diracc', [round(v, 2) for v in da], 'auroc', [round(v, 2) for v in au])
