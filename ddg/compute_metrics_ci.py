"""
compute_metrics_ci.py — full metric suite with error bars + 95% CI for the repurposed-AbAffinity ddG runs.
=========================================================================================================
For every saved dgsub prediction (results/rig/dgsub_<DS>_<SPLIT>.csv + per-seed _s0/_s1/_s2), computes:
  Pearson r, Spearman rho, RMSE, direction accuracy (sign, |ddG|>0.5), and per-#mutation-sites Pearson,
with mean +/- s.d. over the 3 seeds AND a 95% bootstrap CI (10k resamples) on the pooled Pearson.
Writes a persistent master table (CSV) so all experiment metrics are saved in one place.

Usage:  python eval/compute_metrics_ci.py
"""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # AbAffinity_ddG/code
BASE = os.path.dirname(ROOT)                                                  # AbAffinity_ddG
RIG = os.path.join(BASE, 'results', 'rig')
BENCH = os.path.join(ROOT, 'bench')
OUTDIR = os.path.join(BASE, 'paper', 'AbAffinityddG', 'results')
os.makedirs(OUTDIR, exist_ok=True)
DATASETS = ['AB645', 'AB1101', 'S1131']
SPLITS = ['fold_id', 'fold_complex5', 'fold_antigen5']
SPLABEL = {'fold_id': 'random', 'fold_complex5': 'complex-disjoint', 'fold_antigen5': 'antigen-disjoint'}


def nmut_row(r):
    n = 0
    for c in ['heavy', 'light', 'antigen']:
        w, m = str(r['wt_' + c]), str(r['mut_' + c])
        if w and m and w != 'nan' and len(w) == len(m):
            n += sum(a != b for a, b in zip(w, m))
    return n


def zpool(fold, y):
    y = np.asarray(y, float); o = np.zeros_like(y)
    for f in np.unique(fold):
        i = fold == f; o[i] = (y[i] - y[i].mean()) / (y[i].std() + 1e-8)
    return o


def boot_ci(y, p, fold, n=10000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); rs = []
    zt, zp = zpool(fold, y), zpool(fold, p)
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if np.std(zt[s]) > 0 and np.std(zp[s]) > 0:
            rs.append(pearsonr(zt[s], zp[s])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def metrics(m):
    y, p, fold = m['true'].values, m['pred'].values, m['fold'].values
    r = pearsonr(zpool(fold, y), zpool(fold, p))[0]
    rho = spearmanr(y, p)[0]; rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    big = np.abs(y) > 0.5; da = float(np.mean(np.sign(p[big]) == np.sign(y[big]))) if big.sum() else np.nan
    # CIR-DDG-style metrics: RMSE/MAE after univariate linear calibration, AUROC (ddG<0 = positive)
    a, b = np.polyfit(p, y, 1); pc = a * p + b
    rmse_cal = float(np.sqrt(np.mean((y - pc) ** 2))); mae_cal = float(np.mean(np.abs(y - pc)))
    lab = (y < 0).astype(int)
    auroc = float(roc_auc_score(lab, -p)) if 0 < lab.sum() < len(lab) else np.nan
    return r, rho, rmse, da, rmse_cal, mae_cal, auroc


def main():
    pairs = {}
    for ds in DATASETS:
        d = pd.read_csv(os.path.join(BENCH, f'attabseq_{ds}_rigorous.csv'))
        d['nmut'] = d.apply(nmut_row, axis=1)
        pairs[ds] = d[['complex', 'nmut']].drop_duplicates('complex')
    rows = []
    for ds in DATASETS:
        for sp in SPLITS:
            base = os.path.join(RIG, f'dgsub_{ds}_{sp}.csv')
            if not os.path.exists(base):
                continue
            mean = pd.read_csv(base).merge(pairs[ds], on='complex', how='left')
            r, rho, rmse, da, rmse_cal, mae_cal, auroc = metrics(mean)
            lo, hi = boot_ci(mean['true'].values, mean['pred'].values, mean['fold'].values)
            # per-seed spread
            seed_r = []
            for s in (0, 1, 2):
                f = os.path.join(RIG, f'dgsub_{ds}_{sp}_s{s}.csv')
                if os.path.exists(f):
                    ms = pd.read_csv(f); seed_r.append(pearsonr(zpool(ms['fold'].values, ms['true'].values),
                                                                zpool(ms['fold'].values, ms['pred'].values))[0])
            sd = float(np.std(seed_r)) if seed_r else np.nan
            # per-#sites
            m1 = mean[mean.nmut == 1]; mm = mean[mean.nmut > 1]
            r_single = pearsonr(m1['true'], m1['pred'])[0] if len(m1) > 5 else np.nan
            r_multi = pearsonr(mm['true'], mm['pred'])[0] if len(mm) > 5 else np.nan
            rows.append(dict(model='AbAffinity-ddG (dgsub)', dataset=ds, split=SPLABEL[sp], n=len(mean),
                             pearson=round(r, 3), pearson_sd=round(sd, 3) if sd == sd else None,
                             pearson_ci95=f'[{lo:.3f}, {hi:.3f}]', spearman=round(rho, 3),
                             rmse=round(rmse, 3), rmse_cal=round(rmse_cal, 3), mae_cal=round(mae_cal, 3),
                             auroc=round(auroc, 3) if auroc == auroc else None,
                             dir_acc=round(da, 3) if da == da else None,
                             pearson_single=round(r_single, 3) if r_single == r_single else None,
                             pearson_multi=round(r_multi, 3) if r_multi == r_multi else None))
    df = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, 'master_results_dgsub.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print('\nsaved ->', out)


if __name__ == '__main__':
    main()
