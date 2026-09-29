"""
baselines_ddg.py — frozen-ESM2 simple-regressor controls + full metric suite, on rigorous folds.
================================================================================================
Reviewer controls: quantify how much of the signal comes from the pretrained embeddings alone by
regressing frozen ESM-2 (650M) features of wild-type and mutant sequences with simple models
(Ridge, RandomForest, XGBoost, small MLP). Same folds / preprocessing / metrics as our model.

Feature per mutation: mean-pooled ESM-2 embedding of each chain (H,L,Ag) for WT and MUT, assembled as
[emb_wt, emb_mut, emb_mut - emb_wt]  (captures identity + the mutation direction).

Metrics reported (per fold, pooled with per-fold z-scoring; mean +/- s.d. over --seeds):
  Pearson r, Spearman rho, RMSE, and DIRECTION accuracy (sign(ddG) classification, |ddG|>0.5 stratum),
  plus a stratification of error/correlation by number of mutated sites (for AB1101 multi-point).

Usage (embeds+caches ESM-2 once, then trains):
  python bench/baselines_ddg.py --pairs bench/attabseq_AB1101_rigorous.csv --fold_col fold_complex5 --seeds 0 1 2
"""
import os, argparse, pickle, hashlib
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DEVICE = 'cuda'
ESM = 'facebook/esm2_t33_650M_UR50D'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_esm2_meanpool_cache.pkl')
_emb = {}
if os.path.exists(CACHE):
    _emb = pickle.load(open(CACHE, 'rb'))
_model = {}


def esm2_meanpool(seqs):
    """Mean-pooled ESM-2 embedding (1280-d) per unique sequence, cached to disk."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    todo = sorted({s for s in seqs if s and s != 'nan' and s not in _emb})
    if todo:
        if 'tok' not in _model:
            _model['tok'] = AutoTokenizer.from_pretrained(ESM)
            _model['net'] = AutoModel.from_pretrained(ESM).half().to(DEVICE).eval()
        tok, net = _model['tok'], _model['net']
        for i in range(0, len(todo), 8):
            batch = todo[i:i + 8]
            t = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            ids, mask = t['input_ids'].to(DEVICE), t['attention_mask'].to(DEVICE)
            with torch.no_grad():
                h = net(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for s, v in zip(batch, pooled.cpu().numpy()):
                _emb[s] = v.astype(np.float32)
            if (i // 8) % 25 == 0:
                print(f'  embedded {i+len(batch)}/{len(todo)}', flush=True)
        pickle.dump(_emb, open(CACHE, 'wb'))
    z = np.zeros(1280, np.float32)
    return {s: _emb.get(s, z) for s in seqs}


def features(d):
    cols = ['wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen']
    allseq = pd.unique(d[cols].astype(str).values.ravel())
    emb = esm2_meanpool(list(allseq))
    def vec(row, prefix):
        return np.concatenate([emb[str(row[f'{prefix}_{c}'])] for c in ['heavy', 'light', 'antigen']])
    W = np.stack([vec(r, 'wt') for _, r in d.iterrows()])
    M = np.stack([vec(r, 'mut') for _, r in d.iterrows()])
    return np.concatenate([W, M, M - W], axis=1)


def nmut(r):
    n = 0
    for c in ['heavy', 'light', 'antigen']:
        w, m = str(r['wt_' + c]), str(r['mut_' + c])
        if w and m and w != 'nan' and len(w) == len(m):
            n += sum(a != b for a, b in zip(w, m))
    return n


def zpool(fold, y):
    y = np.asarray(y, float); out = np.zeros_like(y)
    for f in np.unique(fold):
        idx = fold == f; out[idx] = (y[idx] - y[idx].mean()) / (y[idx].std() + 1e-8)
    return out


def make(name, seed):
    if name == 'Ridge':  return Ridge(alpha=10.0)
    if name == 'RF':     return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == 'MLP':    return MLPRegressor(hidden_layer_sizes=(256,), max_iter=300, random_state=seed)
    if name == 'XGB':    return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                             subsample=0.8, n_jobs=-1, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True); ap.add_argument('--fold_col', default='fold_complex5')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    d = pd.read_csv(a.pairs).reset_index(drop=True)
    d['nmut'] = d.apply(nmut, axis=1)
    y = d['ddg'].values.astype(float); fold = d[a.fold_col].values
    print(f'{os.path.basename(a.pairs)} | n={len(d)} | fold={a.fold_col} '
          f'| folds={len(np.unique(fold))} | building ESM-2 features ...', flush=True)
    X = features(d)
    models = ['Ridge', 'RF', 'MLP'] + (['XGB'] if HAS_XGB else [])
    for name in models:
        pr_seeds = []
        for s in a.seeds:
            pred = np.zeros(len(d))
            for f in np.unique(fold):
                te = fold == f; tr = ~te
                m = make(name, s); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
            pr_seeds.append(pred)
        P = np.mean(pr_seeds, 0)
        zt, zp = zpool(fold, y), zpool(fold, P)
        r = pearsonr(zt, zp)[0]; rho = spearmanr(y, P)[0]; rmse = np.sqrt(np.mean((y - P) ** 2))
        big = np.abs(y) > 0.5
        dir_acc = np.mean(np.sign(P[big]) == np.sign(y[big])) if big.sum() else np.nan
        # per-seed s.d. of Pearson for error bars
        rs = [pearsonr(zpool(fold, y), zpool(fold, ps))[0] for ps in pr_seeds]
        line = f'  {name:6s} Pearson {r:.3f}±{np.std(rs):.3f} | Spearman {rho:.3f} | RMSE {rmse:.3f} | dir-acc {dir_acc:.3f}'
        # per-nmut stratification
        strata = []
        for grp, lab in [((d.nmut == 1), '1'), ((d.nmut > 1), '>1')]:
            if grp.sum() > 5:
                strata.append(f'{lab}:{pearsonr(y[grp], P[grp])[0]:.2f}(n={grp.sum()})')
        print(line + ('  | by #sites ' + ' '.join(strata) if strata else ''), flush=True)
    print('  [run our model on the SAME --fold_col via moe/mutsite.py, then compare]')


if __name__ == '__main__':
    main()
