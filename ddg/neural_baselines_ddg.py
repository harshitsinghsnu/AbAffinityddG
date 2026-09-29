"""
neural_baselines_ddg.py — parameter-matched simple sequence modules on frozen ESM-2, rigorous folds.
====================================================================================================
Reviewer control: does the proposed architecture beat a *parameter-matched* simple sequence module?
Each mutation is a length-6 token sequence of mean-pooled ESM-2 (650M) chain embeddings
  [wt_H, wt_L, wt_Ag, mut_H, mut_L, mut_Ag]  (each 1280-d),
processed by a small MLP / CNN / LSTM / lightweight Transformer (all ~similar parameter count) that
predicts ddG. Same folds / metrics as our model and the RF/XGBoost controls (baselines_ddg.py):
Pearson (mean+/-s.d. over seeds), Spearman, RMSE, direction accuracy, per-#mutation-sites.

Usage:
  python bench/neural_baselines_ddg.py --pairs bench/attabseq_AB1101_rigorous.csv --fold_col fold_complex5 --seeds 0 1 2
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines_ddg import esm2_meanpool, nmut, zpool          # reuse cached ESM-2 mean-pool + helpers

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D_IN, H = 1280, 128


class MLP(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.Flatten(), nn.Linear(6 * D_IN, H), nn.GELU(), nn.Dropout(0.2), nn.Linear(H, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv1d(D_IN, H, 3, padding=1); s.c2 = nn.Conv1d(H, H, 3, padding=1)
        s.act = nn.GELU(); s.head = nn.Linear(H, 1)
    def forward(s, x):                                          # x: (B,6,1280)
        h = s.act(s.c1(x.transpose(1, 2))); h = s.act(s.c2(h)); return s.head(h.mean(-1)).squeeze(-1)


class LSTMr(nn.Module):
    def __init__(s): super().__init__(); s.lstm = nn.LSTM(D_IN, H, batch_first=True, bidirectional=True); s.head = nn.Linear(2 * H, 1)
    def forward(s, x): o, _ = s.lstm(x); return s.head(o.mean(1)).squeeze(-1)


class TF(nn.Module):
    def __init__(s):
        super().__init__(); s.proj = nn.Linear(D_IN, H)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(H, 4, H * 2, dropout=0.1, batch_first=True), 2)
        s.head = nn.Linear(H, 1)
    def forward(s, x): return s.head(s.enc(s.proj(x)).mean(1)).squeeze(-1)


MODELS = {'MLP': MLP, 'CNN': CNN, 'LSTM': LSTMr, 'Transformer': TF}


def features(d):
    cols = ['wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen']
    emb = esm2_meanpool(list(pd.unique(d[cols].astype(str).values.ravel())))
    z = np.zeros(D_IN, np.float32)
    X = np.stack([np.stack([emb.get(str(r[c]), z) for c in cols]) for _, r in d.iterrows()])  # (N,6,1280)
    return X.astype(np.float32)


def train_eval(Model, Xtr, ytr, Xte, seed, epochs=120):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    n = len(Xtr_t)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]; opt.zero_grad()
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte, device=DEV)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True); ap.add_argument('--fold_col', default='fold_complex5')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    d = pd.read_csv(a.pairs).reset_index(drop=True); d['nmut'] = d.apply(nmut, axis=1)
    y = d['ddg'].values.astype(np.float32); fold = d[a.fold_col].values
    print(f'{os.path.basename(a.pairs)} | n={len(d)} | fold={a.fold_col} '
          f'| folds={len(np.unique(fold))} | ESM-2 features ...', flush=True)
    X = features(d)
    for name, Model in MODELS.items():
        pr_seeds = []
        for s in a.seeds:
            pred = np.zeros(len(d), np.float32)
            for f in np.unique(fold):
                te = fold == f; tr = ~te
                pred[te] = train_eval(Model, X[tr], y[tr], X[te], s)
            pr_seeds.append(pred)
        P = np.mean(pr_seeds, 0)
        rs = [pearsonr(zpool(fold, y), zpool(fold, ps))[0] for ps in pr_seeds]
        rho = spearmanr(y, P)[0]; rmse = float(np.sqrt(np.mean((y - P) ** 2)))
        big = np.abs(y) > 0.5; dacc = float(np.mean(np.sign(P[big]) == np.sign(y[big]))) if big.sum() else np.nan
        strata = [f'{lab}:{pearsonr(y[g], P[g])[0]:.2f}(n={int(g.sum())})'
                  for g, lab in [((d.nmut == 1).values, '1'), ((d.nmut > 1).values, '>1')] if g.sum() > 5]
        print(f'  {name:11s} Pearson {np.mean(rs):.3f}±{np.std(rs):.3f} | Spearman {rho:.3f} | '
              f'RMSE {rmse:.3f} | dir-acc {dacc:.3f}' + ('  | #sites ' + ' '.join(strata) if strata else ''), flush=True)


if __name__ == '__main__':
    main()
