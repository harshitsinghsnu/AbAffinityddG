"""
run_ddg_variants.py
===================
Repurpose AbAffinity (an ABSOLUTE pK_d model) for ddG and benchmark several
repurposing strategies on Graphinity's Experimental_ddG_645 set, under the SAME
10-fold CV splits used everywhere else. One command runs all variants and saves
per-variant fold results + a combined summary + a full log.

Variants
--------
V0  physical-dG (no training):   score->pKd->dG per complex, ddG = dG_mut - dG_wt.
                                  Deterministic; correlation == frozen score-diff.
V1  siamese-cdr (fine-tune):     ddG = scale*(score_wt - score_mut) + bias, whole
                                  model fine-tuned. Heavy uses CDR pooling (as core
                                  was trained). This is the current reference (~0.25).
V2  mutation-local (fine-tune):  same siamese head, but the MUTATED chain is pooled
                                  over a +/-k window around the mutated residue
                                  (instead of whole-CDR / mean) so the single-residue
                                  change is not diluted.
V3  mutation-delta (fine-tune):  V1 pooling PLUS an explicit feature: the WT->mut
                                  per-residue embedding delta at the mutated position,
                                  passed through a small MLP and added to the ddG head.

Fair vs Graphinity: same dataset, same folds, experimental-only, target = ddG.

Usage (one command; logs+results saved):
  python run_ddg_variants.py --cutoffs random --variants V0 V1 V2 V3 \
         > results_ddg_variants/run.log 2>&1
"""
import os, sys, argparse, pickle, time, re
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))                      # ddg/
sys.path.insert(0, HERE)
from mutual_strong import MutualTriStreamStrong                         # noqa: E402

PKG_ROOT = os.path.dirname(HERE)                                        # package root
# All of these are overridable via env vars; defaults assume the standard
# package layout (checkpoints/model.pt, data/*.csv) documented in README.md.
GC = os.environ.get('ABAFF_DATA', os.path.join(PKG_ROOT, 'data'))
SEQS = os.path.join(GC, 'graphinity645_sequences.csv')
SPLITDIR = os.path.join(GC, 'ddg_experimental', 'Experimental_ddG_645',
                        'cdr_seqid_cutoffs', 'Experimental_ddG_645_-Reverse_Mutations_+Non_Binders')
PRETRAINED = os.environ.get('ABAFF_CKPT', os.path.join(PKG_ROOT, 'checkpoints', 'model.pt'))
TOKCACHE = os.path.join(GC, 'esm2_token_cache_650M.pkl')   # auto-created/regenerated on first run
OUT = os.path.join(PKG_ROOT, 'results_ddg_variants')
ESM = 'facebook/esm2_t33_650M_UR50D'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PKD_LOWER, PKD_UPPER = 4.33, 13.47              # SAaIntDB pK_d range (for V0 units; correlation-invariant)
RT_LN10 = 1.987e-3 * 298.15 * 2.302585          # ~1.364 kcal/mol per pKd unit
WINDOW_K = 7                                     # +/- residues for mutation-local pooling


# ---------------------------------------------------------- heavy CDR indices
def cdr_indices(seq):
    cdr = []
    m = re.search(r'C[A-Z]{2,5}[SAGTV]([A-Z]{8,15})WVRQ', seq); cdr += list(range(*m.span(1))) if m else list(range(26, 36))
    m = re.search(r'W[VI]RQ[A-Z]{6,14}W[VL][AS]([A-Z]{10,20})VKGRF', seq); cdr += list(range(*m.span(1))) if m else list(range(50, 64))
    m = re.search(r'WYYCA([A-Z]+)WGQGT', seq) or re.search(r'WYYC[A-Z]([A-Z]+)(?:WGQGT|WGQG)', seq)
    cdr += list(range(*m.span(1))) if m else list(range(95, 110))
    return sorted(set(i for i in cdr if i < len(seq)))


# ---------------------------------------------------------- token embeddings
def embed_tokens(seqs):
    cache = pickle.load(open(TOKCACHE, 'rb')) if os.path.exists(TOKCACHE) else {}
    need = [s for s in seqs if isinstance(s, str) and s and s not in cache]
    if need:
        tok = AutoTokenizer.from_pretrained(ESM); esm = AutoModel.from_pretrained(ESM).to(DEVICE).eval()
        print(f'embedding {len(need)} unique sequences (token-level) on {DEVICE} ...', flush=True)
        for i, s in enumerate(need):
            with torch.no_grad():
                t = tok(s[:1022], return_tensors='pt').to(DEVICE)
                per = esm(**t).last_hidden_state.squeeze(0)[1:-1]         # (L,1280) strip cls/sep
            cache[s] = per.cpu().float().numpy().astype(np.float16)
            if (i + 1) % 100 == 0:
                pickle.dump(cache, open(TOKCACHE, 'wb')); print(f'  {i+1}/{len(need)}', flush=True)
        pickle.dump(cache, open(TOKCACHE, 'wb')); del esm; torch.cuda.empty_cache()
    return cache


def mut_chain_pos(r):
    """Return (chain in {'H','L','A'}, position) of the single substitution, else (None,None)."""
    for ch, wcol, mcol in (('H', 'wt_heavy', 'mut_heavy'), ('L', 'wt_light', 'mut_light'), ('A', 'wt_antigen', 'mut_antigen')):
        w, m = r[wcol], r[mcol]
        if not isinstance(w, str) or not isinstance(m, str) or w == m or len(w) != len(m):
            continue
        d = [i for i in range(len(w)) if w[i] != m[i]]
        if d:
            return ch, d[0]
    return None, None


def pool_cdr(tok, seq):  return tok[cdr_indices(seq)].mean(0) if len(cdr_indices(seq)) else tok.mean(0)
def pool_mean(tok):      return tok.mean(0)
def pool_window(tok, p):
    p = min(max(0, p), len(tok) - 1)                       # clamp into token range (seq truncated at 1022)
    w = tok[max(0, p - WINDOW_K):p + WINDOW_K + 1]
    return w.mean(0) if len(w) else tok.mean(0)


def build_features(df, cache, variant):
    """Return dict idx -> (wt_l, wt_h, wt_a, mut_l, mut_h, mut_a[, delta]) as float32 tensors."""
    feats = {}
    for idx, r in df.iterrows():
        wt_h, wt_l, wt_a = cache[r['wt_heavy']], cache[r['wt_light']], cache[r['wt_antigen']]
        mu_h, mu_l, mu_a = cache[r['mut_heavy']], cache[r['mut_light']], cache[r['mut_antigen']]
        ch, pos = mut_chain_pos(r)
        # default pooling (V0/V1/V3): heavy=CDR, light/antigen=mean
        wl, wh, wa = pool_mean(wt_l), pool_cdr(wt_h, r['wt_heavy']), pool_mean(wt_a)
        ml, mh, ma = pool_mean(mu_l), pool_cdr(mu_h, r['mut_heavy']), pool_mean(mu_a)
        if variant in ('V2', 'V4') and ch is not None:   # mutation-local pooling on the mutated chain
            if ch == 'H': wh, mh = pool_window(wt_h, pos), pool_window(mu_h, pos)
            elif ch == 'L': wl, ml = pool_window(wt_l, pos), pool_window(mu_l, pos)
            elif ch == 'A': wa, ma = pool_window(wt_a, pos), pool_window(mu_a, pos)
        row = [wl, wh, wa, ml, mh, ma]
        if variant in ('V3', 'V4'):                       # explicit WT->mut residue delta at the site
            tokmap = {'H': (wt_h, mu_h), 'L': (wt_l, mu_l), 'A': (wt_a, mu_a)}
            if ch in tokmap:
                wt_t, mu_t = tokmap[ch]
                p = min(pos, len(wt_t) - 1, len(mu_t) - 1)   # clamp (seq truncated at 1022)
                delta = (mu_t[p] - wt_t[p]).astype(np.float32) if p >= 0 else np.zeros(1280, np.float32)
            else:
                delta = np.zeros(1280, np.float32)
            row.append(delta)
        feats[idx] = tuple(torch.tensor(np.asarray(x), dtype=torch.float32) for x in row)
    return feats


# ---------------------------------------------------------- models
class SiameseDDG(nn.Module):
    def __init__(self, pretrained=True, delta=False):
        super().__init__()
        self.core = MutualTriStreamStrong(esm_dim=1280, projected_size=256, num_heads=8, dropout=0.1, n_layers=2)
        if pretrained and os.path.exists(PRETRAINED):
            self.core.load_state_dict(torch.load(PRETRAINED, map_location='cpu')['model_state_dict'], strict=False)
        self.scale = nn.Parameter(torch.tensor(4.0)); self.bias = nn.Parameter(torch.tensor(0.0))
        self.delta = delta
        if delta:
            self.delta_mlp = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Linear(128, 1))

    def score(self, l, h, a): return self.core(l, h, a)['cosine_similarity']

    def forward(self, batch):
        wl, wh, wa, ml, mh, ma = batch[:6]
        out = self.scale * (self.score(wl, wh, wa) - self.score(ml, mh, ma)) + self.bias
        if self.delta:
            out = out + self.delta_mlp(batch[6]).squeeze(-1)
        return out


class ConcatHeadDDG(nn.Module):
    """Approach-2: keep the tri-stream backbone, take its final antibody+antigen
    embeddings for WT and mutant, and regress ddG from [ab_w, ag_w, ab_m, ag_m,
    ab_w-ab_m, ag_w-ag_m] through a fresh MLP head (no cosine)."""
    def __init__(self, pretrained=True, d=256):
        super().__init__()
        self.core = MutualTriStreamStrong(esm_dim=1280, projected_size=d, num_heads=8, dropout=0.1, n_layers=2)
        if pretrained and os.path.exists(PRETRAINED):
            self.core.load_state_dict(torch.load(PRETRAINED, map_location='cpu')['model_state_dict'], strict=False)
        self.head = nn.Sequential(nn.Linear(6 * d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def rep(self, l, h, a):
        o = self.core(l, h, a); return o['antibody_context'], o['antigen_context']

    def forward(self, batch):
        wl, wh, wa, ml, mh, ma = batch[:6]
        ab_w, ag_w = self.rep(wl, wh, wa); ab_m, ag_m = self.rep(ml, mh, ma)
        feat = torch.cat([ab_w, ag_w, ab_m, ag_m, ab_w - ab_m, ag_w - ag_m], dim=-1)
        return self.head(feat).squeeze(-1)


def stack(idxs, feats, ncols):
    cols = [torch.stack([feats[i][c] for i in idxs]).to(DEVICE) for c in range(ncols)]
    return cols


def run_variant(variant, df, feats, ddg, splits, epochs, log):
    ncols = 7 if variant in ('V3', 'V4') else 6
    fold_r = []; P_all, T_all = [], []
    for f in sorted(df['fold_id'].unique()):
        te = df[df.fold_id == f].index.tolist(); tr = df[df.fold_id != f].index.tolist()
        if variant == 'V0':                     # deterministic, no training
            m = SiameseDDG(pretrained=True).to(DEVICE).eval()
            with torch.no_grad():
                X = stack(te, feats, 6)
                s_wt = m.score(X[0], X[1], X[2]).cpu().numpy(); s_mu = m.score(X[3], X[4], X[5]).cpu().numpy()
            pkd_wt = (s_wt + 1) / 2 * (PKD_UPPER - PKD_LOWER) + PKD_LOWER
            pkd_mu = (s_mu + 1) / 2 * (PKD_UPPER - PKD_LOWER) + PKD_LOWER
            pred = RT_LN10 * (pkd_wt - pkd_mu)   # = dG_mut - dG_wt (kcal/mol)
        else:
            torch.manual_seed(0)
            m = (ConcatHeadDDG(pretrained=True) if variant == 'V5'
                 else SiameseDDG(pretrained=True, delta=(variant in ('V3', 'V4')))).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); lossf = nn.MSELoss()
            tr_sh = list(tr); nval = max(8, int(0.15 * len(tr_sh))); np.random.seed(0); np.random.shuffle(tr_sh)
            va, trn = tr_sh[:nval], tr_sh[nval:]
            Xva = stack(va, feats, ncols); yva = torch.tensor(ddg[va], dtype=torch.float32, device=DEVICE)
            best, best_state, bad = 1e9, None, 0
            for ep in range(epochs):
                m.train(); order = list(trn); np.random.shuffle(order)
                for i in range(0, len(order), 32):
                    b = order[i:i + 32]; X = stack(b, feats, ncols); y = torch.tensor(ddg[b], dtype=torch.float32, device=DEVICE)
                    opt.zero_grad(); loss = lossf(m(X), y); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad(): vloss = lossf(m(Xva), yva).item()
                if vloss < best - 1e-4: best, best_state, bad = vloss, {k: v.clone() for k, v in m.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 12: break
            if best_state: m.load_state_dict(best_state)
            m.eval()
            with torch.no_grad(): pred = m(stack(te, feats, ncols)).cpu().numpy()
        true = ddg[te]
        pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        fin = np.isfinite(true) & np.isfinite(pred)
        r = pearsonr(true[fin], pred[fin])[0] if fin.sum() > 2 else np.nan
        fold_r.append(r); P_all += list(pred[fin]); T_all += list(true[fin])
        print(f'  [{variant}] fold {f}: n={len(te)} r={r:.3f}', flush=True)
    P_all, T_all = np.array(P_all), np.array(T_all)
    pooled = pearsonr(T_all, P_all)[0]; rho = spearmanr(T_all, P_all)[0]
    rmse = float(np.sqrt(np.mean((T_all - P_all) ** 2)))
    mf, sd = float(np.nanmean(fold_r)), float(np.nanstd(fold_r))
    print(f'[{variant}] pooled r={pooled:.3f} | mean-fold {mf:.3f}±{sd:.3f} | rho={rho:.3f} | rmse={rmse:.3f}', flush=True)
    return dict(variant=variant, pooled_pearson=pooled, meanfold_pearson=mf, meanfold_std=sd,
                spearman=rho, rmse=rmse, n=len(T_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', nargs='+', default=['random'])
    ap.add_argument('--variants', nargs='+', default=['V0', 'V1', 'V2', 'V3', 'V4', 'V5'])  # V4=V2+V3, V5=concat/diff head
    ap.add_argument('--epochs', type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    seqs = pd.read_csv(SEQS).drop_duplicates('complex')
    uniq = set()
    for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
        uniq |= set(seqs[c].dropna().tolist())
    cache = embed_tokens(list(uniq))

    summary = []
    for cut in a.cutoffs:
        sp = pd.read_csv(os.path.join(SPLITDIR, f'Experimental_ddG_645_-Reverse_Mutations_+Non_Binders-cutoff_{cut}-10foldcv.csv'))
        d = sp.merge(seqs, on='complex', how='inner').dropna(subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
        ddg = d['ddg'].values.astype(np.float32)
        print(f'\n==== cutoff {cut}: {len(d)} mutations, {d.fold_id.nunique()} folds ====', flush=True)
        # build features once per variant (pooling differs)
        for v in a.variants:
            t0 = time.time()
            feats = build_features(d, cache, v)
            res = run_variant(v, d, feats, ddg, None, a.epochs, None); res['cutoff'] = cut
            res['sec'] = round(time.time() - t0)
            summary.append(res)
            pd.DataFrame(summary).to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    df = pd.DataFrame(summary)
    print('\n================ SUMMARY (ddG-645) ================')
    print(df[['cutoff', 'variant', 'pooled_pearson', 'meanfold_pearson', 'meanfold_std', 'spearman', 'rmse']].to_string(index=False))
    print('\nreference on same task: Graphinity-EGNN experimental-only random = 0.440 ± 0.098 (mean-fold)')
    print(f'saved -> {os.path.join(OUT, "summary.csv")}')


if __name__ == '__main__':
    main()
