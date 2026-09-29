"""
run_ddg_seq_improved.py  (v2)
=============================
PURE-SEQUENCE ddG repurposing of AbAffinity (no structure). Extends V5 with:

  #1  ESM-2 masked-LM mutation signal (STRENGTHENED): four antibody-agnostic scalars
        - mm_llr   : masked-marginal  Δlog P(mut) − Δlog P(wt) at the site
        - wt_llr   : wildtype-marginal Δlog P(mut) − Δlog P(wt) at the site
        - entropy  : masked-marginal entropy at the site (local uncertainty)
        - win_cons : mean masked/wt log P(wt residue) over a ±3 window (local conservation)
  #2  Antibody-agnostic descriptors: Δcharge, Δhydropathy, Δvolume, BLOSUM62, in-CDR, is-antibody.
  #4  Neighborhood self-attention over the ±7 local ESM-2 window (variant V7).
  #5  --freeze (head-only) and --loss {mse,rank}.
  NEW  --nseeds N   : ensemble N seeds per fold (avg test preds) to cut cutoff variance.
  NEW  --pretrain_csv PATH : pretrain the head on a broad ddG set (e.g. SKEMPI-general PPI)
                             then fine-tune on ddG-645 folds.  (#3, generalization lever)
  NEW  summary CSV is APPEND-mode (all configs accumulate) with a config tag.

Variants: V6 = V5 head + #1 + #2 ; V7 = V6 + #4.
n_extra = 4 (MLM) + 4 (descriptors) + 2 (flags) = 10.
"""
import os, sys, argparse, pickle, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import EsmForMaskedLM, AutoTokenizer
from Bio.Align import substitution_matrices

import run_ddg_variants as rv

HERE, GC, SEQS, SPLITDIR = rv.HERE, rv.GC, rv.SEQS, rv.SPLITDIR
PRETRAINED, ESM, DEVICE = rv.PRETRAINED, rv.ESM, rv.DEVICE
OUT = os.path.join(GC, 'results_ddg_variants')
MLM_CACHE = os.path.join(GC, 'mlm_feats_cache.pkl')
WIN_K, N_EXTRA = 7, 10
AA20 = set('ACDEFGHIKLMNPQRSTVWY')

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2,
           I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(D=-1., E=-1., K=1., R=1., H=0.1)
BL = substitution_matrices.load('BLOSUM62')


def phys(wt, mut):
    g = lambda t, a: t.get(a, 0.0)
    try: bl = BL[(wt, mut)]
    except Exception: bl = 0.0
    return [g(CHG, mut) - g(CHG, wt), (g(KD, mut) - g(KD, wt)) / 4.5,
            (g(VOL, mut) - g(VOL, wt)) / 50.0, bl / 4.0]


# ---- #1 strengthened ESM-2 MLM features (cached) ----
def compute_mlm(df):
    cache = pickle.load(open(MLM_CACHE, 'rb')) if os.path.exists(MLM_CACHE) else {}
    def seqs_of(r, ch):
        w = {'H': r['wt_heavy'], 'L': r['wt_light'], 'A': r['wt_antigen']}[ch]
        m = {'H': r['mut_heavy'], 'L': r['mut_light'], 'A': r['mut_antigen']}[ch]
        return w, m
    keys, info = {}, {}
    for idx, r in df.iterrows():
        ch, pos = rv.mut_chain_pos(r); info[idx] = (ch, pos)
        if ch is None: continue
        w, m = seqs_of(r, ch)
        if not isinstance(w, str) or pos >= min(len(w), 1021): continue
        k = (w[:1021], pos, w[pos], m[pos]); keys[idx] = k
        if k not in cache: keys.setdefault('_todo', {})[k] = True
    todo = list(keys.get('_todo', {}))
    if todo:
        tok = AutoTokenizer.from_pretrained(ESM); mlm = EsmForMaskedLM.from_pretrained(ESM).to(DEVICE).eval()
        aa_ids = {a: tok.convert_tokens_to_ids(a) for a in AA20}
        print(f'MLM (4-feat) scoring {len(todo)} mutations ...', flush=True)
        for i, (seq, pos, wt_aa, mut_aa) in enumerate(todo):
            with torch.no_grad():
                ids = tok(seq, return_tensors='pt').to(DEVICE)
                # unmasked forward -> wt_llr + window conservation
                lp = F.log_softmax(mlm(**ids).logits[0], dim=-1)                 # (L+2, V)
                wid, mid = aa_ids[wt_aa], aa_ids[mut_aa] if mut_aa in aa_ids else aa_ids['A']
                wt_llr = float(lp[pos + 1, mid] - lp[pos + 1, wid])
                w = 3; js = range(max(0, pos - w), min(len(seq), pos + w + 1))
                win_cons = float(np.mean([float(lp[j + 1, aa_ids.get(seq[j], wid)]) for j in js]))
                # masked forward -> mm_llr + entropy
                ids['input_ids'][0, pos + 1] = tok.mask_token_id
                lo = mlm(**ids).logits[0, pos + 1]; lpm = F.log_softmax(lo, dim=-1)
                mm_llr = float(lpm[mid] - lpm[wid])
                ent = float(-(lpm.exp() * lpm).sum())
            cache[(seq, pos, wt_aa, mut_aa)] = [mm_llr / 5, wt_llr / 5, ent / 3, win_cons / 5]
            if (i + 1) % 100 == 0: pickle.dump(cache, open(MLM_CACHE, 'wb')); print(f'  {i+1}/{len(todo)}', flush=True)
        pickle.dump(cache, open(MLM_CACHE, 'wb')); del mlm; torch.cuda.empty_cache()
    out = {}
    for idx, r in df.iterrows():
        out[idx] = cache.get(keys.get(idx), [0., 0., 0., 0.])
    return out


def win_tokens(tok_arr, pos, k=WIN_K, dim=1280):
    p = min(max(0, pos), len(tok_arr) - 1); out = np.zeros((2 * k + 1, dim), np.float32)
    for j, t in enumerate(range(p - k, p + k + 1)):
        if 0 <= t < len(tok_arr): out[j] = tok_arr[t]
    return out


def build(df, cache, mlm, variant):
    feats = {}
    for idx, r in df.iterrows():
        wt_h, wt_l, wt_a = cache[r['wt_heavy']], cache[r['wt_light']], cache[r['wt_antigen']]
        mu_h, mu_l, mu_a = cache[r['mut_heavy']], cache[r['mut_light']], cache[r['mut_antigen']]
        ch, pos = rv.mut_chain_pos(r)
        wl, wh, wa = rv.pool_mean(wt_l), rv.pool_cdr(wt_h, r['wt_heavy']), rv.pool_mean(wt_a)
        ml, mh, ma = rv.pool_mean(mu_l), rv.pool_cdr(mu_h, r['mut_heavy']), rv.pool_mean(mu_a)
        if ch is not None:
            wt_seq = {'H': r['wt_heavy'], 'L': r['wt_light'], 'A': r['wt_antigen']}[ch]
            mut_seq = {'H': r['mut_heavy'], 'L': r['mut_light'], 'A': r['mut_antigen']}[ch]
            wt_aa = wt_seq[pos] if pos < len(wt_seq) else 'A'; mut_aa = mut_seq[pos] if pos < len(mut_seq) else 'A'
            is_ab = 1.0 if ch in ('H', 'L') else 0.0
            is_cdr = 1.0 if (ch == 'H' and pos in set(rv.cdr_indices(wt_seq))) else 0.0
            extra = mlm[idx] + phys(wt_aa, mut_aa) + [is_cdr, is_ab]
        else:
            extra = [0.0] * N_EXTRA
        row = [torch.tensor(np.asarray(x), dtype=torch.float32) for x in (wl, wh, wa, ml, mh, ma)]
        row.append(torch.tensor(extra, dtype=torch.float32))
        if variant == 'V7':
            tm = {'H': (wt_h, mu_h), 'L': (wt_l, mu_l), 'A': (wt_a, mu_a)}.get(ch, (wt_h, mu_h))
            p = pos if (ch is not None and pos < len(tm[0])) else 0
            row.append(torch.tensor(win_tokens(tm[0], p), dtype=torch.float32))
            row.append(torch.tensor(win_tokens(tm[1], p), dtype=torch.float32))
        feats[idx] = tuple(row)
    return feats


class Head6(nn.Module):
    def __init__(self, d=256, pretrained=True, freeze=False, neigh=False):
        super().__init__()
        self.core = rv.MutualTriStreamStrong(esm_dim=1280, projected_size=d, num_heads=8, dropout=0.1, n_layers=2)
        if pretrained and os.path.exists(PRETRAINED):
            self.core.load_state_dict(torch.load(PRETRAINED, map_location='cpu')['model_state_dict'], strict=False)
        if freeze:
            for p in self.core.parameters(): p.requires_grad = False
        self.neigh = neigh; ein = 6 * d + N_EXTRA
        if neigh:
            self.win_proj = nn.Linear(1280, d); self.attn = nn.MultiheadAttention(d, 4, batch_first=True); ein += d
        self.head = nn.Sequential(nn.Linear(ein, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def rep(self, l, h, a):
        o = self.core(l, h, a); return o['antibody_context'], o['antigen_context']

    def forward(self, b):
        wl, wh, wa, ml, mh, ma, extra = b[:7]
        ab_w, ag_w = self.rep(wl, wh, wa); ab_m, ag_m = self.rep(ml, mh, ma)
        parts = [ab_w, ag_w, ab_m, ag_m, ab_w - ab_m, ag_w - ag_m]
        if self.neigh:
            w = self.win_proj(b[8] - b[7]); a, _ = self.attn(w, w, w); parts.append(a.mean(1))
        parts.append(extra)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def stack(idxs, feats, ncols): return [torch.stack([feats[i][c] for i in idxs]).to(DEVICE) for c in range(ncols)]


def rank_loss(pred, y, margin=0.5):
    dp = pred.unsqueeze(1) - pred.unsqueeze(0); dy = y.unsqueeze(1) - y.unsqueeze(0)
    s = torch.sign(dy); m = (dy.abs() > 1e-6).float()
    return (F.relu(margin - s * dp) * m).sum() / (m.sum() + 1e-8)


def train_one(variant, idx_tr, feats, ddg, epochs, freeze, loss_kind, seed, ncols, init_state):
    torch.manual_seed(seed)
    m = Head6(pretrained=True, freeze=freeze, neigh=(variant == 'V7')).to(DEVICE)
    if init_state is not None: m.load_state_dict(init_state, strict=False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-5, weight_decay=1e-2)
    tr = list(idx_tr); np.random.seed(seed); np.random.shuffle(tr)
    nval = max(8, int(0.15 * len(tr))); va, trn = tr[:nval], tr[nval:]
    Xva = stack(va, feats, ncols); yva = torch.tensor(ddg[va], dtype=torch.float32, device=DEVICE)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); order = list(trn); np.random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]; X = stack(b, feats, ncols); y = torch.tensor(ddg[b], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(); pred = m(X)
            loss = rank_loss(pred, y) if loss_kind == 'rank' else F.mse_loss(pred, y)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = m(Xva); vl = (rank_loss(vp, yva) if loss_kind == 'rank' else F.mse_loss(vp, yva)).item()
        if vl < best - 1e-4: best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
    if state: m.load_state_dict(state)
    return m


def run_variant(variant, df, feats, ddg, args, init_state):
    ncols = 9 if variant == 'V7' else 7
    fold_r, P, T = [], [], []
    pred_records = []                                          # (complex, fold, true, pred) for ddg_eval.py
    for f in sorted(df['fold_id'].unique()):
        te = df[df.fold_id == f].index.tolist(); tr = df[df.fold_id != f].index.tolist()
        if getattr(args, 'train_frac', 1.0) < 1.0:            # few-shot: subsample training labels
            k = max(20, int(args.train_frac * len(tr)))
            tr = list(np.random.RandomState(123).permutation(tr)[:k])
        preds = []
        for s in range(args.nseeds):                          # ensemble
            m = train_one(variant, tr, feats, ddg, args.epochs, args.freeze, args.loss, s, ncols, init_state)
            m.eval()
            with torch.no_grad(): preds.append(m(stack(te, feats, ncols)).cpu().numpy())
        pred = np.mean(preds, axis=0); true = ddg[te]
        r = pearsonr(true, pred)[0] if len(true) > 2 else np.nan
        fold_r.append(r); P += list(pred); T += list(true)
        for cx, tv, pv in zip(df.loc[te, 'complex'].values, true, pred):
            pred_records.append(dict(complex=cx, fold=f, true=float(tv), pred=float(pv)))
        print(f'  [{variant} nseed={args.nseeds} loss={args.loss}] fold {f}: n={len(te)} r={r:.3f}', flush=True)
    if getattr(args, 'save_preds', None):
        pd.DataFrame(pred_records).to_csv(args.save_preds, index=False)
        print(f'  saved per-mutation predictions -> {args.save_preds}', flush=True)
    P, T = np.array(P), np.array(T)
    return dict(variant=variant, freeze=args.freeze, loss=args.loss, nseeds=args.nseeds,
                train_frac=getattr(args, 'train_frac', 1.0),
                pretrain=os.path.basename(args.pretrain_csv) if args.pretrain_csv else 'none',
                pooled_pearson=pearsonr(T, P)[0], meanfold_pearson=float(np.nanmean(fold_r)),
                meanfold_std=float(np.nanstd(fold_r)), spearman=spearmanr(T, P)[0],
                rmse=float(np.sqrt(np.mean((T - P) ** 2))))


def run_traintest(variant, df, feats, ddg, args, init_state):
    """Single train/val/test split (Graphinity's synthetic protocol) instead of 10-fold CV."""
    ncols = 9 if variant == 'V7' else 7
    tr = df[df['split'].isin(['train', 'val'])].index.tolist()
    te = df[df['split'] == 'test'].index.tolist()
    preds = []
    for s in range(args.nseeds):
        m = train_one(variant, tr, feats, ddg, args.epochs, args.freeze, args.loss, s, ncols, init_state)
        m.eval()
        with torch.no_grad(): preds.append(m(stack(te, feats, ncols)).cpu().numpy())
    pred = np.mean(preds, axis=0); true = ddg[te]
    return dict(variant=variant, loss=args.loss, nseeds=args.nseeds, n_train=len(tr), n_test=len(te),
                pearson=pearsonr(true, pred)[0], spearman=spearmanr(true, pred)[0],
                rmse=float(np.sqrt(np.mean((true - pred) ** 2))))


def pretrain_head(variant, csv, args):
    """Train a head on a broad ddG set (all rows, no folds); return its state_dict."""
    pre = pd.read_csv(csv).dropna(subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
    uniq = set()
    for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
        uniq |= set(pre[c].dropna().tolist())
    cache = rv.embed_tokens(list(uniq)); mlm = compute_mlm(pre)
    feats = build(pre, cache, mlm, variant); ddg = pre['ddg'].values.astype(np.float32)
    print(f'PRETRAIN {variant} on {len(pre)} rows from {os.path.basename(csv)} ...', flush=True)
    m = train_one(variant, list(pre.index), feats, ddg, args.pretrain_epochs, False, args.loss, 0,
                  9 if variant == 'V7' else 7, None)
    return {k: v.clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', nargs='+', default=['random', '90', '70'])
    ap.add_argument('--variants', nargs='+', default=['V7'])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--pretrain_epochs', type=int, default=15)
    ap.add_argument('--nseeds', type=int, default=1)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--loss', choices=['mse', 'rank'], default='mse')
    ap.add_argument('--pretrain_csv', default=None)
    ap.add_argument('--pairs_csv', default=None, help='self-contained pairs CSV with a train/val/test split col (e.g. synthetic FlexddG)')
    ap.add_argument('--train_frac', type=float, default=1.0, help='few-shot: fraction of training labels to use (data-efficiency curve)')
    ap.add_argument('--save_preds', default=None, help='write per-mutation predictions here (for ddg_eval.py paired bootstrap)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- synthetic / arbitrary train-test mode (same data + split as Graphinity) ----
    if a.pairs_csv:
        d = pd.read_csv(a.pairs_csv).dropna(subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
        uniq = set()
        for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
            uniq |= set(d[c].dropna().tolist())
        print(f'{len(d)} pairs | splits {d["split"].value_counts().to_dict()} | embedding {len(uniq)} seqs', flush=True)
        cache = rv.embed_tokens(list(uniq)); ddg = d['ddg'].values.astype(np.float32); mlm = compute_mlm(d)
        out_csv = os.path.join(OUT, 'summary_synthetic.csv'); res_rows = []
        for v in a.variants:
            feats = build(d, cache, mlm, v)
            res = run_traintest(v, d, feats, ddg, a, None); res['data'] = os.path.basename(a.pairs_csv)
            res_rows.append(res)
            pd.DataFrame([res]).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
            print(f"[{v} nseed={a.nseeds} loss={a.loss}] {os.path.basename(a.pairs_csv)}: "
                  f"test Pearson {res['pearson']:.3f} | rho {res['spearman']:.3f} | rmse {res['rmse']:.3f} "
                  f"(train {res['n_train']}, test {res['n_test']})", flush=True)
        print('\n(reference: Graphinity FlexddG synthetic ~0.6-0.7; FoldX ~0.9)')
        return

    seqs = pd.read_csv(SEQS).drop_duplicates('complex')
    uniq = set()
    for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
        uniq |= set(seqs[c].dropna().tolist())
    cache = rv.embed_tokens(list(uniq))

    init_states = {}
    if a.pretrain_csv:
        for v in a.variants:
            init_states[v] = pretrain_head(v, a.pretrain_csv, a)

    csv_path = os.path.join(OUT, 'summary_seq_improved.csv')
    rows = []; _sp_base = a.save_preds
    for cut in a.cutoffs:
        sp = pd.read_csv(os.path.join(SPLITDIR, f'Experimental_ddG_645_-Reverse_Mutations_+Non_Binders-cutoff_{cut}-10foldcv.csv'))
        d = sp.merge(seqs, on='complex', how='inner').dropna(subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
        ddg = d['ddg'].values.astype(np.float32); mlm = compute_mlm(d)
        print(f'\n==== cutoff {cut}: {len(d)} mutations ====', flush=True)
        a.save_preds = _sp_base.replace('.csv', f'_{cut}.csv') if _sp_base else None   # per-cutoff preds
        for v in a.variants:
            feats = build(d, cache, mlm, v)
            res = run_variant(v, d, feats, ddg, a, init_states.get(v)); res['cutoff'] = cut
            rows.append(res)
            # APPEND mode: accumulate across all runs
            hdr = not os.path.exists(csv_path)
            pd.DataFrame([res]).to_csv(csv_path, mode='a', header=hdr, index=False)
            print(f"[{v} nseed={a.nseeds} loss={a.loss} freeze={a.freeze} pre={res['pretrain']}] {cut}: "
                  f"mean-fold {res['meanfold_pearson']:.3f}+/-{res['meanfold_std']:.3f} | rho {res['spearman']:.3f}", flush=True)
    print('\n================ RUN SUMMARY ================')
    print(pd.DataFrame(rows)[['cutoff', 'variant', 'loss', 'nseeds', 'pretrain', 'meanfold_pearson', 'meanfold_std', 'spearman']].to_string(index=False))
    print('reference: V5 0.43/-0.08/0.01 ; V7-mse 0.46/0.09/0.17 ; Graphinity 0.44/0.385/0.354')


if __name__ == '__main__':
    main()
