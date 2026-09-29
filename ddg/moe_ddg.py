"""
moe_ddg.py — AbAffinity-MoE-ddG: a pure-sequence Mixture-of-Experts for antibody-antigen ddG.
==============================================================================================
Novel head on top of the frozen AbAffinity backbone (MutualTriStreamStrong over ESM-2 650M),
with a LEARNED GATING network that routes each mutation to specialised expert MLPs. Two modes,
both trained end-to-end on measured ddG with a load-balancing auxiliary loss:

  --mode direct     : experts consume the Siamese difference features
                      [ab_wt, ag_wt, ab_mut, ag_mut, ab_wt-ab_mut, ag_wt-ag_mut, (window), descr]
                      ddG_hat = sum_k gate_k(descr) * expert_k(features)

  --mode subtract   : (the "one expert bank for WT, one for MUT, subtract" idea)
                      a SHARED expert bank maps a single complex rep [ab, ag] -> scalar affinity;
                      s_wt  = sum_k gate_k(descr) * expert_k([ab_wt , ag_wt ])
                      s_mut = sum_k gate_k(descr) * expert_k([ab_mut, ag_mut])
                      ddG_hat = s_wt - s_mut     (leverages AbAffinity's strong absolute-pKd signal)

Gating input = the 10 antibody-agnostic mutation descriptors already computed by the pipeline
(4 ESM-2 MLM scores + Δcharge/Δhydropathy/Δvolume/BLOSUM62 + in-CDR + is-antibody). This lets
experts specialise by mutation regime (conservative/radical, CDR/framework, small/large effect),
directly targeting the |ddG|>2 large-effect failure of the single-head model.

Reuses the existing pipeline for ESM-2 token caching, CV splits, CDR pooling, MLM + descriptor
features (imported from run_ddg_seq_improved / run_ddg_variants). Writes per-mutation predictions
compatible with ddg_eval.py, and runs 3 SEEDS per fold by default.

Usage (run where the ESM-2 caches live is handled automatically via PIPE):
  python moe_ddg.py --mode direct   --cutoffs random 90 70 antigencold --nexperts 4 --seeds 0 1 2 \
                    --save_preds .../moe_direct.csv
  python moe_ddg.py --mode subtract --cutoffs random --nexperts 6 --seeds 0 1 2 \
                    --save_preds .../moe_subtract.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

# --- import the existing AbAffinity ddG pipeline (caches, splits, features live next to it) ---
PIPE = os.environ.get('ABAFF_PIPE',
    r"C:\Users\hs494\OneDrive - Shiv Nadar Institution of Eminence\Desktop\BALM_Ag_Ab\3_stream\graphinity_comparison")
sys.path.insert(0, PIPE)
import run_ddg_variants as rv          # noqa: E402  (AbAffinity core, ESM-2 caching, pooling)
import run_ddg_seq_improved as si      # noqa: E402  (build features, compute_mlm, stack, rank_loss)

N_EXTRA = si.N_EXTRA                    # 10 gating descriptors
PRETRAINED, DEVICE = rv.PRETRAINED, rv.DEVICE


# =====================================================================================
# Model
# =====================================================================================
class Expert(nn.Module):
    def __init__(self, d_in, d_hidden, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeadMoE(nn.Module):
    """AbAffinity backbone + gated Mixture-of-Experts ddG head (direct | subtract | dgsub)."""
    def __init__(self, d=256, K=4, topk=None, mode='direct', neigh=True,
                 pretrained=True, freeze=False, gate_hidden=64, expert_dim=None,
                 gate_noise=0.0, expert_drop=0.0, no_moe=False, anchor_pkd=False, anchor_dg=False):
        super().__init__()
        self.core = rv.MutualTriStreamStrong(esm_dim=1280, projected_size=d,
                                             num_heads=8, dropout=0.1, n_layers=2)
        self.anchor_pkd = anchor_pkd; self.anchor_dg = anchor_dg
        self.pkd_lo, self.pkd_hi = 4.33, 13.47               # AbAffinity checkpoint pkd_bounds
        if pretrained and os.path.exists(PRETRAINED):
            ckpt = torch.load(PRETRAINED, map_location='cpu')
            self.core.load_state_dict(ckpt['model_state_dict'], strict=False)
            b = ckpt.get('pkd_bounds') or (ckpt.get('config', {}) or {}).get('pkd_bounds')
            if b is not None:
                self.pkd_lo, self.pkd_hi = float(b[0]), float(b[1])
        if freeze:
            for p in self.core.parameters():
                p.requires_grad = False
        self.mode, self.K = mode, K
        self.topk = K if topk is None else min(topk, K)
        self.gate_noise, self.expert_drop, self.no_moe = gate_noise, expert_drop, no_moe
        self.last_imp = self.last_ent = self.last_pkd_wt = None
        self.neigh = neigh and (mode in ('direct', 'dgsub', 'local'))
        if self.neigh:
            self.win_proj = nn.Linear(1280, d)
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)

        # gate: mutation descriptors -> K logits
        self.gate = nn.Sequential(nn.Linear(N_EXTRA, gate_hidden), nn.GELU(),
                                  nn.Linear(gate_hidden, K))
        # experts
        if mode in ('direct', 'dgsub'):
            ein = 6 * d + N_EXTRA + (d if self.neigh else 0)
        elif mode == 'local':                            # mutation-local ONLY: ±7 window + descriptors
            ein = d + N_EXTRA
        else:                                            # subtract: single-complex rep [ab, ag]
            ein = 2 * d
        eh = expert_dim or d                             # expert hidden width (capacity knob)
        self.experts = nn.ModuleList([Expert(ein, eh) for _ in range(K)])
        if mode == 'dgsub':                              # physics anchor: ΔΔG ∝ pKd_wt - pKd_mut
            self.dg_scale = nn.Parameter(torch.tensor(1.36))
            self.pkd_w = nn.Parameter(torch.tensor(4.0))  # calibrate cos_wt -> absolute WT pKd
            self.pkd_b = nn.Parameter(torch.tensor(8.0))

    def rep(self, l, h, a):
        o = self.core(l, h, a)
        return o['antibody_context'], o['antigen_context']

    def _gate_weights(self, extra):
        logits = self.gate(extra)                        # (B, K)
        if self.training and self.gate_noise > 0:        # noisy gating for exploration
            logits = logits + torch.randn_like(logits) * self.gate_noise
        if self.topk < self.K:                           # sparse top-k routing
            thr = logits.topk(self.topk, dim=-1).values[:, -1:].detach()
            logits = logits.masked_fill(logits < thr, float('-inf'))
        return F.softmax(logits, dim=-1)                 # (B, K)

    def _combine(self, feats, g):
        outs = torch.stack([e(feats) for e in self.experts], dim=-1)   # (B, K)
        if self.training and self.expert_drop > 0:       # drop-expert regularisation
            mask = (torch.rand(self.K, device=g.device) > self.expert_drop).float()
            if mask.sum() == 0:
                mask[torch.randint(self.K, (1,))] = 1.0
            g = g * mask; g = g / (g.sum(-1, keepdim=True) + 1e-8)
        return (g * outs).sum(-1)                         # (B,)

    def forward(self, b):
        wl, wh, wa, ml, mh, ma, extra = b[:7]
        g = self._gate_weights(extra)                    # (B, K)
        if self.mode == 'local':                         # mutation-local only (no whole-complex, no core)
            w = self.win_proj(b[8] - b[7]); att, _ = self.attn(w, w, w)
            pred = self._combine(torch.cat([att.mean(1), extra], dim=-1), g)
            imp = g.mean(0); self.last_imp = imp.detach()
            self.last_ent = -(g * (g + 1e-9).log()).sum(-1).mean()
            return pred, (imp * imp).sum() * self.K
        ab_w, ag_w = self.rep(wl, wh, wa)
        ab_m, ag_m = self.rep(ml, mh, ma)
        if self.mode in ('direct', 'dgsub'):
            parts = [ab_w, ag_w, ab_m, ag_m, ab_w - ab_m, ag_w - ag_m]
            if self.neigh:
                w = self.win_proj(b[8] - b[7]); att, _ = self.attn(w, w, w)
                parts.append(att.mean(1))
            parts.append(extra)
            moe = self._combine(torch.cat(parts, dim=-1), g)
            if self.mode == 'dgsub':                      # AbAffinity cosine = absolute-affinity anchor
                cos_wt = F.cosine_similarity(ab_w, ag_w, dim=-1)
                cos_mut = F.cosine_similarity(ab_m, ag_m, dim=-1)
                span = self.pkd_hi - self.pkd_lo
                pkd_wt = self.pkd_lo + span * (cos_wt + 1.0) / 2.0   # AbAffinity native pKd conversion
                pkd_mut = self.pkd_lo + span * (cos_mut + 1.0) / 2.0
                if self.anchor_dg:                        # pKd->dG (kcal/mol) per state, then dG_mut - dG_wt
                    dg_wt = -1.364 * pkd_wt               # dG = -RT ln10 * pKd
                    dg_mut = -1.364 * pkd_mut
                    anchor = self.dg_scale * (dg_mut - dg_wt)
                elif self.anchor_pkd:                     # pKd difference (main model)
                    anchor = self.dg_scale * (pkd_wt - pkd_mut)
                else:                                     # scaled cosine difference (equivalent up to scale)
                    anchor = self.dg_scale * (cos_wt - cos_mut)
                pred = anchor + (0.0 if self.no_moe else moe)  # anchor-only ablation via --no_moe
                self.last_pkd_wt = self.pkd_w * cos_wt + self.pkd_b   # for WT-affinity supervision
            else:
                pred = moe
        else:                                            # subtract
            s_wt = self._combine(torch.cat([ab_w, ag_w], dim=-1), g)
            s_mut = self._combine(torch.cat([ab_m, ag_m], dim=-1), g)
            pred = s_wt - s_mut
        imp = g.mean(0); self.last_imp = imp.detach()
        aux = (imp * imp).sum() * self.K                 # load balance (min at uniform)
        self.last_ent = -(g * (g + 1e-9).log()).sum(-1).mean()   # gate entropy (exploration bonus)
        return pred, aux


# =====================================================================================
# Training / evaluation (3 seeds, CV folds), reusing pipeline feature builders
# =====================================================================================
def _weight(y, args):
    """Per-sample weighting to emphasise large-effect mutations (|ddG| large)."""
    a = getattr(args, 'walpha', 0.5); tau = getattr(args, 'wtau', 2.0)
    ay = y.abs(); sch = getattr(args, 'wscheme', 'linear')
    if sch == 'linear':                                   # w = 1 + a|ddG|
        w = 1.0 + a * ay
    elif sch == 'quad':                                   # w = 1 + a|ddG|^2
        w = 1.0 + a * ay ** 2
    elif sch == 'hinge':                                  # only penalise above threshold tau
        w = 1.0 + a * torch.clamp(ay - tau, min=0.0)
    elif sch == 'bin':                                    # regime weights: <1 / 1-2 / >=2
        w = torch.ones_like(ay)
        w = torch.where(ay >= 1.0, torch.full_like(ay, 1.5), w)
        w = torch.where(ay >= 2.0, torch.full_like(ay, 3.0), w)
    else:
        w = torch.ones_like(ay)
    return w


def _base_loss(pred, y, args):
    if args.loss == 'rank':
        return si.rank_loss(pred, y)
    if args.loss in ('wmse', 'whuber'):                   # weighted regression on large effects
        w = _weight(y, args)
        if args.loss == 'whuber':                         # Huber = robust to large-effect outliers
            per = F.huber_loss(pred, y, reduction='none', delta=getattr(args, 'huber_delta', 1.0))
        else:
            per = (pred - y) ** 2
        return (w * per).mean() / w.mean()
    return F.mse_loss(pred, y)


def _lossf(pred, y, args):
    """Full weighted-multitask base+rank (WT/large-effect terms are added in train_one)."""
    loss = _base_loss(pred, y, args)
    if getattr(args, 'rank_coef', 0.0) > 0 and args.loss != 'rank':
        loss = loss + args.rank_coef * si.rank_loss(pred, y)
    return loss


def train_one(idx_tr, feats, ddg, ncols, args, seed, groups=None, init_state=None, wtpkd=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = HeadMoE(d=getattr(args, 'backbone_dim', 256), K=args.nexperts, topk=args.topk, mode=args.mode,
                neigh=(args.mode in ('direct', 'dgsub', 'local')), freeze=args.freeze,
                pretrained=not getattr(args, 'scratch', False),
                expert_dim=getattr(args, 'expert_dim', None),
                gate_noise=getattr(args, 'gate_noise', 0.0),
                expert_drop=getattr(args, 'expert_drop', 0.0),
                no_moe=getattr(args, 'no_moe', False),
                anchor_pkd=getattr(args, 'anchor_pkd', False),
                anchor_dg=getattr(args, 'anchor_dg', False)).to(DEVICE)
    if init_state is not None:                            # warm-start from synthetic pretraining
        m.load_state_dict(init_state, strict=False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-2)
    tr = list(idx_tr); np.random.shuffle(tr)
    if getattr(args, 'grouped_val', False) and groups is not None:   # hold out whole complexes
        gs = list(dict.fromkeys(groups[i] for i in tr)); np.random.RandomState(seed).shuffle(gs)
        vg = set(gs[:max(1, int(0.15 * len(gs)))])
        va = [i for i in tr if groups[i] in vg]; trn = [i for i in tr if groups[i] not in vg]
        if len(va) < 4 or len(trn) < 4:                              # fallback if grouping too coarse
            nval = max(8, int(0.15 * len(tr))); va, trn = tr[:nval], tr[nval:]
    else:
        nval = max(8, int(0.15 * len(tr))); va, trn = tr[:nval], tr[nval:]
    Xva = si.stack(va, feats, ncols)
    yva = torch.tensor(ddg[va], dtype=torch.float32, device=DEVICE)
    best, state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        m.train(); order = list(trn); np.random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            X = si.stack(b, feats, ncols)
            y = torch.tensor(ddg[b], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(); pred, aux = m(X)
            base = _lossf(pred, y, args)
            if getattr(args, 'bigrank_coef', 0) > 0:          # aux ranking on large-effect subset
                big = y.abs() > 2.0
                if int(big.sum()) > 2:
                    base = base + args.bigrank_coef * si.rank_loss(pred[big], y[big])
            if getattr(args, 'wtaff_coef', 0) > 0 and wtpkd is not None and m.last_pkd_wt is not None:
                wp = wtpkd[b]; msk = ~np.isnan(wp)        # supervise ΔG_wt against known WT pKd
                if msk.sum() > 1:
                    mt = torch.tensor(msk, device=DEVICE)
                    tgt = torch.tensor(wp[msk], dtype=torch.float32, device=DEVICE)
                    base = base + args.wtaff_coef * F.mse_loss(m.last_pkd_wt[mt], tgt)
            ecoef = getattr(args, 'gate_entropy', 0.0) * max(0.0, 1.0 - ep / max(1, int(0.5 * args.epochs)))
            (base + args.lb_coef * aux - ecoef * m.last_ent).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp, _ = m(Xva)
            vl = _lossf(vp, yva, args).item()
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12:
                break
    if state:
        m.load_state_dict(state)
    return m


def cv_eval(d, cache, args, tag, init_state=None):
    """Run seeded CV over d[fold_col] and return (result dict, ensemble preds df, per-seed dfs)."""
    fc = getattr(args, 'fold_col', 'fold_id')
    if fc != 'fold_id':
        d = d.copy(); d['fold_id'] = d[fc]               # route the chosen split into fold_id
    ncols = 9                                            # build with V7 features (incl. window)
    ddg = d['ddg'].values.astype(np.float32)
    wtpkd = d['wt_pkd'].values.astype(np.float32) if 'wt_pkd' in d.columns else None   # WT-affinity anchor
    groups = (d['pdb'] if 'pdb' in d.columns else d['complex']).values   # for grouped inner-val
    mlm = si.compute_mlm(d)
    feats = si.build(d, cache, mlm, 'V7')
    print(f'\n==== {tag}: {len(d)} mutations | mode={args.mode} K={args.nexperts} '
          f'topk={args.topk} seeds={args.seeds} ====', flush=True)

    pred_records, fold_r, P, T = [], [], [], []
    seed_records = {s: [] for s in args.seeds}           # per-seed preds (for 3-seed mean±std)
    folds = sorted(d['fold_id'].unique())
    if getattr(args, 'only_fold', None) is not None:     # fixed train/test split (e.g. MMSeqs2 identity)
        folds = [args.only_fold]
    for f in folds:
        te = d[d.fold_id == f].index.tolist()
        tr = d[d.fold_id != f].index.tolist()
        seed_preds = []
        for s in args.seeds:                             # 3-seed ensemble per fold
            m = train_one(tr, feats, ddg, ncols, args, s, groups, init_state, wtpkd)
            if getattr(args, 'monitor', False) and s == args.seeds[0] and m.last_imp is not None:
                print(f'    expert util (fold {f}): {np.round(m.last_imp.cpu().numpy(), 3)}', flush=True)
            m.eval()
            with torch.no_grad():
                p, _ = m(si.stack(te, feats, ncols))
            sp = p.cpu().numpy(); seed_preds.append(sp)
            for cx, tv, pv in zip(d.loc[te, 'complex'].values, ddg[te], sp):
                seed_records[s].append(dict(complex=cx, fold=int(f), true=float(tv), pred=float(pv)))
        pred = np.mean(seed_preds, axis=0); true = ddg[te]
        r = pearsonr(true, pred)[0] if len(true) > 2 else np.nan
        fold_r.append(r); P += list(pred); T += list(true)
        for cx, tv, pv in zip(d.loc[te, 'complex'].values, true, pred):
            pred_records.append(dict(complex=cx, fold=int(f), true=float(tv), pred=float(pv)))
        print(f'  fold {f}: n={len(te)} r={r:.3f}', flush=True)

    P, T = np.array(P), np.array(T)
    res = dict(method=f'MoE-{args.mode}', cutoff=tag, K=args.nexperts, topk=args.topk,
               nseeds=len(args.seeds), loss=args.loss,
               pooled_pearson=float(pearsonr(T, P)[0]),
               meanfold_pearson=float(np.nanmean(fold_r)),
               meanfold_std=float(np.nanstd(fold_r)),
               spearman=float(spearmanr(T, P)[0]),
               rmse=float(np.sqrt(np.mean((T - P) ** 2))))
    seed_dfs = {s: pd.DataFrame(recs) for s, recs in seed_records.items()}
    return res, pd.DataFrame(pred_records), seed_dfs


def run_cutoff(cut, seqs, cache, args, init_state=None):
    """645-benchmark path: load a CDR-seqid split from SPLITDIR, merge sequences, run CV."""
    sp = pd.read_csv(os.path.join(
        si.SPLITDIR,
        f'Experimental_ddG_645_-Reverse_Mutations_+Non_Binders-cutoff_{cut}-10foldcv.csv'))
    d = sp.merge(seqs, on='complex', how='inner').dropna(
        subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
    return cv_eval(d, cache, args, cut, init_state)


def pretrain_state(csv, cache, args):
    """Pretrain the MoE on a large (synthetic) ddG pairs CSV; return its state_dict for warm-start."""
    import copy
    d = pd.read_csv(csv).dropna(subset=['wt_heavy', 'wt_antigen']).reset_index(drop=True)
    uniq = set()
    for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
        uniq |= set(d[c].dropna().astype(str).tolist())
    rv.embed_tokens(list(uniq))                          # add synthetic seqs to the ESM-2 token cache
    ddg = d['ddg'].values.astype(np.float32)
    mlm = si.compute_mlm(d); feats = si.build(d, cache, mlm, 'V7')
    groups = (d['pdb'] if 'pdb' in d.columns else d['complex']).values
    pa = copy.copy(args); pa.epochs = args.pretrain_epochs
    print(f'PRETRAIN {args.mode} on {len(d)} synthetic mutations ({args.pretrain_epochs} ep) ...', flush=True)
    m = train_one(list(d.index), feats, ddg, 9, pa, seed=0, groups=groups)
    return {k: v.clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['direct', 'subtract', 'dgsub', 'local'], default='direct')
    ap.add_argument('--pairs_csv', default=None,
                    help='self-contained benchmark CSV (wt_/mut_ seqs + ddg + fold_id), e.g. SKEMPI AB/AG')
    ap.add_argument('--fold_col', default='fold_id',
                    help='which CV column to use (fold_record/fold_pdb/fold_protein/fold_cluster from build_rigorous_splits.py)')
    ap.add_argument('--cutoffs', nargs='+', default=['random', '90', '70', 'antigencold'])
    ap.add_argument('--nexperts', type=int, default=4)
    ap.add_argument('--topk', type=int, default=None, help='sparse routing; default dense (=K)')
    ap.add_argument('--no_moe', action='store_true', help='ablation: anchor-only dgsub (drop the MoE correction)')
    ap.add_argument('--anchor_dg', action='store_true',
                    help='dgsub ablation: pKd->dG per state (dG=-RTln10*pKd), then dG_mut - dG_wt')
    ap.add_argument('--anchor_pkd', action='store_true',
                    help='dgsub ablation: convert cos->pKd (pKd_lo+(pKd_hi-pKd_lo)(cos+1)/2) per side then subtract, '
                         'instead of scaling the raw cosine difference')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--loss', choices=['mse', 'rank', 'wmse', 'whuber'], default='rank')
    ap.add_argument('--wpow', type=float, default=1.0, help='(legacy wmse) weight = 1 + wpow*|ddG|')
    ap.add_argument('--wscheme', choices=['linear', 'quad', 'hinge', 'bin'], default='linear', help='large-effect weighting scheme for wmse/whuber')
    ap.add_argument('--walpha', type=float, default=0.5, help='weighting strength alpha')
    ap.add_argument('--wtau', type=float, default=2.0, help='hinge threshold tau (|ddG| above which to penalise)')
    ap.add_argument('--huber_delta', type=float, default=1.0, help='Huber delta')
    ap.add_argument('--rank_coef', type=float, default=0.0, help='add lambda_rank * ranking loss on top of regression (multitask)')
    ap.add_argument('--expert_dim', type=int, default=None, help='expert hidden width (capacity knob, e.g. 512); backbone stays 256')
    ap.add_argument('--backbone_dim', type=int, default=256, help='AbAffinity projected_size; use 512 only with --scratch (256 checkpoint cannot load into 512)')
    ap.add_argument('--scratch', action='store_true', help='train the tri-stream backbone from scratch on ddG (no AbAffinity pretrain); enables --backbone_dim 512')
    ap.add_argument('--gate_noise', type=float, default=0.0, help='std of Gaussian noise added to gate logits during training (exploration)')
    ap.add_argument('--gate_entropy', type=float, default=0.0, help='early gate-entropy bonus (decays over first half of training)')
    ap.add_argument('--expert_drop', type=float, default=0.0, help='drop-expert regularisation prob during training')
    ap.add_argument('--bigrank_coef', type=float, default=0.0, help='aux ranking loss on |ddG|>2 subset (large-effect fix)')
    ap.add_argument('--wtaff_coef', type=float, default=0.0, help='dgsub: aux loss anchoring cos_wt to known WT pKd (needs wt_pkd column)')
    ap.add_argument('--monitor', action='store_true', help='print expert-utilization vector per fold')
    ap.add_argument('--only_fold', type=int, default=None, help='evaluate only this fold (fixed train/test split, e.g. MMSeqs2 identity: test=fold 0)')
    ap.add_argument('--pretrain_csv', default=None, help='pretrain on a large synthetic ddG pairs CSV (e.g. flexddg_pairs_none.csv), then fine-tune on the folds')
    ap.add_argument('--pretrain_epochs', type=int, default=15)
    ap.add_argument('--lb_coef', type=float, default=0.01, help='load-balancing aux-loss weight')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--grouped_val', action='store_true', help='hold out whole complexes for inner-val early stopping (better for OOD splits)')
    ap.add_argument('--save_preds', default=None,
                    help='base path; per-cutoff files get _<cutoff>.csv appended')
    ap.add_argument('--summary_csv', default=os.path.join(si.OUT, 'summary_moe.csv'))
    a = ap.parse_args()

    # ---- self-contained benchmark (e.g. SKEMPI AB/AG): CV over the CSV's own fold_id ----
    if a.pairs_csv:
        d = pd.read_csv(a.pairs_csv).dropna(
            subset=['wt_heavy', 'wt_antigen', a.fold_col]).reset_index(drop=True)
        uniq = set()
        for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
            uniq |= set(d[c].dropna().tolist())
        cache = rv.embed_tokens(list(uniq))
        init_state = pretrain_state(a.pretrain_csv, cache, a) if a.pretrain_csv else None
        tag = os.path.basename(a.pairs_csv).replace('.csv', '')
        res, preds, seed_dfs = cv_eval(d, cache, a, tag, init_state)
        if a.save_preds:
            os.makedirs(os.path.dirname(a.save_preds) or '.', exist_ok=True)
            preds.to_csv(a.save_preds, index=False)
            for s, sdf in seed_dfs.items():
                sdf.to_csv(a.save_preds.replace('.csv', f'_s{s}.csv'), index=False)
            print(f'  saved preds -> {a.save_preds}  (+ per-seed _s*.csv)', flush=True)
        hdr = not os.path.exists(a.summary_csv)
        pd.DataFrame([res]).to_csv(a.summary_csv, mode='a', header=hdr, index=False)
        print(f"[MoE-{a.mode} K={a.nexperts}] {tag}: pooled {res['pooled_pearson']:.3f} | "
              f"mean-fold {res['meanfold_pearson']:.3f}+/-{res['meanfold_std']:.3f} | "
              f"rho {res['spearman']:.3f}", flush=True)
        return

    seqs = pd.read_csv(si.SEQS).drop_duplicates('complex')
    uniq = set()
    for c in ('wt_heavy', 'wt_light', 'wt_antigen', 'mut_heavy', 'mut_light', 'mut_antigen'):
        uniq |= set(seqs[c].dropna().tolist())
    cache = rv.embed_tokens(list(uniq))
    init_state = pretrain_state(a.pretrain_csv, cache, a) if a.pretrain_csv else None

    rows = []
    for cut in a.cutoffs:
        res, preds, seed_dfs = run_cutoff(cut, seqs, cache, a, init_state)
        rows.append(res)
        if a.save_preds:
            path = a.save_preds.replace('.csv', f'_{cut}.csv')
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            preds.to_csv(path, index=False)                       # seed-ensemble preds
            for s, sdf in seed_dfs.items():                       # per-seed preds (3-seed spread)
                sdf.to_csv(a.save_preds.replace('.csv', f'_s{s}_{cut}.csv'), index=False)
            print(f'  saved preds -> {path}  (+ per-seed _s*_{cut}.csv)', flush=True)
        hdr = not os.path.exists(a.summary_csv)
        pd.DataFrame([res]).to_csv(a.summary_csv, mode='a', header=hdr, index=False)
        print(f"[MoE-{a.mode} K={a.nexperts}] {cut}: pooled {res['pooled_pearson']:.3f} | "
              f"mean-fold {res['meanfold_pearson']:.3f}+/-{res['meanfold_std']:.3f} | "
              f"rho {res['spearman']:.3f}", flush=True)

    print('\n================ MoE RUN SUMMARY ================')
    print(pd.DataFrame(rows)[['method', 'cutoff', 'K', 'nseeds', 'loss',
                              'pooled_pearson', 'meanfold_pearson', 'spearman']].to_string(index=False))
    print('reference (best single-head V7+rank+ens3): random 0.469 pooled ; Graphinity 0.459/0.301/0.157')


if __name__ == '__main__':
    main()
