# AbAffinity-ΔΔG — reproducibility package

Code and results for *"Lightweight Chain-Aware Modeling for Antibody Affinity and
Mutational Effects"* (AbAffinity absolute-affinity model + affinity-anchored
ΔΔG mutation transfer). This package contains only what is needed to
**verify the exact numbers reported in the paper's tables and figures** — it
is a cleaned subset of a much larger research repository, not the full
experimental history.

## What's included

```
models/         mutual_strong.py — the frozen-ESM-2 chain-aware tri-stream
                architecture (heavy-CDR pooling, heavy-light self-attention +
                fusion gate, gated cross-attention, cosine→pKd calibration).
                Reference copy; also duplicated into ddg/ so the ddG scripts
                import it without extra path setup.

ddg/            Affinity-anchored mutation-transfer pipeline:
                  run_ddg_variants.py       AbAffinity core reused for ddG (ESM-2
                                             caching, CDR pooling, WT/mutant scoring)
                  run_ddg_seq_improved.py   mutation descriptors, feature stacking,
                                             ranking loss
                  moe_ddg.py                main training script: affinity anchor
                                             + gated correction head (Eq. 4).
                                             --anchor_pkd / --anchor_dg / --no_moe
                                             select the three Table S7 anchor
                                             variants and the anchor-only ablation.
                  baselines_ddg.py          Ridge / RandomForest / XGBoost / MLP
                                             frozen-ESM-2 regressors (Table S5)
                  neural_baselines_ddg.py   parameter-matched CNN / LSTM / Transformer
                                             (Table S5, S1131 comparison)
                  compute_metrics_ci.py     Pearson/Spearman/RMSE/direction-accuracy/
                                             AUROC with bootstrap CIs

figures/        make_affinity_master_figure.py  -> figure_affinity_results.{png,pdf}
                                                    (Figure 2, Tables S1-S4)
                make_ddg_master_figure.py       -> figure_ddg_results.{png,pdf}
                                                    (Figure 3, Tables S5-S8)
                ig_1vfb_composite.png            Integrated-Gradients + structural
                                                  mapping panel used in Figure 2e
                Pre-rendered PDF/PNG outputs are included so the figures can be
                inspected without rerunning anything.

results/        Saved per-fold predictions (true/pred/fold) for the S1131
                mutation benchmark, one subfolder per anchor/ablation variant
                used by make_ddg_master_figure.py:
                  anchorpkd_final/   main model, pKd-difference anchor (Table S5,
                                     S7 "pKd difference (main)", Table S8)
                  anchoronly_final/  anchor-only ablation, no correction head
                                     (Table S5 "Affinity anchor only", Fig. 3e)
                  anchordg_final/    ΔG-difference anchor (Table S7 "ΔG difference")
                  cosine_final/,
                  wmt_preds/         raw-cosine-difference anchor (Table S7
                                     "Cosine difference")
                Each split has a pooled CSV (s1131_fold_<split>.csv) and three
                per-seed CSVs (_s0/_s1/_s2) — this is exactly what the figure
                script reads and aggregates (3-seed mean ± s.d.).

paper/          LaTeX sources for the absolute-affinity and ddG results
                sections and supplementary tables, for direct cross-reference
                against the numbers above.
```

## What's *not* included, and why

Tables S1–S4 and Figure 2's architecture/backbone/external-benchmark numbers
(0.842 random-split Pearson, ESM-2 vs. ProtBERT/AntiBERTy/ProGen2, vs. MVSF-AB
on SAbDab/AB-Bind/SKEMPI/held-out) are **reused from the companion AbAffinity
paper's codebase** (chain-aware absolute-affinity model, trained and
evaluated there under 10-fold CV on SAAINT-DB). `make_affinity_master_figure.py`
embeds those exact verified numbers directly (see the script's `AB`, `BP`,
`PCC_*`, `RMSE_*` arrays) rather than recomputing them, so Figure 2 reproduces
immediately with no external dependency.

Full **end-to-end retraining** of the ddG pipeline (rather than regenerating
figures/tables from the saved `results/` CSVs) additionally requires:
- The pretrained AbAffinity checkpoint and ESM-2 embedding/token caches from
  the companion absolute-affinity repository (`run_ddg_variants.py` loads a
  specific fold checkpoint and cached sequence embeddings by path).
- Setting `ABAFF_PIPE` (used by `moe_ddg.py`) to that companion repository's
  `graphinity_comparison/` directory, e.g.:
  ```bash
  export ABAFF_PIPE=/path/to/AbAffinity-main/graphinity_comparison
  ```
- The S1131 / AB645 / SKEMPI mutation input CSVs and structures, which are
  not redistributed here (see the original SKEMPI 2.0 / AB-Bind licenses).

This is a deliberate scope decision: shipping the multi-GB checkpoint and
embedding caches was not practical for this package, and it isn't needed to
verify the paper's reported numbers — regenerating the figures from
`results/` (below) reproduces every number in Tables S5–S8 exactly.

## Reproducing the figures (no external data needed)

```bash
pip install numpy pandas matplotlib scipy scikit-learn
cd figures/
python make_ddg_master_figure.py       # -> figure_ddg_results.png / .pdf
python make_affinity_master_figure.py  # -> figure_affinity_results.png / .pdf
```

`make_ddg_master_figure.py` computes panels a/b/d/e/f live from the CSVs in
`../results/`, and panel c (external published-method comparison, Table S6)
from literal values taken directly from Table S6 / the cited papers. Expected
output (mean ± s.d. over 3 seeds), matching Tables S5 and S7:

| | Random | Complex-disjoint | Antigen-disjoint |
|---|---|---|---|
| AbAffinity-ΔΔG (pKd anchor, main) | 0.801 ± 0.004 | 0.713 ± 0.017 | 0.675 ± 0.019 |
| Affinity anchor only | 0.722 | 0.687 | 0.650 |
| Cosine-difference anchor | 0.803 ± 0.008 | 0.700 ± 0.010 | 0.670 ± 0.039 |
| ΔG-difference anchor | 0.792 ± 0.006 | 0.726 ± 0.008 | 0.682 ± 0.018 |

## Citation

If you use this code, please cite the paper (see `paper/` for the full text)
and the companion absolute-affinity model:

> Singh, H., Malhotra, A., Srivastava, S.P., Singh, R.K., Gorantla, R.
> Antibody–antigen affinity prediction with chain-aware protein language
> modeling. *bioRxiv* (2026). doi:10.64898/2026.06.19.733375
