# AbAffinity-ΔΔG — code

Code for *"Lightweight Chain-Aware Modeling for Antibody Affinity and
Mutational Effects"* (AbAffinity absolute-affinity model + affinity-anchored
ΔΔG mutation transfer). This is a cleaned subset of a much larger research
repository containing only the code that produced the numbers and figures
reported in the paper — not the full experimental history, and not the
saved result files themselves.

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
                                             select the three anchor-formulation
                                             variants (Table S7) and the
                                             anchor-only ablation (Table S5).
                  baselines_ddg.py          Ridge / RandomForest / XGBoost / MLP
                                             frozen-ESM-2 regressors (Table S5)
                  neural_baselines_ddg.py   parameter-matched CNN / LSTM / Transformer
                                             (Table S5, S1131 comparison)
                  compute_metrics_ci.py     Pearson/Spearman/RMSE/direction-accuracy/
                                             AUROC with bootstrap CIs

figures/        make_affinity_master_figure.py  -> Figure 2 (Tables S1-S4)
                make_ddg_master_figure.py       -> Figure 3 (Tables S5-S8)
                ig_1vfb_composite.png            Integrated-Gradients + structural
                                                  mapping panel used as an input to
                                                  Figure 2e
```

## What's *not* included, and why

**No `results/` or `paper/` directories** — this repo is code only. The
per-fold prediction CSVs that the figure scripts read (and the exact numbers
they reproduce) are kept private/local; the paper's LaTeX sources are not
published here either. If you clone this repo, `make_ddg_master_figure.py`
and `make_affinity_master_figure.py` will **not run out of the box** — they
expect a `../results/` directory with saved per-fold predictions
(`anchorpkd_final/`, `anchoronly_final/`, `anchordg_final/`, `cosine_final/`,
`wmt_preds/`, each with `s1131_fold_{record,complex5,antigen5}[_s0/_s1/_s2].csv`
columns `complex,fold,true,pred`) that this repo does not ship.

Tables S1–S4 and Figure 2's architecture/backbone/external-benchmark numbers
are **reused from the companion AbAffinity paper's codebase** (the chain-aware
absolute-affinity model, trained and evaluated there under 10-fold CV on
SAAINT-DB) — `make_affinity_master_figure.py` embeds those numbers as literal
arrays (`AB`, `BP`, `PCC_*`, `RMSE_*`) rather than recomputing them.

## Full end-to-end retraining

Retraining the ddG pipeline from raw sequences (rather than just inspecting
the code) additionally requires:
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

Example training call for the main model (pKd-difference anchor, gated
correction head, 3 seeds):
```bash
cd ddg/
python moe_ddg.py --mode dgsub --anchor_pkd --seeds 0 1 2 \
    --pairs_csv <path-to-s1131-pairs.csv> --fold_col fold_id \
    --cutoffs random 90 70 antigencold
```
See `moe_ddg.py --help` for the full set of flags (`--no_moe` for the
anchor-only ablation, `--anchor_dg` for the ΔG-difference anchor variant,
etc.).

## Citation

If you use this code, please cite the paper and the companion
absolute-affinity model:

> Singh, H., Malhotra, A., Srivastava, S.P., Singh, R.K., Gorantla, R.
> Antibody–antigen affinity prediction with chain-aware protein language
> modeling. *bioRxiv* (2026). doi:10.64898/2026.06.19.733375
