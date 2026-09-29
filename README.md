# AbAffinity-ΔΔG — reproducibility package

Code, pretrained checkpoint, and input data for *"Lightweight Chain-Aware
Modeling for Antibody Affinity and Mutational Effects"* (AbAffinity
absolute-affinity model + affinity-anchored ΔΔG mutation transfer). This is
a cleaned, self-contained subset of a much larger research repository —
enough to actually retrain the ddG model and regenerate its figures, not
just read the code.

## What's included

```
checkpoints/    model.pt — pretrained AbAffinity tri-stream checkpoint
                (frozen ESM-2 + trained interaction network, fold 1 of the
                random-split SAAINT-DB 10-fold CV). This is what the ddG
                pipeline fine-tunes / anchors on (11 MB).

data/           graphinity645_sequences.csv   heavy/light/antigen sequences,
                                               keyed by complex ID
                ddg_input_random.csv,
                ddg_input_70.csv,
                ddg_input_90.csv,
                ddg_input_antigencold.csv     S1131-derived WT/mutant pairs
                                               with fold assignments for each
                                               CV split regime used in the
                                               paper

models/         mutual_strong.py, main_symmetric_mean.py — the frozen-ESM-2
                chain-aware tri-stream architecture (heavy-CDR pooling,
                heavy-light self-attention + fusion gate, gated cross-
                attention, cosine→pKd calibration) and its embedding/
                dataset utilities. Reference copy; also duplicated into
                ddg/ so the ddG scripts import them without extra setup.

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

No `results/` or `paper/` directories are published — this package ships
code + checkpoint + input data, not the paper's saved prediction CSVs or
LaTeX sources.

## Step-by-step: reproduce the main ddG result from scratch

1. **Install dependencies** (Python 3.11; exact versions verified in
   `requirements.txt`)
   ```bash
   pip install -r requirements.txt
   ```
   `pip install torch==2.6.0` alone resolves to a CPU-only wheel. For GPU
   training (strongly recommended — ESM-2 650M forward passes over S1131
   take a long time on CPU), install the matching CUDA build first, e.g.
   for CUDA 12.4 (what this pipeline was run and verified with):
   ```bash
   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt   # picks up everything else
   ```
   Table S5's XGBoost baseline (`ddg/baselines_ddg.py`) is optional — see
   the note in `requirements.txt`; the rest of the pipeline runs without it.

2. **Clone this repo** — the checkpoint (`checkpoints/model.pt`) and input
   pairs (`data/ddg_input_*.csv`) are already included, so no external
   downloads are required for this step.

3. **Train the main model** (pKd-difference anchor + gated correction head,
   3 seeds, matching Table S5 / Figure 3's "AbAffinity-ΔΔG (ours)" row):
   ```bash
   cd ddg/
   python moe_ddg.py --mode dgsub --anchor_pkd --seeds 0 1 2 \
       --pairs_csv ../data/ddg_input_random.csv --fold_col fold_id \
       --cutoffs random
   ```
   On first run this will download ESM-2 650M (~2.5 GB, one-time,
   `facebook/esm2_t33_650M_UR50D`) and build a local token-embedding cache
   next to the input data (`data/esm2_token_cache_650M.pkl`) — this cache
   is not shipped in the repo (it would be several GB) but is regenerated
   automatically and reused on subsequent runs.

4. **Reproduce the anchor-only ablation** (Table S5 "Affinity anchor only",
   Figure 3e) by adding `--no_moe`:
   ```bash
   python moe_ddg.py --mode dgsub --anchor_pkd --no_moe --seeds 0 1 2 \
       --pairs_csv ../data/ddg_input_random.csv --fold_col fold_id --cutoffs random
   ```

5. **Reproduce the other anchor formulations** (Table S7):
   ```bash
   # raw-cosine-difference anchor
   python moe_ddg.py --mode dgsub --seeds 0 1 2 \
       --pairs_csv ../data/ddg_input_random.csv --fold_col fold_id --cutoffs random
   # ΔG-difference anchor
   python moe_ddg.py --mode dgsub --anchor_dg --seeds 0 1 2 \
       --pairs_csv ../data/ddg_input_random.csv --fold_col fold_id --cutoffs random
   ```

6. **Run on the grouped splits** (complex-disjoint / antigen-disjoint) by
   swapping `--pairs_csv` / `--cutoffs` to the corresponding `data/ddg_input_*.csv`
   file — see `moe_ddg.py --help` for the exact fold-column/cutoff naming used
   for each split regime.

7. **Baselines** (Table S5's XGBoost / RandomForest / MLP / LSTM rows):
   ```bash
   python baselines_ddg.py --pairs_csv ../data/ddg_input_random.csv
   python neural_baselines_ddg.py --pairs_csv ../data/ddg_input_random.csv
   ```

8. **Score predictions** with bootstrap CIs:
   ```bash
   python compute_metrics_ci.py --preds <path-to-predictions.csv>
   ```

9. **Regenerate the figures** once you have prediction CSVs saved in the
   directory layout `make_ddg_master_figure.py` expects (see the script's
   `RES`/subfolder names — `anchorpkd_final/`, `anchoronly_final/`,
   `anchordg_final/`, `cosine_final/`):
   ```bash
   cd ../figures/
   python make_ddg_master_figure.py       # -> figure_ddg_results.png / .pdf
   python make_affinity_master_figure.py  # -> figure_affinity_results.png / .pdf
   ```

Expected result for step 3 (mean ± s.d. over the 3 seeds), matching Table S5:

| | Random | Complex-disjoint | Antigen-disjoint |
|---|---|---|---|
| AbAffinity-ΔΔG (pKd anchor, main) | 0.801 ± 0.004 | 0.713 ± 0.017 | 0.675 ± 0.019 |
| Affinity anchor only | 0.722 | 0.687 | 0.650 |

## Path configuration

All data/checkpoint paths are resolved relative to this package by default
and can be overridden with environment variables if you relocate files:

| Variable | Default | Purpose |
|---|---|---|
| `ABAFF_PIPE` | `ddg/` (this repo) | Where `moe_ddg.py` looks for `run_ddg_variants.py` / `run_ddg_seq_improved.py` |
| `ABAFF_DATA` | `data/` | Where `run_ddg_variants.py` looks for `graphinity645_sequences.csv` and the ESM-2 token cache |
| `ABAFF_CKPT` | `checkpoints/model.pt` | Pretrained AbAffinity checkpoint path |

## What's reused from the companion AbAffinity paper, and what's not shipped

Tables S1–S4 and Figure 2's architecture/backbone/external-benchmark numbers
are reused from the companion AbAffinity absolute-affinity paper (trained
and evaluated there under 10-fold CV on SAAINT-DB); `make_affinity_master_figure.py`
embeds those numbers as literal arrays rather than recomputing them.

Not shipped, and not needed for the steps above: the 14 GB ESM-2 token
cache (auto-regenerated), the Graphinity structure-based comparison inputs
(PDB structures, only needed if you also want to reproduce the *structural*
baseline — this package is sequence-only), and the FlexddG synthetic
pretraining pairs (`--pretrain_csv`, an optional pretraining step not used
for the main reported numbers).

## Citation

If you use this code or checkpoint, please cite the paper and the companion
absolute-affinity model:

> Singh, H., Malhotra, A., Srivastava, S.P., Singh, R.K., Gorantla, R.
> Antibody–antigen affinity prediction with chain-aware protein language
> modeling. *bioRxiv* (2026). doi:10.64898/2026.06.19.733375
