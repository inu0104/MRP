#!/usr/bin/env sh
set -eu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-2026}"

COMMON_DATASETS="esci_reranking_us mslr esci scidocs wands alloprof"
COMMON_BASES="raw ts diag spline hcal smart"
COMMON_SEED=2026
COMMON_MODEL_RUN=0
COMMON_MEMBER=0
COMMON_CAL_SEEDS="1 2 3"

python3 experiments/table/evaluate_monotone_reliability_projection.py \
  --datasets ${COMMON_DATASETS} \
  --methods ${COMMON_BASES} \
  --model-run-index ${COMMON_MODEL_RUN} \
  --member-index ${COMMON_MEMBER} \
  --calibration-seeds ${COMMON_CAL_SEEDS} \
  --variants M0_confidence Label1D \
  --lambda-anchor 0 \
  --output-dir results/tables/fixed_decision_mrp_main_no_anchor_gpu0 \
  --seed ${COMMON_SEED}

python3 experiments/table/make_mrp_main_table.py \
  --summary results/tables/fixed_decision_mrp_main_no_anchor_gpu0/mrp_summary.csv \
  --datasets ${COMMON_DATASETS} \
  --bases ${COMMON_BASES} \
  --model-run-index ${COMMON_MODEL_RUN} \
  --member-index ${COMMON_MEMBER} \
  --calibration-seeds ${COMMON_CAL_SEEDS} \
  --output-dir results/tables/fixed_decision_mrp_main_no_anchor_table_gpu0 \
  --tex paper/table/main_mrp_table.tex \
  --refresh-full-metrics \
  --seed ${COMMON_SEED}

python3 experiments/table/make_budgeted_fallback_table.py \
  --summary results/tables/fixed_decision_mrp_main_no_anchor_gpu0/mrp_summary.csv \
  --tex paper/table/budgeted_fallback.tex \
  --csv results/tables/fixed_decision_mrp_main_no_anchor_gpu0/budgeted_fallback_deltas.csv

python3 experiments/table/evaluate_monotone_reliability_projection.py \
  --datasets ${COMMON_DATASETS} \
  --methods ${COMMON_BASES} \
  --model-run-index ${COMMON_MODEL_RUN} \
  --member-index ${COMMON_MEMBER} \
  --calibration-seeds ${COMMON_CAL_SEEDS} \
  --variants M0_confidence LabelConstant PerLabelIsotonic Shared1D Label1D Label2D \
  --lambda-anchor 0 \
  --output-dir results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0 \
  --seed ${COMMON_SEED}

python3 experiments/table/make_mrp_ablation_table.py \
  --overall results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_overall.csv \
  --summary results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_summary.csv \
  --tex paper/table/mrp_ablation_results.tex \
  --csv results/tables/monotone_reliability_projection_ablation_no_anchor_gpu0/mrp_ablation_compact.csv

python3 experiments/table/analyze_mrp_structure.py \
  --datasets ${COMMON_DATASETS} \
  --bases ${COMMON_BASES} \
  --max-seeds 3 \
  --output-dir results/analysis/mrp_structure_smart_plus_alloprof_gpu0 \
  --seed ${COMMON_SEED}

python3 experiments/table/evaluate_mrp_simplex_projection.py \
  --datasets ${COMMON_DATASETS} \
  --methods ${COMMON_BASES} \
  --model-run-index ${COMMON_MODEL_RUN} \
  --member-index ${COMMON_MEMBER} \
  --calibration-seeds ${COMMON_CAL_SEEDS} \
  --lambda-anchor 0 \
  --output-dir results/tables/mrp_simplex_projection_gpu0 \
  --seed ${COMMON_SEED}

python3 experiments/table/make_mrp_space_artifacts.py \
  --projection-dir results/tables/mrp_simplex_projection_gpu0 \
  --table-dir paper/table \
  --figure-dir paper/figure
