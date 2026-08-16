# Table Assembly

Paper-facing MRP table builders live here.

Current final pipeline:

- `experiments/table/run_fixed_decision_mrp_main.sh`

The script regenerates the main MRP table, budgeted fallback table, structural
ablation table, label-conditional residual diagnostic, label-wise curve figure,
and MRC simplex-projection table from the saved relevance logits.
