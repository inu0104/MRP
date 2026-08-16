# Experiments

Paper-facing relevance experiment runners live here. The final table pipeline
uses saved logits from `experiments/text/` and the MRP scripts under
`experiments/table/`.

Current organization:

| Path | Purpose |
| --- | --- |
| `experiments/text/` | Relevance dataset logit-generation scripts. |
| `experiments/table/` | Final MRP/MRC evaluation and table-generation scripts. |
