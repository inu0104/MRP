# Text and Relevance Data Preparation

These scripts prepare saved logits for the relevance datasets used by the final MRP paper pipeline. Raw datasets are downloaded or read locally by the preparation scripts, while generated logits are written under `.local/runs/relevance_projection/...` and ignored by git.

Final datasets:

- Amazon ESCI: product relevance classification.
- MSLR-WEB10K: graded web relevance prediction.
- Alloprof-Rerank: question-document relevance reranking.
- ESCI-Rerank-US: binary e-commerce reranking.
- WANDS: product search relevance prediction.
- SciDocs: scientific document reranking.

The table-generation scripts expect each prepared run directory to contain `logits_and_labels.npz` and `metadata.json`. These local run directories are not tracked because they can be regenerated and may be large.

Current preparation scripts:

- `amazon_esci_logit_prep.py`
- `esci_rerank_us_logit_prep.py`
- `mslr_web10k_logit_prep.py`
- `mteb_qrels_logit_prep.py`
- `mteb_reranking_logit_prep.py`
- `wands_logit_prep.py`

`mslr_web10k_lightgbm_logit_prep.py` writes compatible saved logits from a stronger, better-calibrated LightGBM base and backs the robustness check reported in the paper. All paper tables are generated from the six datasets listed above.
