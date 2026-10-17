# Inputs and data handling

Public test fixtures and generated experimental artifacts are included where described. Private client records, account data, credentials, original submission archives, and unrelated business documents are excluded. Runtime data should remain outside version control.

Code imports external libraries through their normal installations; it does not vendor those libraries. Their licenses and notices remain with the installed distributions. No blanket software license is assigned to third-party material.


## Public benchmark datasets

The unmodified source data in `data/` are licensed CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Original source documentation is retained. Attribution:

- Alpaydin, E., and Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI. https://doi.org/10.24432/C50P49
- Alpaydin, E., and Alimoglu, F. (1996). Pen-Based Recognition of Handwritten Digits. UCI. https://doi.org/10.24432/C5MG6K
- Hopkins, M., Reeber, E., Forman, G., and Suermondt, J. (1999). Spambase. UCI. https://doi.org/10.24432/C53G6X
- Srinivasan, A. (1993). Statlog (Landsat Satellite). UCI. https://doi.org/10.24432/C55887

Preprocessing changes are implemented in `run_grid.py`: Spambase conflicting-label and duplicate handling, fixed train/test pools, training-only normalization, and recorded feature-order controls. Generated histories, metrics, selected checkpoints, and predictions describe this study rather than new source measurements.
