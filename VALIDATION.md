# Validation record

Release review: 17 October 2026.

The completed study contains **960 main runs plus 160 compute-control runs**. Balanced trees exceed random-tree mean accuracy in **14/16 native settings**, but exceed the corrected-compute dense baseline in only **5/16**. Dense models remain faster. Fresh release checks exactly reproduce one balanced run on each of four datasets, including checkpoints and held-out predictions.

The listed fresh checks exercise the included public edition. Historical full experiments, representative replays, and synthetic fixtures are identified separately. Test success does not certify commercial deployment.

```sh
python run_grid.py --self-check
```

This finite tabular study does not establish a universal scaling law or general topology advantage. Accuracy and timing depend on data, model budgets, feature order, and hardware. Four fresh representative reproductions do not mean all 1,120 runs were retrained for this release.
