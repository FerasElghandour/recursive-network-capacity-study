# Balance helps in some settings; it is not a general scaling advantage

The expanded study completed **1,120 training runs** over four datasets, four
capacity settings and ten seeds, with feature-order and corrected-compute
controls. Balanced trees usually had higher mean accuracy than unbalanced trees,
but the gains depended on the task, width and comparator. Standard dense networks
remained competitive and substantially faster. No universal benefit or new
scaling law was demonstrated.

## What repeated across the study

Among the 16 native dataset/width settings, balanced trees had higher mean test
accuracy than random trees in **14/16**, and skewed trees in **15/16**. These counts
describe the direction of means, not 14 or 15 independent statistically secure
discoveries. Many differences were small: only six random-tree and ten skewed-tree
comparisons had pointwise bootstrap intervals entirely above zero. Those intervals
are not corrected for searching across the grid.

Against the supplementary dense baseline with a corrected training arithmetic
budget, balanced trees had higher mean accuracy in **5/16** settings: all four
Spambase widths and the widest scanned-digit model. Dense networks were more
accurate in the other eleven settings. On Spambase, the width8 and width64
advantages were small and uncertain; width16 and width32 had positive pointwise
intervals. This is a task-dependent inductive bias, not a reason to prefer balance
for every neural network.

## How the advantage changed with size

Hidden width increased from 8 to 64, corresponding to roughly 39–49 times as many
parameters within a dataset. Depth and number of leaves stayed fixed. Training
data and epoch count stayed fixed in the main grid; larger models used more
arithmetic. This is a capacity sweep under one training schedule, not a
compute-optimal scaling-law fit.

| Comparison: balanced minus baseline accuracy | Width 8 | Width 64 | Interpretation |
|---|---:|---:|---|
| Scanned digits, skewed tree | +3.10 pp | +0.71 pp | Benefit shrinks but remains positive on average |
| Pen trajectories, skewed tree | +6.89 pp | +0.13 pp | Large narrow-model mean is sensitive to one poor training run |
| Scanned digits, corrected-compute dense | −2.67 pp | +0.48 pp | Balance catches up and overtakes on this task |
| Pen trajectories, corrected-compute dense | −6.25 pp | −0.63 pp | Balance catches up substantially but still trails dense |
| Email spam, corrected-compute dense | +0.19 pp | +0.08 pp | No growing advantage |
| Satellite, corrected-compute dense | −1.05 pp | −0.14 pp | Gap narrows; no reliable widest-model win |

The direction of a size effect depends on the comparator. On the digit tasks,
extra width tends to make extremely skewed trees less problematic, reducing the benefit of
balancing them. On the two digit tasks, wider balanced models catch up to the
two-hidden-layer dense baseline. Email and satellite do not show a comparable,
clear improvement of balance with scale.

In the prespecified native analysis, the accuracy-advantage slope against skewed
trees on scanned digits was −0.475 pp per parameter doubling, bootstrap interval
[−0.679, −0.282], Holm-adjusted exact sign-flip p=0.0234 across 12 slope tests.
Against the fixed-epoch dense baseline, the scanned-digit slope was +0.499
[+0.401, +0.603], adjusted p=0.0234; pen trajectories +0.878 [+0.534, +1.193],
adjusted p=0.0391. These tests concern fixed-epoch models; the corrected-compute
comparison is a separate descriptive control. Other accuracy slope tests did
not meet the 0.05 adjusted threshold. All slopes are specific to this narrow
range, optimizer and dataset, and are not neural scaling exponents.

### The large pen-digit average needs caution

At width8, one skewed seed reached only 34.08% test accuracy versus 85.33% for
its balanced pair, a +51.26 pp difference. That run is retained because poor
training outcomes are part of architecture reliability, not automatically erroneous data.
Across all ten pairs, the balanced-minus-skewed mean was +6.89 pp, median +2.79 pp,
bootstrap interval [+0.53, +17.39]. Balance won seven seeds, lost two and tied one.
The large mean should not be described as a typical seven-point improvement.
At width64, its +0.13 pp mean interval was [−0.12, +0.39].

## Input grouping changes the result

For scanned digits at width64, the native balanced tree reached 96.52% accuracy
versus 96.04% for the fixed-epoch dense model. After shuffling features before
grouping them into leaves, the balanced tree reached 95.39% versus 95.92% dense.
The change in balanced-minus-dense advantage was −1.01 pp, paired pointwise
interval [−1.42, −0.56]. The balanced advantage over random trees also fell from
+0.68 pp to +0.14 pp. Other tasks had small or reversed changes under shuffling.

Thus the numerical half-split rule alone does not explain performance. A balanced
hierarchy combines particular inputs at particular depths. Suitable grouping and
shorter paths are plausible mechanisms, but they were not independently isolated.

## Widest-model results with corrected training arithmetic

Ten-seed means at tree width64. Dense widths are chosen per dataset using only
parameter and arithmetic counts. Each dense model receives enough examples to
match the tree's corrected linear forward/backward budget. Test loss is natural-log
cross-entropy, lower is better; training times are seconds on one CPU thread.

| Task | Balanced accuracy | Dense accuracy | Balanced loss | Dense loss | Train seconds: balanced / dense |
|---|---:|---:|---:|---:|---:|
| Scanned digits | 96.52% | 96.04% | 0.1333 | 0.1572 | 1.403 / 0.590 |
| Pen trajectories | 96.84% | 97.47% | 0.1969 | 0.1404 | 2.680 / 1.160 |
| Email spam | 94.30% | 94.22% | 0.1759 | 0.1815 | 1.286 / 0.527 |
| Satellite | 90.02% | 90.16% | 0.2648 | 0.2724 | 1.830 / 0.724 |

Widest-model balanced-minus-dense accuracy intervals: scanned digits +0.48 pp
[+0.23, +0.76]; pen −0.63 [−0.85, −0.44]; spam +0.08 [−0.07, +0.25]; satellite
−0.14 [−0.72, +0.49]. These supplementary intervals are descriptive and not
multiplicity-adjusted. Accuracy and loss sometimes disagree, as they did for
satellite at width64: highest-class correctness and probability quality measure
different properties.

Across all native sizes/tasks, the balanced implementation took **2.3–4.0 times
as long to train** as the corrected-compute dense baseline. Balanced, random and
skewed trees had similar elapsed costs; balancing did not establish a consistent
CPU speed benefit. This eager implementation executes branches serially, so it
does not exploit potential parallelism in the shorter balanced graph. Timing
results are implementation- and hardware-specific, not architectural lower bounds.

## Controls, matching and verification

- Main grid: 640 native-order runs plus 320 shuffled-feature runs; 60 epochs,
  batch256, AdamW lr0.003, weight decay0.0001, float32, one CPU thread.
- Supplement: 160 dense runs, typically 61.9–71.2 equivalent epochs, with changed
  sample/update counts to match corrected training arithmetic. It was specified
  and hashed while main training was still underway, before its test metrics.
- Every tree comparison has exactly equal parameter counts and linear arithmetic,
  shared initial parameter tensors, paired splits and minibatch streams. The only
  architectural changes are the child connections and their consequences.
- Dense parameter and forward-MAC differences are at most 3.5%. Corrected dense
  training arithmetic differs from tree totals by at most 0.000219%, due to rounding
  sample exposure to the nearest example. This matches matrix products, not all
  optimizer, activation, bias or memory costs.
- A profiler agreed with the corrected forward/backward linear MAC formulas in
  all 24 checks spanning tasks, small/large widths and tree/dense models.
- Four complete balanced width16/seed0 retrainings, one per dataset, reproduced
  their selected checkpoint tensors, selected epochs, train/validation metrics
  and test logits exactly.

Code, raw data, cleaning indices, split indices, preprocessing constants,
environment/package versions, source hashes, every epoch's curves, all per-run
metrics, selected checkpoints, topologies and held-out logits are saved. See
[README.md](README.md) for offline reproduction commands and data attribution;
[MEASURED_RESULTS.md](MEASURED_RESULTS.md) for full tables and uncertainty.

## What the experiment cannot establish

Ten seeds and four fixed legacy datasets do not establish performance on a
population of tasks. Bootstrap intervals cover algorithm/validation-split
randomness conditional on these test sets; they do not cover new writers, new
geographies, future spam or dataset selection. Validation subsets overlap.
Original scanned-digit test results had already been inspected in the pilot.
New real test metrics were not used during the expanded implementation checks.

The native cell-level Holm correction covers 48 tests per metric. With ten pairs,
the smallest two-sided exact sign-flip p is 2/1024; even the first Holm threshold
is smaller than this. Therefore none of those cell-level corrected tests can
reject at 0.05 by construction. This conservative analysis prevents strong
simultaneous cell claims; it does not show that all effects are absent. Pointwise
intervals, consistency counts and the 12-test size-interaction family are reported
with their different inferential scopes made explicit.

The optimizer and schedule were not tuned per model or capacity. All trees are
simple feed-forward patch-merging structures without residual connections,
normalization, adaptive growth or routing. Width and total arithmetic covary in
the main grid. Peak memory and accelerator performance were not tested. Satellite
neighborhoods come from one scene; its supplied split is not spatial separation.
Spambase required deduplication and exclusion of six conflicting-label records.
These limits rule out claims of a new scaling exponent or a generally superior
training method.
