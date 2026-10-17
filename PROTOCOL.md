# Expanded experiment protocol — fixed before training

## Questions and design

Does balanced recursive structure reliably improve held-out classification loss
and accuracy against random, maximally skewed, and standard dense architectures?
Does its advantage change across model capacities? Is any result sensitive to
input feature grouping?

Primary grid: four datasets × widths 8,16,32,64 × seeds 0–9 × four architectures
= 640 runs. Secondary feature-order control: the same four datasets, all ten seeds
and four architectures, widths 16 and 64, with one fixed feature permutation per
dataset and seed = 320 runs. Total 960. Every run uses 60 epochs and batch 256.
The same seed pairs data splits, initial tree tensors, topology across widths,
feature permutations across models, and minibatch order. All tree modules execute.

## Data and preprocessing

UCI optical digits (64 features,10 classes), pen digits (16,10), Spambase (57,2),
and Statlog Landsat satellite (36,6). Use original train/test files where supplied.
Optical/pen digits have writer-disjoint original tests per dataset documentation.
Satellite is the supplied split, not a spatially separated generalization test.

Spambase: before splitting, remove every feature-identical group with conflicting
labels (3 groups,6 records); retain one copy of each remaining identical record.
This leaves 4,204 unique, unambiguous records from 4,601. Split 20% per class into
a fixed test set, seed 20261015. Retained raw indices saved. No model-based filtering.

For every dataset and seed, independently stratify the remaining original training
pool into 80% train and 20% validation using seed 20261015+seed. These validation
splits overlap across runs; seed intervals do not measure test-population sampling.
Pixels optical/16, pen/100. Spambase log1p then train-only standardization;
satellite train-only standardization. Constant feature scales set to 1.
No test-based fitting, augmentation, class reweighting or imputation. Check all
numeric values finite. Map satellite labels {1,2,3,4,5,7} to {0,1,2,3,4,5}.

Native grouping: optical eight 2x4 image patches in row-major order; pen eight
(x,y) trajectory points; spam eight contiguous groups of sizes 8,7,7,7,7,7,7,7;
satellite nine consecutive four-band pixels. No dummy features or padding.
Control: permute all features once per dataset and seed, then partition them into
the same group sizes. Feed the same feature ordering to dense and tree models.

## Models and matching

Leaf i = ReLU(Linear(group_dim_i,h)). Merge = ReLU(Linear(2h,h)).
Classifier = Linear(h,K). Independent weights, biases, no normalization, residuals,
growth, or adaptive routing. Balanced split floor(n/2)+ceil(n/2), random uniform
split from 1 to n-1 recursively, skewed 1+(n-1), orientation alternates by seed.
All trees have exactly equal parameter/MAC counts and identical initial tensors.
Postorder merge slots fix initialization pairing. Random topology stays the same
across widths and feature-order settings within a dataset and seed.

Dense baseline: standard two-hidden-layer ReLU network. Choose decreasing hidden
widths by minimizing the maximum relative parameter/forward-linear-MAC mismatch
against the corresponding tree, without data. Search first hidden width 8–1024,
second width at least 8, and first/second width ratio at most four, to avoid
degenerate narrow classification bottlenecks. Search sufficient second-width
neighbors around the analytical minimax optimum and feasible boundaries.
Report actual mismatch; tree matching is exact, dense matching approximate.

CPU one thread, float32, deterministic algorithms, AdamW lr=.003, weight_decay=.0001,
batch 256, 60 epochs. No tuning, schedule or early stopping. Randomize all run
execution order using a fixed scheduling seed. Train the full grid before test
evaluation, using minimum validation loss to select a checkpoint for every run.
Synthetic-only smoke checks; no new real test metrics used during implementation.

## Evaluation and statistics

Save per-epoch train/validation loss and accuracy, per-run train/validation/test
metrics, selected epochs, parameters, forward linear MACs, estimated training MACs
(3x forward MACs × samples seen), training-only time, total run time, inference
time, selected checkpoints, logits, splits, preprocessing and environment hashes.
Actual training FLOPs and peak memory are not profiled. Timing is serial eager CPU
execution and includes optimizer/batching; it does not exploit parallel branches.

Primary endpoint: paired balanced-minus-comparator test accuracy for each native
dataset/width (48 comparisons). Also report loss differences in a separate family.
Mean, sample SD, seed wins, percentile bootstrap 95% intervals (20,000 paired seed
resamples), exact two-sided sign-flip tests over 2^10 signs, Holm adjusted p-values
within each metric family. Treat tests as exploratory: sign-flip exchangeability
is an assumption; intervals reflect random initialization/validation splits and
random baseline topology, conditional on these fixed tests and hyperparameters.

Size interaction: per seed regress paired advantage on log2(tree parameter count)
across four widths, separately per dataset/comparator. Report mean slope, paired
bootstrap CI, exact sign-flip p and Holm adjustment over 12 slopes per metric.
Report width64-minus-width8 advantage, also paired over seeds. A slope describes
observed capacity dependence at fixed epochs, not a scaling exponent or a
compute-optimal law. A wider model receives more arithmetic at the same epochs.

Secondary controls: repeat paired differences for shuffled features at widths16
and64; compare native/shuffled advantages with paired seeds, no confirmatory
significance claims. Report consistency across cells and task variation instead
of treating 640 runs as independent evidence for an unlimited dataset population.

## Verification and limitations

Architecture checks: finite gradients, correct shapes, each feature used once,
identical initial tensors and matching budgets for trees at every dataset/width.
Repeat one balanced run at each dataset (width16,seed0) and compare selected
checkpoints and test logits exactly. Hash and count all artifacts.

Limitations: four modest legacy datasets,10 seeds, overlapping validation splits,
fixed test sets, original digits test previously inspected in the pilot, fixed
optimizer not tuned separately per architecture/size, no modern large-scale task,
no tree-depth sweep or accelerator branch parallelism. Parameter width and total
training arithmetic covary. Balance inherently changes depth and receptive field;
feature permutation addresses part of grouping sensitivity but cannot isolate a
pure arithmetic rule from these structural mechanisms. No adaptive-system claims.
