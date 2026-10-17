# Supplementary corrected-training-compute control

Added while the main grid was still training, before any of its test metrics
were evaluated. Motivation is arithmetic accounting, not observed test results.
The original protocol and main-grid implementation remain unchanged and hashed.

Forward-linear-MAC matching plus equal epochs does not exactly match the linear
forward/backward workload of trees and dense networks. Input gradients are not
requested at the first layer. Thus per-example linear training MACs are:

- Tree: 3 × forward MACs − sum(leaf input dimensions) × hidden width.
- Dense: 3 × forward MACs − input dimension × first hidden width.

This excludes bias, nonlinearities, loss, optimizer, memory and indexing, so it
is still an arithmetic model rather than measured hardware FLOPs. It is more
precise than the main protocol's 3x proxy. All three trees match exactly under
both measures. Dense baselines can have materially different corrected totals.

Add one corrected-compute-matched dense run for every native dataset/width/seed:
4 × 4 × 10 = 160 additional runs; full study total 1,120 runs. Same dense widths,
initialization, optimizer, batch size and seed-specific minibatch stream as main
grid. Train on the nearest integer number of samples matching the corresponding
tree's corrected linear training budget for 60 epochs. Permit a partial final
batch. Sample-count rounding error is at most half one example's arithmetic.
Report actual counts, fractional epochs, updates and remaining discrepancy.

Evaluate validation loss after each completed pass over the training data and
after the final partial pass, selecting its minimum. Hold test evaluation until
all 160 supplementary models are trained. Main-grid evaluation may finish first,
but supplementary settings will not be selected using those test metrics.
Execute the supplementary training serially after main-grid execution to avoid
timing interference. This gives the dense network a closely matched parameter
count and corrected linear-training arithmetic, at the cost of different data
exposure and update counts. Report those differences explicitly.

Report seed-paired balanced-minus-supplementary-dense accuracy/loss and bootstrap
intervals for the 16 native dataset/width cells, descriptive only. Check corrected
arithmetic totals, initial parameter equality to main dense models, and preserve
all metrics, histories, logits and checkpoints. No hyperparameter tuning or new
architecture search. This supplementary control does not change the primary
48-comparison family or the original size-interaction analysis.
