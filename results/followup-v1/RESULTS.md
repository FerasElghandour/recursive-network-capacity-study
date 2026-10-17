# Paired topology follow-up

A locally frozen follow-up uses 32 new paired seeds, two datasets, width 16, 60 epochs, and native/permuted feature order. All 256 models completed. Test evaluation began after all training and validation-only checkpoint selection.

| Primary comparison | Mean difference (pp) | Descriptive 95% interval (pp) | Holm-adjusted p |
|---|---:|---:|---:|
| optdigits_native | +0.541 | +0.263 to +0.847 | 0.00212 |
| optdigits_order_interaction | +0.021 | -0.421 to +0.456 | 1.00000 |
| pendigits_native | +0.153 | -0.125 to +0.430 | 0.89181 |
| pendigits_order_interaction | -0.004 | -0.407 to +0.401 | 1.00000 |

The Optdigits native-order comparison supports a small balanced-versus-random effect at this width and training budget. Pendigits and the two order interactions do not provide clear evidence of effects in the declared family. The interaction results do not support feature grouping as an established explanation. This comparison does not test a new dense-network advantage.

The datasets and their test results had been inspected previously. New seeds give a within-benchmark follow-up, not external validation or independent preregistration. Bootstrap intervals are descriptive and pointwise; the p values use a four-hypothesis Holm correction and a fixed 200,000-draw paired sign-flip procedure. That procedure assumes sign symmetry under the null. Overlapping train/validation splits and fixed test sets limit generalization.

All 256 saved models reproduced their saved predictions exactly in the recorded Mac runtime. Four representative native-order conditions were freshly retrained and reproduced their checkpoint tensors exactly. Two NumPy archives contain all 256 model states. Full per-run training histories are in `history.csv.gz`; every seed result is in `runs.csv` and `paired.csv`.

```sh
python followup.py --out new-followup
python verify_followup.py --results results/followup-v1
python verify_followup.py --results results/followup-v1 --strict-logits
```

The exact reference uses Python 3.14.3, Torch 2.14.1, NumPy 2.5.4, CPU, one Torch thread on the M2 Max. `requirements-followup-macos.txt` pins the runtime dependency closure. Standard replay permits small cross-runtime differences in logits/loss; strict mode requires exact reference-logit hashes. Timings were collected alongside other local jobs and are not a hardware-performance comparison.

[Frozen settings](../../FOLLOWUP_PROTOCOL.json) · [Raw paired results](paired.csv) · [Full run table](runs.csv) · [Verification](verification.json)
