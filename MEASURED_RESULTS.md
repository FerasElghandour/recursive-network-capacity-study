# Expanded balanced-recursion experiment: measured tables

Completed 960 runs; 4 datasets, 4 sizes, 10 paired seeds; native grid plus feature-order controls.

Total recorded elapsed time: 23.2 minutes.

All means below use validation-selected checkpoints. Main-grid models trained for 60 epochs; the supplementary control adjusts sample exposure.
Accuracy ± values are sample SD across seeds, in percentage points.
Paired intervals are pointwise percentile bootstrap 95% intervals over ten seed pairs.
They are conditional on fixed test sets; validation splits overlap.

## Native feature order: accuracy, loss and time

| Dataset | Width | Model | Parameters | Test accuracy (%) ± SD | Test loss ± SD | Train seconds ± SD |
|---|---:|---|---:|---:|---:|---:|
| optdigits | 8 | balanced | 1,618 | 91.69 ± 0.99 | 0.2997 ± 0.0370 | 0.808 ± 0.051 |
| optdigits | 8 | random | 1,618 | 89.72 ± 2.48 | 0.3731 ± 0.0960 | 0.799 ± 0.018 |
| optdigits | 8 | skewed | 1,618 | 88.59 ± 1.73 | 0.4224 ± 0.0624 | 0.794 ± 0.028 |
| optdigits | 8 | mlp | 1,575 | 94.09 ± 0.44 | 0.1982 ± 0.0190 | 0.211 ± 0.005 |
| optdigits | 16 | balanced | 5,018 | 94.80 ± 0.61 | 0.2016 ± 0.0312 | 0.934 ± 0.026 |
| optdigits | 16 | random | 5,018 | 93.79 ± 1.02 | 0.2527 ± 0.0522 | 0.938 ± 0.025 |
| optdigits | 16 | skewed | 5,018 | 92.82 ± 0.92 | 0.2850 ± 0.0415 | 0.936 ± 0.019 |
| optdigits | 16 | mlp | 4,935 | 95.07 ± 0.42 | 0.1663 ± 0.0162 | 0.249 ± 0.007 |
| optdigits | 32 | balanced | 17,194 | 95.65 ± 0.42 | 0.1687 ± 0.0172 | 1.041 ± 0.021 |
| optdigits | 32 | random | 17,194 | 95.26 ± 0.50 | 0.2065 ± 0.0295 | 1.066 ± 0.078 |
| optdigits | 32 | skewed | 17,194 | 94.96 ± 0.82 | 0.2382 ± 0.0256 | 1.049 ± 0.036 |
| optdigits | 32 | mlp | 17,045 | 95.66 ± 0.31 | 0.1661 ± 0.0162 | 0.333 ± 0.009 |
| optdigits | 64 | balanced | 63,050 | 96.52 ± 0.33 | 0.1333 ± 0.0108 | 1.403 ± 0.070 |
| optdigits | 64 | random | 63,050 | 95.84 ± 0.87 | 0.1602 ± 0.0312 | 1.384 ± 0.032 |
| optdigits | 64 | skewed | 63,050 | 95.81 ± 0.61 | 0.1950 ± 0.0321 | 1.397 ± 0.079 |
| optdigits | 64 | mlp | 62,799 | 96.04 ± 0.39 | 0.1565 ± 0.0151 | 0.545 ± 0.015 |
| pendigits | 8 | balanced | 1,234 | 88.89 ± 3.19 | 0.3796 ± 0.0783 | 1.577 ± 0.039 |
| pendigits | 8 | random | 1,234 | 88.88 ± 4.94 | 0.3952 ± 0.1349 | 1.558 ± 0.027 |
| pendigits | 8 | skewed | 1,234 | 82.00 ± 17.00 | 0.5925 ± 0.4609 | 1.584 ± 0.066 |
| pendigits | 8 | mlp | 1,195 | 94.81 ± 0.38 | 0.2191 ± 0.0187 | 0.418 ± 0.005 |
| pendigits | 16 | balanced | 4,250 | 95.27 ± 0.53 | 0.2418 ± 0.0286 | 1.807 ± 0.034 |
| pendigits | 16 | random | 4,250 | 95.19 ± 0.31 | 0.2298 ± 0.0461 | 1.845 ± 0.034 |
| pendigits | 16 | skewed | 4,250 | 94.27 ± 1.17 | 0.2866 ± 0.0337 | 1.838 ± 0.041 |
| pendigits | 16 | mlp | 4,191 | 96.46 ± 0.27 | 0.1756 ± 0.0205 | 0.506 ± 0.013 |
| pendigits | 32 | balanced | 15,658 | 96.56 ± 0.30 | 0.2101 ± 0.0307 | 2.081 ± 0.052 |
| pendigits | 32 | random | 15,658 | 96.36 ± 0.59 | 0.2028 ± 0.0482 | 2.095 ± 0.064 |
| pendigits | 32 | skewed | 15,658 | 96.23 ± 0.38 | 0.2530 ± 0.0608 | 2.107 ± 0.038 |
| pendigits | 32 | mlp | 15,544 | 97.10 ± 0.38 | 0.1518 ± 0.0148 | 0.693 ± 0.110 |
| pendigits | 64 | balanced | 59,978 | 96.84 ± 0.29 | 0.1969 ± 0.0183 | 2.680 ± 0.085 |
| pendigits | 64 | random | 59,978 | 96.79 ± 0.49 | 0.1857 ± 0.0372 | 2.679 ± 0.069 |
| pendigits | 64 | skewed | 59,978 | 96.71 ± 0.39 | 0.2242 ± 0.0691 | 2.772 ± 0.104 |
| pendigits | 64 | mlp | 59,763 | 97.44 ± 0.15 | 0.1399 ± 0.0137 | 1.134 ± 0.048 |
| spambase | 8 | balanced | 1,490 | 94.33 ± 0.44 | 0.1724 ± 0.0095 | 0.738 ± 0.018 |
| spambase | 8 | random | 1,490 | 94.16 ± 0.35 | 0.1718 ± 0.0079 | 0.756 ± 0.031 |
| spambase | 8 | skewed | 1,490 | 94.22 ± 0.53 | 0.1738 ± 0.0091 | 0.785 ± 0.095 |
| spambase | 8 | mlp | 1,438 | 94.14 ± 0.29 | 0.1801 ± 0.0091 | 0.188 ± 0.005 |
| spambase | 16 | balanced | 4,770 | 94.42 ± 0.39 | 0.1714 ± 0.0108 | 0.848 ± 0.019 |
| spambase | 16 | random | 4,770 | 94.08 ± 0.64 | 0.1748 ± 0.0101 | 0.848 ± 0.025 |
| spambase | 16 | skewed | 4,770 | 94.05 ± 0.64 | 0.1742 ± 0.0147 | 0.867 ± 0.047 |
| spambase | 16 | mlp | 4,688 | 94.02 ± 0.36 | 0.1833 ± 0.0062 | 0.225 ± 0.006 |
| spambase | 32 | balanced | 16,706 | 94.65 ± 0.22 | 0.1722 ± 0.0070 | 0.950 ± 0.023 |
| spambase | 32 | random | 16,706 | 94.33 ± 0.39 | 0.1738 ± 0.0063 | 0.953 ± 0.024 |
| spambase | 32 | skewed | 16,706 | 94.27 ± 0.40 | 0.1747 ± 0.0113 | 0.950 ± 0.027 |
| spambase | 32 | mlp | 16,568 | 94.14 ± 0.40 | 0.1865 ± 0.0083 | 0.311 ± 0.012 |
| spambase | 64 | balanced | 62,082 | 94.30 ± 0.21 | 0.1759 ± 0.0071 | 1.286 ± 0.146 |
| spambase | 64 | random | 62,082 | 94.27 ± 0.32 | 0.1767 ± 0.0078 | 1.271 ± 0.057 |
| spambase | 64 | skewed | 62,082 | 94.53 ± 0.37 | 0.1731 ± 0.0092 | 1.244 ± 0.045 |
| spambase | 64 | mlp | 61,843 | 94.22 ± 0.33 | 0.1815 ± 0.0066 | 0.515 ± 0.060 |
| satimage | 8 | balanced | 1,502 | 87.94 ± 0.51 | 0.3082 ± 0.0108 | 1.051 ± 0.031 |
| satimage | 8 | random | 1,502 | 88.06 ± 0.61 | 0.3108 ± 0.0125 | 1.054 ± 0.030 |
| satimage | 8 | skewed | 1,502 | 87.34 ± 0.88 | 0.3237 ± 0.0121 | 1.039 ± 0.017 |
| satimage | 8 | mlp | 1,452 | 88.88 ± 0.37 | 0.2892 ± 0.0054 | 0.255 ± 0.007 |
| satimage | 16 | balanced | 5,046 | 88.96 ± 0.45 | 0.2891 ± 0.0075 | 1.250 ± 0.033 |
| satimage | 16 | random | 5,046 | 89.27 ± 0.50 | 0.2891 ± 0.0110 | 1.232 ± 0.030 |
| satimage | 16 | skewed | 5,046 | 88.51 ± 0.53 | 0.3010 ± 0.0115 | 1.243 ± 0.035 |
| satimage | 16 | mlp | 4,955 | 89.22 ± 0.44 | 0.2786 ± 0.0091 | 0.305 ± 0.030 |
| satimage | 32 | balanced | 18,278 | 89.59 ± 0.52 | 0.2799 ± 0.0135 | 1.382 ± 0.031 |
| satimage | 32 | random | 18,278 | 89.34 ± 0.42 | 0.2831 ± 0.0096 | 1.334 ± 0.037 |
| satimage | 32 | skewed | 18,278 | 89.40 ± 0.59 | 0.2878 ± 0.0152 | 1.383 ± 0.033 |
| satimage | 32 | mlp | 18,129 | 90.12 ± 0.47 | 0.2708 ± 0.0084 | 0.413 ± 0.018 |
| satimage | 64 | balanced | 69,318 | 90.02 ± 0.59 | 0.2648 ± 0.0050 | 1.830 ± 0.051 |
| satimage | 64 | random | 69,318 | 89.92 ± 0.49 | 0.2713 ± 0.0144 | 1.794 ± 0.066 |
| satimage | 64 | skewed | 69,318 | 89.68 ± 0.30 | 0.2868 ± 0.0120 | 1.771 ± 0.054 |
| satimage | 64 | mlp | 69,056 | 90.16 ± 0.68 | 0.2724 ± 0.0144 | 0.697 ± 0.037 |

## Native paired accuracy effects

Positive differences favor balanced. Holm correction covers all 48 native cell comparisons.

| Dataset | Width | Comparator | Balanced advantage (pp) | 95% interval (pp) | Wins/10 | Holm p |
|---|---:|---|---:|---:|---:|---:|
| optdigits | 8 | random | +1.96 | [+0.71, +3.27] | 8 | 0.8438 |
| optdigits | 8 | skewed | +3.10 | [+1.70, +4.51] | 9 | 0.2461 |
| optdigits | 8 | mlp | -2.40 | [-3.04, -1.79] | 0 | 0.0938 |
| optdigits | 16 | random | +1.01 | [+0.37, +1.60] | 8 | 0.7949 |
| optdigits | 16 | skewed | +1.99 | [+1.60, +2.39] | 10 | 0.0938 |
| optdigits | 16 | mlp | -0.27 | [-0.63, +0.09] | 4 | 1.0000 |
| optdigits | 32 | random | +0.40 | [+0.00, +0.74] | 9 | 1.0000 |
| optdigits | 32 | skewed | +0.69 | [+0.20, +1.26] | 7 | 0.9062 |
| optdigits | 32 | mlp | -0.01 | [-0.17, +0.17] | 4 | 1.0000 |
| optdigits | 64 | random | +0.68 | [+0.14, +1.28] | 7 | 1.0000 |
| optdigits | 64 | skewed | +0.71 | [+0.26, +1.17] | 8 | 0.8750 |
| optdigits | 64 | mlp | +0.48 | [+0.24, +0.76] | 9 | 0.1680 |
| pendigits | 8 | random | +0.01 | [-3.09, +3.30] | 5 | 1.0000 |
| pendigits | 8 | skewed | +6.89 | [+0.53, +17.39] | 7 | 1.0000 |
| pendigits | 8 | mlp | -5.93 | [-7.76, -3.98] | 0 | 0.0938 |
| pendigits | 16 | random | +0.08 | [-0.22, +0.41] | 5 | 1.0000 |
| pendigits | 16 | skewed | +1.00 | [+0.37, +1.70] | 8 | 0.5938 |
| pendigits | 16 | mlp | -1.19 | [-1.53, -0.88] | 0 | 0.0938 |
| pendigits | 32 | random | +0.20 | [-0.11, +0.50] | 6 | 1.0000 |
| pendigits | 32 | skewed | +0.33 | [+0.04, +0.62] | 7 | 1.0000 |
| pendigits | 32 | mlp | -0.53 | [-0.89, -0.23] | 1 | 0.4570 |
| pendigits | 64 | random | +0.05 | [-0.30, +0.40] | 4 | 1.0000 |
| pendigits | 64 | skewed | +0.13 | [-0.12, +0.39] | 5 | 1.0000 |
| pendigits | 64 | mlp | -0.60 | [-0.82, -0.41] | 0 | 0.0938 |
| spambase | 8 | random | +0.17 | [+0.00, +0.42] | 5 | 1.0000 |
| spambase | 8 | skewed | +0.11 | [-0.11, +0.30] | 7 | 1.0000 |
| spambase | 8 | mlp | +0.19 | [-0.13, +0.45] | 8 | 1.0000 |
| spambase | 16 | random | +0.34 | [+0.02, +0.74] | 7 | 1.0000 |
| spambase | 16 | skewed | +0.37 | [+0.01, +0.78] | 7 | 1.0000 |
| spambase | 16 | mlp | +0.40 | [+0.10, +0.70] | 7 | 1.0000 |
| spambase | 32 | random | +0.32 | [+0.13, +0.50] | 7 | 0.8438 |
| spambase | 32 | skewed | +0.38 | [+0.14, +0.61] | 7 | 0.8438 |
| spambase | 32 | mlp | +0.51 | [+0.20, +0.83] | 7 | 0.8750 |
| spambase | 64 | random | +0.04 | [-0.14, +0.20] | 5 | 1.0000 |
| spambase | 64 | skewed | -0.23 | [-0.38, -0.08] | 2 | 0.8750 |
| spambase | 64 | mlp | +0.08 | [-0.07, +0.25] | 5 | 1.0000 |
| satimage | 8 | random | -0.12 | [-0.64, +0.40] | 4 | 1.0000 |
| satimage | 8 | skewed | +0.60 | [+0.18, +1.11] | 8 | 0.8438 |
| satimage | 8 | mlp | -0.94 | [-1.43, -0.50] | 1 | 0.2461 |
| satimage | 16 | random | -0.31 | [-0.60, +0.02] | 2 | 1.0000 |
| satimage | 16 | skewed | +0.44 | [-0.00, +0.87] | 7 | 1.0000 |
| satimage | 16 | mlp | -0.26 | [-0.51, +0.04] | 2 | 1.0000 |
| satimage | 32 | random | +0.25 | [-0.03, +0.53] | 7 | 1.0000 |
| satimage | 32 | skewed | +0.18 | [-0.16, +0.54] | 6 | 1.0000 |
| satimage | 32 | mlp | -0.53 | [-0.82, -0.26] | 1 | 0.3125 |
| satimage | 64 | random | +0.10 | [-0.26, +0.46] | 6 | 1.0000 |
| satimage | 64 | skewed | +0.34 | [-0.05, +0.73] | 8 | 1.0000 |
| satimage | 64 | mlp | -0.14 | [-0.72, +0.49] | 3 | 1.0000 |

## Change of accuracy advantage with size

Slope is percentage points per doubling of tree parameter count; this is not a scaling exponent.
Endpoint change is the width64 advantage minus the width8 advantage. Positive means balance gains relative accuracy with width.
Holm correction covers 12 slope tests per metric.

| Dataset | Comparator | Slope (pp/doubling) | Slope 95% interval | Width64−width8 change (pp) | Change 95% interval | Holm slope p |
|---|---|---:|---:|---:|---:|---:|
| optdigits | random | -0.25 | [-0.42, -0.09] | -1.28 | [-2.26, -0.40] | 0.2461 |
| optdigits | skewed | -0.48 | [-0.68, -0.28] | -2.39 | [-3.72, -1.07] | 0.0234 |
| optdigits | mlp | +0.50 | [+0.40, +0.60] | +2.89 | [+2.28, +3.57] | 0.0234 |
| pendigits | random | +0.01 | [-0.50, +0.51] | +0.05 | [-3.27, +3.23] | 1.0000 |
| pendigits | skewed | -1.11 | [-2.75, -0.08] | -6.75 | [-17.30, -0.26] | 0.4238 |
| pendigits | mlp | +0.88 | [+0.53, +1.19] | +5.32 | [+3.21, +7.28] | 0.0391 |
| spambase | random | -0.02 | [-0.08, +0.02] | -0.13 | [-0.48, +0.13] | 1.0000 |
| spambase | skewed | -0.06 | [-0.10, -0.01] | -0.33 | [-0.55, -0.13] | 0.3594 |
| spambase | mlp | -0.01 | [-0.07, +0.06] | -0.11 | [-0.42, +0.31] | 1.0000 |
| satimage | random | +0.07 | [-0.01, +0.15] | +0.21 | [-0.29, +0.73] | 0.8008 |
| satimage | skewed | -0.06 | [-0.14, +0.02] | -0.27 | [-0.83, +0.25] | 1.0000 |
| satimage | mlp | +0.11 | [-0.00, +0.21] | +0.80 | [+0.09, +1.41] | 0.4688 |

## Shuffled-feature control: paired accuracy effects

Secondary sensitivity analysis; these intervals and tests are not corrected for multiplicity.

| Dataset | Width | Comparator | Balanced advantage (pp) | 95% interval (pp) | Wins/10 |
|---|---:|---|---:|---:|---:|
| optdigits | 16 | random | +0.95 | [+0.22, +1.67] | 7 |
| optdigits | 16 | skewed | +1.11 | [+0.54, +1.69] | 8 |
| optdigits | 16 | mlp | -1.86 | [-2.34, -1.31] | 0 |
| optdigits | 64 | random | +0.14 | [-0.14, +0.51] | 4 |
| optdigits | 64 | skewed | +0.28 | [-0.03, +0.58] | 8 |
| optdigits | 64 | mlp | -0.52 | [-0.85, -0.21] | 2 |
| pendigits | 16 | random | +0.20 | [-0.54, +0.89] | 5 |
| pendigits | 16 | skewed | +0.99 | [+0.30, +1.72] | 7 |
| pendigits | 16 | mlp | -1.55 | [-2.08, -1.06] | 0 |
| pendigits | 64 | random | -0.03 | [-0.18, +0.14] | 4 |
| pendigits | 64 | skewed | +0.10 | [-0.27, +0.49] | 5 |
| pendigits | 64 | mlp | -0.64 | [-0.94, -0.38] | 0 |
| spambase | 16 | random | -0.01 | [-0.25, +0.24] | 4 |
| spambase | 16 | skewed | -0.14 | [-0.36, +0.07] | 3 |
| spambase | 16 | mlp | +0.19 | [-0.17, +0.58] | 5 |
| spambase | 64 | random | +0.02 | [-0.29, +0.38] | 4 |
| spambase | 64 | skewed | +0.11 | [-0.13, +0.37] | 3 |
| spambase | 64 | mlp | +0.01 | [-0.42, +0.43] | 5 |
| satimage | 16 | random | +0.19 | [-0.24, +0.69] | 4 |
| satimage | 16 | skewed | +0.51 | [+0.07, +0.95] | 7 |
| satimage | 16 | mlp | -0.44 | [-0.76, -0.08] | 2 |
| satimage | 64 | random | +0.16 | [-0.09, +0.40] | 7 |
| satimage | 64 | skewed | +0.23 | [-0.21, +0.73] | 6 |
| satimage | 64 | mlp | -0.60 | [-1.15, -0.14] | 2 |

## Budget matching

| Dataset | Width | Tree parameters | Tree MACs/example | MLP parameters | MLP MACs/example | Parameter mismatch | MAC mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| optdigits | 8 | 1,618 | 1,488 | 1,575 | 1,535 | -2.66% | +3.16% |
| optdigits | 16 | 5,018 | 4,768 | 4,935 | 4,847 | -1.65% | +1.66% |
| optdigits | 32 | 17,194 | 16,704 | 17,045 | 16,844 | -0.87% | +0.84% |
| optdigits | 64 | 63,050 | 62,080 | 62,799 | 62,319 | -0.40% | +0.38% |
| pendigits | 8 | 1,234 | 1,104 | 1,195 | 1,136 | -3.16% | +2.90% |
| pendigits | 16 | 4,250 | 4,000 | 4,191 | 4,063 | -1.39% | +1.57% |
| pendigits | 32 | 15,658 | 15,168 | 15,544 | 15,284 | -0.73% | +0.76% |
| pendigits | 64 | 59,978 | 59,008 | 59,763 | 59,202 | -0.36% | +0.33% |
| spambase | 8 | 1,490 | 1,368 | 1,438 | 1,404 | -3.49% | +2.63% |
| spambase | 16 | 4,770 | 4,528 | 4,688 | 4,603 | -1.72% | +1.66% |
| spambase | 32 | 16,706 | 16,224 | 16,568 | 16,364 | -0.83% | +0.86% |
| spambase | 64 | 62,082 | 61,120 | 61,843 | 61,350 | -0.38% | +0.38% |
| satimage | 8 | 1,502 | 1,360 | 1,452 | 1,404 | -3.33% | +3.24% |
| satimage | 16 | 5,046 | 4,768 | 4,955 | 4,847 | -1.80% | +1.66% |
| satimage | 32 | 18,278 | 17,728 | 18,129 | 17,874 | -0.82% | +0.82% |
| satimage | 64 | 69,318 | 68,224 | 69,056 | 68,484 | -0.38% | +0.38% |

## Inference limits

The smallest attainable two-sided exact sign-flip p-value with ten pairs is 2/1024 = 0.001953.
Across a family of 48 tests, Holm cannot reject even its first test at 0.05 with this minimum p-value.
Consequently, cell-level Holm non-rejection here is not evidence that effects are absent.
Pointwise intervals and consistent effect directions remain descriptive, not simultaneous confidence claims.
The 12-test interaction family can resolve the smallest exact p-values; its power is still limited.

The sweep changes width and arithmetic together at fixed epochs. It does not estimate a compute-optimal scaling law.
Four legacy datasets and fixed optimizer settings cannot establish general superiority across tasks.
Topology changes path depths and receptive fields; a feature-order control cannot isolate these mechanisms completely.
Native and shuffled conditions share test examples and seed splits and are not independent replications.

See PROTOCOL.md, README.md, FINDINGS.md, raw CSVs, checkpoints and predictions for the complete record.

## Supplementary dense baseline matched on corrected training arithmetic

160 additional runs, specified before main-grid test evaluation. Total study: 1,120 training runs.
Dense input layers have no input-gradient calculation, so corrected training MACs are 3×forward minus input-layer MACs.
Dense sample counts are adjusted to match the tree total to the nearest example; data exposure and update counts differ.
This accounts for linear forward/backward products, not optimizer, bias, activation or memory costs.
See COMPUTE_CONTROL_PROTOCOL.md. Intervals below are pointwise/descriptive without multiplicity correction.

| Dataset | Width | Corrected-compute dense accuracy (%) | Dense loss | Dense equivalent epochs | Balanced−dense accuracy (pp) | 95% interval (pp) |
|---|---:|---:|---:|---:|---:|---:|
| optdigits | 8 | 94.35 ± 0.39 | 0.1971 | 69.97 | -2.67 | [-3.29, -2.05] |
| optdigits | 16 | 95.20 ± 0.46 | 0.1661 | 69.86 | -0.40 | [-0.81, +0.02] |
| optdigits | 32 | 95.66 ± 0.30 | 0.1670 | 66.70 | -0.01 | [-0.18, +0.16] |
| optdigits | 64 | 96.04 ± 0.39 | 0.1572 | 66.85 | +0.48 | [+0.23, +0.76] |
| pendigits | 8 | 95.13 ± 0.37 | 0.2092 | 68.23 | -6.25 | [-8.06, -4.33] |
| pendigits | 16 | 96.44 ± 0.21 | 0.1758 | 65.85 | -1.16 | [-1.45, -0.89] |
| pendigits | 32 | 97.15 ± 0.38 | 0.1503 | 63.01 | -0.59 | [-0.93, -0.26] |
| pendigits | 64 | 97.47 ± 0.19 | 0.1404 | 61.87 | -0.63 | [-0.85, -0.44] |
| spambase | 8 | 94.14 ± 0.29 | 0.1801 | 71.25 | +0.19 | [-0.13, +0.45] |
| spambase | 16 | 94.02 ± 0.36 | 0.1833 | 69.74 | +0.40 | [+0.10, +0.70] |
| spambase | 32 | 94.14 ± 0.40 | 0.1865 | 65.21 | +0.51 | [+0.20, +0.83] |
| spambase | 64 | 94.22 ± 0.33 | 0.1815 | 66.68 | +0.08 | [-0.07, +0.25] |
| satimage | 8 | 88.99 ± 0.30 | 0.2869 | 67.96 | -1.05 | [-1.51, -0.65] |
| satimage | 16 | 89.22 ± 0.44 | 0.2787 | 65.57 | -0.26 | [-0.51, +0.04] |
| satimage | 32 | 90.12 ± 0.47 | 0.2708 | 66.99 | -0.53 | [-0.82, -0.26] |
| satimage | 64 | 90.16 ± 0.68 | 0.2724 | 64.18 | -0.14 | [-0.72, +0.49] |
