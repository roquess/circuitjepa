# CircuitJEPA results summary

Mean ± sample std (ddof=1) over seeds. `collapsed` = number of seeds flagged collapsed
(diag gap < 0.05 and target total variance < 5 % of the same-seed untrained baseline).
Probe: linear map fitted on exact target embeddings of the training set, evaluated
on predicted (`pred`) vs un-advanced context (`unadv`) embeddings; target = ⟨Z_i⟩(t₀+s).

## Backend: exact (m = 64)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.539 ± 0.341 | 0.665 ± 0.186 | 1.552 ± 0.370 | 1.240 ± 0.177 | 0/10 | 0.0483 ± 0.0058 | 0.0715 ± 0.0258 | 0.461 ± 0.259 |
| no_ema | 0.369 ± 0.332 | 0.486 ± 0.286 | 1.288 ± 0.515 | 1.253 ± 0.118 | 0/10 | 0.0548 ± 0.0098 | 0.0691 ± 0.0236 | 0.639 ± 0.273 |
| no_predictor | 0.031 ± 0.040 | 0.116 ± 0.118 | 0.712 ± 0.344 | 1.189 ± 0.087 | 0/10 | 1.9033 ± 5.8633 | 1.9033 ± 5.8633 | 0.000 ± 0.000 |
| untrained | 0.008 ± 0.013 | -0.021 ± 0.047 | 1.366 ± 0.184 | 1.202 ± 0.061 | 0/10 | 0.0788 ± 0.0139 | 0.0815 ± 0.0189 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.661 ± 0.336 | 0.0424 ± 0.0053 | 0.0566 ± 0.0103 |
| full | 2 | 0.504 ± 0.389 | 0.0471 ± 0.0062 | 0.0831 ± 0.0314 |
| full | 3 | 0.478 ± 0.408 | 0.0541 ± 0.0108 | 0.0716 ± 0.0350 |
| no_ema | 1 | 0.433 ± 0.390 | 0.0528 ± 0.0171 | 0.0522 ± 0.0100 |
| no_ema | 2 | 0.309 ± 0.381 | 0.0533 ± 0.0106 | 0.0839 ± 0.0336 |
| no_ema | 3 | 0.378 ± 0.363 | 0.0578 ± 0.0147 | 0.0675 ± 0.0262 |
| no_predictor | 1 | 0.072 ± 0.095 | 1.9115 ± 5.8969 | 1.9115 ± 5.8969 |
| no_predictor | 2 | 0.022 ± 0.047 | 1.9249 ± 5.9174 | 1.9249 ± 5.9174 |
| no_predictor | 3 | 0.009 ± 0.018 | 1.8752 ± 5.7828 | 1.8752 ± 5.7828 |
| untrained | 1 | 0.011 ± 0.023 | 0.0802 ± 0.0203 | 0.0590 ± 0.0091 |
| untrained | 2 | 0.009 ± 0.018 | 0.0785 ± 0.0158 | 0.0995 ± 0.0290 |
| untrained | 3 | 0.004 ± 0.014 | 0.0780 ± 0.0172 | 0.0810 ± 0.0210 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 64 | 0.0001 | 59/64 | 1.9e-100 | True | True |
| full_seed1_vs_untrained_seed1 | 64 | 0.0001 | 21/64 | 2.5e-22 | True | True |
| full_seed2_vs_untrained_seed2 | 64 | 0.0001 | 10/64 | 6.1e-08 | True | True |
| full_seed3_vs_untrained_seed3 | 64 | 0.0001 | 5/64 | 0.0033 | True | True |
| full_seed4_vs_untrained_seed4 | 64 | 0.0001 | 28/64 | 1.7e-33 | True | True |
| full_seed5_vs_untrained_seed5 | 64 | 0.0001 | 36/64 | 6.9e-48 | True | True |
| full_seed6_vs_untrained_seed6 | 64 | 0.0001 | 50/64 | 1.9e-77 | True | True |
| full_seed7_vs_untrained_seed7 | 64 | 0.0001 | 59/64 | 1.9e-100 | True | True |
| full_seed8_vs_untrained_seed8 | 64 | 0.0001 | 15/64 | 6.3e-14 | True | True |
| full_seed9_vs_untrained_seed9 | 64 | 0.0001 | 62/64 | 2e-109 | True | True |

## Backend: exact_heldout (m = 64)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.530 ± 0.365 | 0.651 ± 0.203 | 1.596 ± 0.374 | 1.245 ± 0.191 | 0/10 | 0.0549 ± 0.0066 | 0.0760 ± 0.0224 | 0.461 ± 0.259 |
| no_ema | 0.366 ± 0.350 | 0.503 ± 0.273 | 1.298 ± 0.510 | 1.249 ± 0.123 | 0/10 | 0.0617 ± 0.0093 | 0.0731 ± 0.0237 | 0.639 ± 0.273 |
| no_predictor | 0.053 ± 0.046 | 0.132 ± 0.123 | 0.686 ± 0.373 | 1.193 ± 0.100 | 0/10 | 1.8912 ± 5.7979 | 1.8912 ± 5.7979 | 0.000 ± 0.000 |
| untrained | 0.016 ± 0.019 | 0.004 ± 0.034 | 1.369 ± 0.184 | 1.215 ± 0.065 | 0/10 | 0.0871 ± 0.0095 | 0.0821 ± 0.0169 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.633 ± 0.387 | 0.0745 ± 0.0143 | 0.0815 ± 0.0055 |
| full | 2 | 0.522 ± 0.389 | 0.0432 ± 0.0108 | 0.0785 ± 0.0352 |
| full | 3 | 0.430 ± 0.422 | 0.0479 ± 0.0067 | 0.0674 ± 0.0350 |
| no_ema | 1 | 0.424 ± 0.418 | 0.0827 ± 0.0188 | 0.0769 ± 0.0096 |
| no_ema | 2 | 0.317 ± 0.388 | 0.0508 ± 0.0135 | 0.0707 ± 0.0341 |
| no_ema | 3 | 0.360 ± 0.367 | 0.0523 ± 0.0125 | 0.0719 ± 0.0382 |
| no_predictor | 1 | 0.105 ± 0.107 | 1.8531 ± 5.5940 | 1.8531 ± 5.5940 |
| no_predictor | 2 | 0.030 ± 0.041 | 1.9333 ± 5.9713 | 1.9333 ± 5.9713 |
| no_predictor | 3 | 0.025 ± 0.026 | 1.8828 ± 5.8128 | 1.8828 ± 5.8128 |
| untrained | 1 | 0.005 ± 0.015 | 0.1188 ± 0.0201 | 0.0688 ± 0.0092 |
| untrained | 2 | 0.030 ± 0.046 | 0.0681 ± 0.0115 | 0.0940 ± 0.0282 |
| untrained | 3 | 0.010 ± 0.021 | 0.0756 ± 0.0145 | 0.0824 ± 0.0239 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 64 | 0.0001 | 58/64 | 1.2e-97 | True | True |
| full_seed1_vs_untrained_seed1 | 64 | 0.0001 | 22/64 | 7.8e-24 | True | True |
| full_seed2_vs_untrained_seed2 | 64 | 0.0001 | 7/64 | 6.5e-05 | True | True |
| full_seed3_vs_untrained_seed3 | 64 | 0.0001 | 4/64 | 0.018 | True | True |
| full_seed4_vs_untrained_seed4 | 64 | 0.0001 | 31/64 | 1.1e-38 | True | True |
| full_seed5_vs_untrained_seed5 | 64 | 0.0001 | 34/64 | 4e-44 | True | True |
| full_seed6_vs_untrained_seed6 | 64 | 0.0001 | 50/64 | 1.9e-77 | True | True |
| full_seed7_vs_untrained_seed7 | 64 | 0.0001 | 62/64 | 2e-109 | True | True |
| full_seed8_vs_untrained_seed8 | 64 | 0.0001 | 8/64 | 7.2e-06 | True | True |
| full_seed9_vs_untrained_seed9 | 64 | 0.0001 | 63/64 | 1e-112 | True | True |

## Backend: aer (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.656 ± 0.250 | 0.655 ± 0.169 | 1.483 ± 0.346 | 1.246 ± 0.173 | 0/10 | 0.0460 ± 0.0156 | 0.0599 ± 0.0283 | 0.461 ± 0.259 |
| no_ema | 0.494 ± 0.330 | 0.493 ± 0.278 | 1.210 ± 0.499 | 1.253 ± 0.127 | 0/10 | 0.0443 ± 0.0093 | 0.0606 ± 0.0226 | 0.639 ± 0.273 |
| no_predictor | 0.138 ± 0.097 | 0.133 ± 0.125 | 0.651 ± 0.373 | 1.174 ± 0.108 | 0/10 | 1.8103 ± 5.4129 | 2.3112 ± 7.0495 | 0.000 ± 0.000 |
| untrained | 0.037 ± 0.044 | -0.016 ± 0.053 | 1.336 ± 0.211 | 1.194 ± 0.059 | 0/10 | 0.0627 ± 0.0121 | 0.0708 ± 0.0207 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.780 ± 0.257 | 0.0169 ± 0.0052 | 0.0251 ± 0.0083 |
| full | 2 | 0.600 ± 0.327 | 0.0314 ± 0.0170 | 0.0559 ± 0.0340 |
| full | 3 | 0.600 ± 0.306 | 0.0823 ± 0.0325 | 0.0922 ± 0.0433 |
| no_ema | 1 | 0.540 ± 0.341 | 0.0222 ± 0.0129 | 0.0266 ± 0.0084 |
| no_ema | 2 | 0.420 ± 0.382 | 0.0355 ± 0.0115 | 0.0557 ± 0.0281 |
| no_ema | 3 | 0.517 ± 0.372 | 0.0702 ± 0.0124 | 0.0929 ± 0.0348 |
| no_predictor | 1 | 0.260 ± 0.190 | 1.7995 ± 5.4370 | 2.3630 ± 7.2903 |
| no_predictor | 2 | 0.080 ± 0.140 | 1.8913 ± 5.6578 | 2.3990 ± 7.3452 |
| no_predictor | 3 | 0.083 ± 0.088 | 1.7517 ± 5.1888 | 2.1948 ± 6.6025 |
| untrained | 1 | 0.020 ± 0.063 | 0.0377 ± 0.0158 | 0.0263 ± 0.0087 |
| untrained | 2 | 0.020 ± 0.063 | 0.0519 ± 0.0166 | 0.0726 ± 0.0274 |
| untrained | 3 | 0.067 ± 0.086 | 0.0925 ± 0.0149 | 0.1065 ± 0.0302 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0006 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 6/16 | 0.00028 | True | True |
| full_seed3_vs_untrained_seed3 | 16 | 0.0011 | 6/16 | 0.00028 | True | True |
| full_seed4_vs_untrained_seed4 | 16 | 0.0001 | 9/16 | 1.1e-07 | True | True |
| full_seed5_vs_untrained_seed5 | 16 | 0.0001 | 12/16 | 5.1e-12 | True | True |
| full_seed6_vs_untrained_seed6 | 16 | 0.0001 | 12/16 | 5.1e-12 | True | True |
| full_seed7_vs_untrained_seed7 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed8_vs_untrained_seed8 | 16 | 0.0009 | 6/16 | 0.00028 | True | True |
| full_seed9_vs_untrained_seed9 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |

## Backend: aer_heldout (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.719 ± 0.291 | 0.705 ± 0.190 | 1.546 ± 0.433 | 1.277 ± 0.189 | 0/10 | 0.0649 ± 0.0169 | 0.0783 ± 0.0104 | 0.461 ± 0.259 |
| no_ema | 0.519 ± 0.344 | 0.549 ± 0.272 | 1.244 ± 0.534 | 1.230 ± 0.119 | 0/10 | 0.0752 ± 0.0179 | 0.0702 ± 0.0116 | 0.639 ± 0.273 |
| no_predictor | 0.113 ± 0.109 | 0.082 ± 0.151 | 0.682 ± 0.319 | 1.209 ± 0.102 | 0/10 | 1.9354 ± 5.7705 | 2.0059 ± 5.9535 | 0.000 ± 0.000 |
| untrained | 0.056 ± 0.046 | -0.006 ± 0.043 | 1.346 ± 0.185 | 1.196 ± 0.060 | 0/10 | 0.1017 ± 0.0132 | 0.0694 ± 0.0105 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.771 ± 0.279 | 0.0894 ± 0.0291 | 0.0919 ± 0.0162 |
| full | 2 | 0.700 ± 0.369 | 0.0538 ± 0.0219 | 0.0703 ± 0.0206 |
| full | 3 | 0.660 ± 0.353 | 0.0394 ± 0.0117 | 0.0658 ± 0.0383 |
| no_ema | 1 | 0.486 ± 0.376 | 0.0992 ± 0.0263 | 0.0832 ± 0.0135 |
| no_ema | 2 | 0.500 ± 0.391 | 0.0673 ± 0.0279 | 0.0554 ± 0.0080 |
| no_ema | 3 | 0.580 ± 0.394 | 0.0478 ± 0.0166 | 0.0637 ± 0.0401 |
| no_predictor | 1 | 0.114 ± 0.188 | 1.9826 ± 5.8481 | 2.3098 ± 6.8312 |
| no_predictor | 2 | 0.175 ± 0.206 | 1.9286 ± 5.7621 | 1.9670 ± 5.8796 |
| no_predictor | 3 | 0.060 ± 0.097 | 1.8748 ± 5.6686 | 1.6116 ± 4.7844 |
| untrained | 1 | 0.057 ± 0.074 | 0.1406 ± 0.0160 | 0.0736 ± 0.0143 |
| untrained | 2 | 0.100 ± 0.175 | 0.0787 ± 0.0224 | 0.0616 ± 0.0174 |
| untrained | 3 | 0.020 ± 0.063 | 0.0655 ± 0.0164 | 0.0698 ± 0.0236 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0001 | 11/16 | 1.8e-10 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0001 | 9/16 | 1.1e-07 | True | True |
| full_seed3_vs_untrained_seed3 | 16 | 0.0031 | 3/16 | 0.074 | True | False |
| full_seed4_vs_untrained_seed4 | 16 | 0.0022 | 7/16 | 2.6e-05 | True | True |
| full_seed5_vs_untrained_seed5 | 16 | 0.0001 | 13/16 | 1e-13 | True | True |
| full_seed6_vs_untrained_seed6 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed7_vs_untrained_seed7 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed8_vs_untrained_seed8 | 16 | 0.0001 | 8/16 | 1.9e-06 | True | True |
| full_seed9_vs_untrained_seed9 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |

## Backend: fake_fez (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.681 ± 0.249 | 0.656 ± 0.169 | 1.224 ± 0.276 | 1.156 ± 0.155 | 0/10 | 0.0424 ± 0.0109 | 0.0575 ± 0.0267 | 0.461 ± 0.259 |
| no_ema | 0.481 ± 0.331 | 0.493 ± 0.279 | 1.000 ± 0.405 | 1.168 ± 0.112 | 0/10 | 0.0423 ± 0.0082 | 0.0585 ± 0.0210 | 0.639 ± 0.273 |
| no_predictor | 0.169 ± 0.102 | 0.130 ± 0.126 | 0.530 ± 0.309 | 1.113 ± 0.102 | 0/10 | 1.0327 ± 2.9279 | 1.4140 ± 4.1597 | 0.000 ± 0.000 |
| untrained | 0.037 ± 0.044 | -0.016 ± 0.053 | 1.087 ± 0.169 | 1.107 ± 0.053 | 0/10 | 0.0592 ± 0.0112 | 0.0679 ± 0.0192 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.800 ± 0.249 | 0.0167 ± 0.0051 | 0.0237 ± 0.0078 |
| full | 2 | 0.640 ± 0.324 | 0.0307 ± 0.0176 | 0.0530 ± 0.0324 |
| full | 3 | 0.617 ± 0.305 | 0.0736 ± 0.0226 | 0.0894 ± 0.0411 |
| no_ema | 1 | 0.540 ± 0.341 | 0.0206 ± 0.0114 | 0.0253 ± 0.0077 |
| no_ema | 2 | 0.400 ± 0.400 | 0.0335 ± 0.0106 | 0.0525 ± 0.0260 |
| no_ema | 3 | 0.500 ± 0.377 | 0.0678 ± 0.0103 | 0.0911 ± 0.0326 |
| no_predictor | 1 | 0.320 ± 0.253 | 1.0457 ± 3.0717 | 1.5442 ± 4.6629 |
| no_predictor | 2 | 0.120 ± 0.140 | 1.0846 ± 3.0804 | 1.4105 ± 4.1756 |
| no_predictor | 3 | 0.083 ± 0.088 | 0.9787 ± 2.6811 | 1.3083 ± 3.7273 |
| untrained | 1 | 0.020 ± 0.063 | 0.0343 ± 0.0139 | 0.0245 ± 0.0077 |
| untrained | 2 | 0.020 ± 0.063 | 0.0480 ± 0.0148 | 0.0693 ± 0.0260 |
| untrained | 3 | 0.067 ± 0.086 | 0.0892 ± 0.0146 | 0.1030 ± 0.0281 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0007 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 7/16 | 2.6e-05 | True | True |
| full_seed3_vs_untrained_seed3 | 16 | 0.0007 | 6/16 | 0.00028 | True | True |
| full_seed4_vs_untrained_seed4 | 16 | 0.0001 | 10/16 | 5.1e-09 | True | True |
| full_seed5_vs_untrained_seed5 | 16 | 0.0001 | 12/16 | 5.1e-12 | True | True |
| full_seed6_vs_untrained_seed6 | 16 | 0.0001 | 14/16 | 1.5e-15 | True | True |
| full_seed7_vs_untrained_seed7 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed8_vs_untrained_seed8 | 16 | 0.0009 | 6/16 | 0.00028 | True | True |
| full_seed9_vs_untrained_seed9 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |

## Backend: fake_marrakesh (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.688 ± 0.238 | 0.656 ± 0.168 | 1.327 ± 0.303 | 1.191 ± 0.163 | 0/10 | 0.0428 ± 0.0109 | 0.0585 ± 0.0273 | 0.461 ± 0.259 |
| no_ema | 0.494 ± 0.330 | 0.492 ± 0.278 | 1.082 ± 0.445 | 1.200 ± 0.117 | 0/10 | 0.0431 ± 0.0086 | 0.0596 ± 0.0219 | 0.639 ± 0.273 |
| no_predictor | 0.150 ± 0.111 | 0.131 ± 0.125 | 0.575 ± 0.336 | 1.130 ± 0.103 | 0/10 | 1.0867 ± 3.1154 | 1.5543 ± 4.6255 | 0.000 ± 0.000 |
| untrained | 0.037 ± 0.044 | -0.015 ± 0.055 | 1.183 ± 0.185 | 1.141 ± 0.058 | 0/10 | 0.0609 ± 0.0115 | 0.0690 ± 0.0199 | 0.836 ± 0.086 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.820 ± 0.220 | 0.0166 ± 0.0051 | 0.0245 ± 0.0082 |
| full | 2 | 0.640 ± 0.337 | 0.0307 ± 0.0168 | 0.0535 ± 0.0335 |
| full | 3 | 0.617 ± 0.305 | 0.0746 ± 0.0239 | 0.0910 ± 0.0416 |
| no_ema | 1 | 0.540 ± 0.341 | 0.0213 ± 0.0119 | 0.0255 ± 0.0079 |
| no_ema | 2 | 0.420 ± 0.382 | 0.0351 ± 0.0118 | 0.0543 ± 0.0273 |
| no_ema | 3 | 0.517 ± 0.372 | 0.0681 ± 0.0113 | 0.0924 ± 0.0336 |
| no_predictor | 1 | 0.300 ± 0.236 | 1.0868 ± 3.2000 | 1.6932 ± 5.1572 |
| no_predictor | 2 | 0.080 ± 0.140 | 1.1632 ± 3.3330 | 1.5183 ± 4.5472 |
| no_predictor | 3 | 0.083 ± 0.088 | 1.0228 ± 2.8636 | 1.4685 ± 4.2479 |
| untrained | 1 | 0.020 ± 0.063 | 0.0361 ± 0.0150 | 0.0254 ± 0.0081 |
| untrained | 2 | 0.020 ± 0.063 | 0.0503 ± 0.0159 | 0.0704 ± 0.0260 |
| untrained | 3 | 0.067 ± 0.086 | 0.0904 ± 0.0147 | 0.1042 ± 0.0292 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0009 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 8/16 | 1.9e-06 | True | True |
| full_seed3_vs_untrained_seed3 | 16 | 0.001 | 6/16 | 0.00028 | True | True |
| full_seed4_vs_untrained_seed4 | 16 | 0.0001 | 9/16 | 1.1e-07 | True | True |
| full_seed5_vs_untrained_seed5 | 16 | 0.0001 | 12/16 | 5.1e-12 | True | True |
| full_seed6_vs_untrained_seed6 | 16 | 0.0001 | 14/16 | 1.5e-15 | True | True |
| full_seed7_vs_untrained_seed7 | 16 | 0.0001 | 16/16 | 5.4e-20 | True | True |
| full_seed8_vs_untrained_seed8 | 16 | 0.0009 | 7/16 | 2.6e-05 | True | True |
| full_seed9_vs_untrained_seed9 | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |

## Backend: qpu (m = 16, shots = 450–1000 (per run))

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.646 ± 0.282 | 0.589 ± 0.210 | 1.037 ± 0.245 | 1.075 ± 0.178 | 0/3 | 0.0388 ± 0.0086 | 0.0680 ± 0.0249 | 0.453 ± 0.330 |
| no_ema | 0.438 ± 0.000 | 0.455 ± 0.000 | 1.246 ± 0.000 | 1.160 ± 0.000 | 0/1 | 0.0411 ± 0.0000 | 0.0409 ± 0.0000 | 0.611 ± 0.000 |
| no_predictor | 0.125 ± 0.000 | 0.151 ± 0.000 | 0.395 ± 0.000 | 1.011 ± 0.000 | 0/1 | 0.4694 ± 0.0000 | 0.5438 ± 0.0000 | 0.000 ± 0.000 |
| untrained | 0.062 ± 0.000 | 0.006 ± 0.038 | 0.870 ± 0.109 | 1.031 ± 0.064 | 0/2 | 0.0596 ± 0.0043 | 0.0812 ± 0.0067 | 0.868 ± 0.120 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.867 ± 0.115 | 0.0179 ± 0.0104 | 0.0226 ± 0.0063 |
| full | 2 | 0.667 ± 0.416 | 0.0188 ± 0.0088 | 0.0721 ± 0.0372 |
| full | 3 | 0.444 ± 0.347 | 0.0727 ± 0.0145 | 0.1026 ± 0.0319 |
| no_ema | 1 | 0.200 ± 0.000 | 0.0289 ± 0.0000 | 0.0186 ± 0.0000 |
| no_ema | 2 | 0.400 ± 0.000 | 0.0303 ± 0.0000 | 0.0340 ± 0.0000 |
| no_ema | 3 | 0.667 ± 0.000 | 0.0602 ± 0.0000 | 0.0651 ± 0.0000 |
| no_predictor | 1 | 0.200 ± 0.000 | 0.2044 ± 0.0000 | 0.2226 ± 0.0000 |
| no_predictor | 2 | 0.000 ± 0.000 | 0.4681 ± 0.0000 | 0.6884 ± 0.0000 |
| no_predictor | 3 | 0.167 ± 0.000 | 0.6914 ± 0.0000 | 0.6910 ± 0.0000 |
| untrained | 1 | 0.000 ± 0.000 | 0.0356 ± 0.0026 | 0.0337 ± 0.0148 |
| untrained | 2 | 0.000 ± 0.000 | 0.0359 ± 0.0027 | 0.0740 ± 0.0155 |
| untrained | 3 | 0.167 ± 0.000 | 0.0993 ± 0.0113 | 0.1267 ± 0.0073 |

Tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). The criterion was pre-registered for the seed-0 pair on the QPU only; all other tests are exploratory. p = 0.0001 is the floor of 10,000 sign-flip resamples.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (pre-registered) | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0005 | 10/16 | 5.1e-09 | True | True |
| full_seed2_vs_untrained_seed0 | 16 | 0.0016 | 6/16 | 0.00028 | True | True |

