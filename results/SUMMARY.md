# CircuitJEPA results summary

Mean ± sample std (ddof=1) over seeds. `collapsed` = number of seeds flagged collapsed
(diag gap < 0.05 and target total variance < 5 % of the same-seed untrained baseline).
Probe: linear map fitted on exact target embeddings of the training set, evaluated
on predicted (`pred`) vs un-advanced context (`unadv`) embeddings; target = ⟨Z_i⟩(t₀+s).

## Backend: exact (m = 64)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.469 ± 0.402 | 0.632 ± 0.210 | 1.562 ± 0.479 | 1.249 ± 0.232 | 0/3 | 0.0498 ± 0.0082 | 0.0830 ± 0.0289 | 0.453 ± 0.330 |
| no_ema | 0.391 ± 0.327 | 0.573 ± 0.219 | 1.527 ± 0.522 | 1.293 ± 0.201 | 0/3 | 0.0544 ± 0.0086 | 0.0838 ± 0.0372 | 0.523 ± 0.255 |
| no_predictor | 0.031 ± 0.054 | 0.175 ± 0.204 | 0.839 ± 0.138 | 1.123 ± 0.146 | 0/3 | 0.0474 ± 0.0012 | 0.0474 ± 0.0012 | 0.000 ± 0.000 |
| untrained | 0.000 ± 0.000 | -0.030 ± 0.051 | 1.279 ± 0.161 | 1.176 ± 0.074 | 0/3 | 0.0803 ± 0.0059 | 0.0918 ± 0.0118 | 0.894 ± 0.095 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.574 ± 0.370 | 0.0425 ± 0.0069 | 0.0614 ± 0.0111 |
| full | 2 | 0.493 ± 0.411 | 0.0470 ± 0.0075 | 0.0992 ± 0.0388 |
| full | 3 | 0.362 ± 0.479 | 0.0582 ± 0.0124 | 0.0838 ± 0.0332 |
| no_ema | 1 | 0.500 ± 0.434 | 0.0535 ± 0.0131 | 0.0592 ± 0.0148 |
| no_ema | 2 | 0.333 ± 0.395 | 0.0527 ± 0.0134 | 0.1020 ± 0.0508 |
| no_ema | 3 | 0.362 ± 0.239 | 0.0568 ± 0.0156 | 0.0847 ± 0.0413 |
| no_predictor | 1 | 0.056 ± 0.096 | 0.0466 ± 0.0007 | 0.0466 ± 0.0007 |
| no_predictor | 2 | 0.043 ± 0.075 | 0.0510 ± 0.0018 | 0.0510 ± 0.0018 |
| no_predictor | 3 | 0.000 ± 0.000 | 0.0443 ± 0.0036 | 0.0443 ± 0.0036 |
| untrained | 1 | 0.000 ± 0.000 | 0.0937 ± 0.0161 | 0.0606 ± 0.0075 |
| untrained | 2 | 0.000 ± 0.000 | 0.0707 ± 0.0077 | 0.1156 ± 0.0203 |
| untrained | 3 | 0.000 ± 0.000 | 0.0793 ± 0.0092 | 0.0923 ± 0.0084 |

Pre-registered tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). Seed 0 is the registered pair.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (registered) | 64 | 0.0001 | 59/64 | 1.9e-100 | True | True |
| full_seed1_vs_untrained_seed1 | 64 | 0.0001 | 21/64 | 2.5e-22 | True | True |
| full_seed2_vs_untrained_seed2 | 64 | 0.0001 | 10/64 | 6.1e-08 | True | True |

## Backend: aer (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.604 ± 0.295 | 0.611 ± 0.214 | 1.467 ± 0.453 | 1.269 ± 0.219 | 0/3 | 0.0419 ± 0.0116 | 0.0741 ± 0.0294 | 0.453 ± 0.330 |
| no_ema | 0.521 ± 0.321 | 0.578 ± 0.209 | 1.476 ± 0.501 | 1.289 ± 0.209 | 0/3 | 0.0465 ± 0.0037 | 0.0724 ± 0.0357 | 0.523 ± 0.255 |
| no_predictor | 0.125 ± 0.165 | 0.233 ± 0.188 | 0.704 ± 0.266 | 1.123 ± 0.205 | 0/3 | 0.1000 ± 0.0617 | 0.0778 ± 0.0380 | 0.000 ± 0.000 |
| untrained | 0.021 ± 0.036 | -0.019 ± 0.041 | 1.207 ± 0.137 | 1.158 ± 0.068 | 0/3 | 0.0695 ± 0.0085 | 0.0797 ± 0.0191 | 0.894 ± 0.095 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.800 ± 0.200 | 0.0175 ± 0.0103 | 0.0255 ± 0.0081 |
| full | 2 | 0.600 ± 0.346 | 0.0180 ± 0.0097 | 0.0820 ± 0.0450 |
| full | 3 | 0.444 ± 0.347 | 0.0823 ± 0.0237 | 0.1080 ± 0.0349 |
| no_ema | 1 | 0.533 ± 0.416 | 0.0266 ± 0.0153 | 0.0277 ± 0.0056 |
| no_ema | 2 | 0.467 ± 0.306 | 0.0362 ± 0.0073 | 0.0700 ± 0.0415 |
| no_ema | 3 | 0.556 ± 0.347 | 0.0717 ± 0.0119 | 0.1117 ± 0.0565 |
| no_predictor | 1 | 0.133 ± 0.231 | 0.0880 ± 0.0853 | 0.0590 ± 0.0567 |
| no_predictor | 2 | 0.200 ± 0.200 | 0.1164 ± 0.0816 | 0.0820 ± 0.0752 |
| no_predictor | 3 | 0.056 ± 0.096 | 0.0964 ± 0.0480 | 0.0901 ± 0.0280 |
| untrained | 1 | 0.000 ± 0.000 | 0.0482 ± 0.0126 | 0.0293 ± 0.0135 |
| untrained | 2 | 0.000 ± 0.000 | 0.0476 ± 0.0116 | 0.0785 ± 0.0293 |
| untrained | 3 | 0.056 ± 0.096 | 0.1055 ± 0.0146 | 0.1228 ± 0.0234 |

Pre-registered tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). Seed 0 is the registered pair.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (registered) | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0006 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 6/16 | 0.00028 | True | True |

## Backend: fake_fez (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.625 ± 0.272 | 0.612 ± 0.213 | 1.216 ± 0.362 | 1.178 ± 0.200 | 0/3 | 0.0399 ± 0.0110 | 0.0715 ± 0.0277 | 0.453 ± 0.330 |
| no_ema | 0.521 ± 0.321 | 0.580 ± 0.209 | 1.216 ± 0.398 | 1.201 ± 0.182 | 0/3 | 0.0435 ± 0.0039 | 0.0697 ± 0.0331 | 0.523 ± 0.255 |
| no_predictor | 0.125 ± 0.165 | 0.226 ± 0.189 | 0.575 ± 0.227 | 1.063 ± 0.198 | 0/3 | 0.0963 ± 0.0609 | 0.0693 ± 0.0376 | 0.000 ± 0.000 |
| untrained | 0.021 ± 0.036 | -0.017 ± 0.031 | 0.982 ± 0.122 | 1.073 ± 0.062 | 0/3 | 0.0656 ± 0.0079 | 0.0758 ± 0.0165 | 0.894 ± 0.095 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.867 ± 0.115 | 0.0170 ± 0.0101 | 0.0242 ± 0.0073 |
| full | 2 | 0.600 ± 0.346 | 0.0175 ± 0.0092 | 0.0778 ± 0.0427 |
| full | 3 | 0.444 ± 0.347 | 0.0777 ± 0.0212 | 0.1057 ± 0.0335 |
| no_ema | 1 | 0.533 ± 0.416 | 0.0241 ± 0.0134 | 0.0266 ± 0.0059 |
| no_ema | 2 | 0.467 ± 0.306 | 0.0335 ± 0.0084 | 0.0661 ± 0.0373 |
| no_ema | 3 | 0.556 ± 0.347 | 0.0681 ± 0.0084 | 0.1085 ± 0.0527 |
| no_predictor | 1 | 0.133 ± 0.231 | 0.0623 ± 0.0650 | 0.0507 ± 0.0475 |
| no_predictor | 2 | 0.200 ± 0.200 | 0.1155 ± 0.0788 | 0.0626 ± 0.0323 |
| no_predictor | 3 | 0.056 ± 0.096 | 0.1086 ± 0.0621 | 0.0905 ± 0.0372 |
| untrained | 1 | 0.000 ± 0.000 | 0.0445 ± 0.0125 | 0.0271 ± 0.0115 |
| untrained | 2 | 0.000 ± 0.000 | 0.0447 ± 0.0096 | 0.0742 ± 0.0256 |
| untrained | 3 | 0.056 ± 0.096 | 0.1005 ± 0.0132 | 0.1177 ± 0.0209 |

Pre-registered tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). Seed 0 is the registered pair.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (registered) | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0007 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 7/16 | 2.6e-05 | True | True |

## Backend: fake_marrakesh (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.646 ± 0.253 | 0.614 ± 0.212 | 1.314 ± 0.398 | 1.214 ± 0.208 | 0/3 | 0.0406 ± 0.0114 | 0.0723 ± 0.0281 | 0.453 ± 0.330 |
| no_ema | 0.521 ± 0.321 | 0.579 ± 0.209 | 1.320 ± 0.437 | 1.236 ± 0.191 | 0/3 | 0.0446 ± 0.0037 | 0.0707 ± 0.0343 | 0.523 ± 0.255 |
| no_predictor | 0.125 ± 0.165 | 0.229 ± 0.189 | 0.624 ± 0.247 | 1.080 ± 0.195 | 0/3 | 0.0977 ± 0.0619 | 0.0722 ± 0.0339 | 0.000 ± 0.000 |
| untrained | 0.021 ± 0.036 | -0.020 ± 0.040 | 1.074 ± 0.125 | 1.104 ± 0.070 | 0/3 | 0.0671 ± 0.0078 | 0.0766 ± 0.0177 | 0.894 ± 0.095 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 0.867 ± 0.115 | 0.0171 ± 0.0104 | 0.0248 ± 0.0076 |
| full | 2 | 0.667 ± 0.306 | 0.0176 ± 0.0094 | 0.0794 ± 0.0436 |
| full | 3 | 0.444 ± 0.347 | 0.0794 ± 0.0226 | 0.1058 ± 0.0333 |
| no_ema | 1 | 0.533 ± 0.416 | 0.0250 ± 0.0141 | 0.0268 ± 0.0052 |
| no_ema | 2 | 0.467 ± 0.306 | 0.0346 ± 0.0079 | 0.0679 ± 0.0401 |
| no_ema | 3 | 0.556 ± 0.347 | 0.0691 ± 0.0097 | 0.1096 ± 0.0542 |
| no_predictor | 1 | 0.133 ± 0.231 | 0.0673 ± 0.0664 | 0.0543 ± 0.0464 |
| no_predictor | 2 | 0.200 ± 0.200 | 0.1149 ± 0.0782 | 0.0642 ± 0.0452 |
| no_predictor | 3 | 0.056 ± 0.096 | 0.1089 ± 0.0635 | 0.0938 ± 0.0381 |
| untrained | 1 | 0.000 ± 0.000 | 0.0457 ± 0.0131 | 0.0279 ± 0.0117 |
| untrained | 2 | 0.000 ± 0.000 | 0.0458 ± 0.0097 | 0.0748 ± 0.0273 |
| untrained | 3 | 0.056 ± 0.096 | 0.1026 ± 0.0133 | 0.1186 ± 0.0221 |

Pre-registered tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). Seed 0 is the registered pair.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (registered) | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |
| full_seed1_vs_untrained_seed1 | 16 | 0.0009 | 8/16 | 1.9e-06 | True | True |
| full_seed2_vs_untrained_seed2 | 16 | 0.0004 | 8/16 | 1.9e-06 | True | True |

## Backend: qpu (m = 16, shots = 1000)

| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |
|---|---|---|---|---|---|---|---|---|
| full | 0.938 ± 0.000 | 0.827 ± 0.000 | 1.218 ± 0.000 | 1.241 ± 0.000 | 0/1 | 0.0316 ± 0.0000 | 0.0753 ± 0.0000 | 0.273 ± 0.000 |
| no_ema | 0.438 ± 0.000 | 0.455 ± 0.000 | 1.246 ± 0.000 | 1.160 ± 0.000 | 0/1 | 0.0411 ± 0.0000 | 0.0409 ± 0.0000 | 0.611 ± 0.000 |
| untrained | 0.062 ± 0.000 | 0.033 ± 0.000 | 0.947 ± 0.000 | 1.076 ± 0.000 | 0/1 | 0.0626 ± 0.0000 | 0.0860 ± 0.0000 | 0.783 ± 0.000 |

Per time step s:

| variant | s | top-1 | probe pred | probe unadv |
|---|---|---|---|---|
| full | 1 | 1.000 ± 0.000 | 0.0139 ± 0.0000 | 0.0244 ± 0.0000 |
| full | 2 | 1.000 ± 0.000 | 0.0108 ± 0.0000 | 0.0728 ± 0.0000 |
| full | 3 | 0.833 ± 0.000 | 0.0637 ± 0.0000 | 0.1197 ± 0.0000 |
| no_ema | 1 | 0.200 ± 0.000 | 0.0289 ± 0.0000 | 0.0186 ± 0.0000 |
| no_ema | 2 | 0.400 ± 0.000 | 0.0303 ± 0.0000 | 0.0340 ± 0.0000 |
| no_ema | 3 | 0.667 ± 0.000 | 0.0602 ± 0.0000 | 0.0651 ± 0.0000 |
| untrained | 1 | 0.000 ± 0.000 | 0.0337 ± 0.0000 | 0.0442 ± 0.0000 |
| untrained | 2 | 0.000 ± 0.000 | 0.0378 ± 0.0000 | 0.0850 ± 0.0000 |
| untrained | 3 | 0.167 ± 0.000 | 0.1073 ± 0.0000 | 0.1215 ± 0.0000 |

Pre-registered tests (full vs untrained; gap: paired sign-flip, p < 0.01; top-1: binomial vs chance 1/m, p < 0.05). Seed 0 is the registered pair.

| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |
|---|---|---|---|---|---|---|
| full_seed0_vs_untrained_seed0 (registered) | 16 | 0.0001 | 15/16 | 1.3e-17 | True | True |

