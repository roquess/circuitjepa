# QJEPA results summary

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

