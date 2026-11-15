---
title: "CircuitJEPA: a fully circuit-based Joint-Embedding Predictive Architecture for quantum dynamics"
author: Steve Roques
date: November 2026
abstract: |
  CircuitJEPA is a Joint-Embedding Predictive Architecture (JEPA) in which the encoder, the exponential-moving-average (EMA) target encoder and the action-conditioned predictor are all parametrized quantum circuits. The data are states of a 3-qubit transverse-field Ising chain evolved by Trotter steps. From a context state, the model learns, without reconstruction or negative pairs, to predict the latent embedding of the state reached $s \in \{1, 2, 3\}$ steps later. Embeddings are the 15 two-qubit Pauli expectation values of a 2-qubit latent register, compared with a cosine similarity. I train in exact simulation over three seeds and evaluate the trained circuits on noiseless and noisy simulators and on the IBM Heron QPU `ibm_marrakesh`. In exact simulation the full model reaches a held-out retrieval top-1 accuracy of 0.469 ± 0.402 (chance 1/64), with large variance across seeds (0.922, 0.328, 0.156). On the QPU, the pre-registered seed-0 model, which is also the best seed, retrieves 15 of 16 targets and passes the pre-registered criterion against an untrained baseline (sign-flip p = 1.0 × 10^-4^, binomial p = 1.3 × 10^-17^). Negative and inconclusive findings are reported with the same weight: removing the EMA and the stop-gradient did not cause collapse, a linear dynamics probe does not separate the full model from the no-predictor ablation, and the setting was chosen on test metrics. This is a feasibility study on 5 qubits, with no claim of quantum advantage or scalability.
bibliography: references.bib
link-citations: true
numbersections: true
geometry: margin=2.5cm
fontsize: 11pt
mainfont: TeX Gyre Pagella
mathfont: TeX Gyre Pagella Math
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{htbp}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}}
---

# Introduction

A Joint-Embedding Predictive Architecture (JEPA) learns representations by predicting the *embedding* of a target from the embedding of a context [@lecun2022path]. It never reconstructs the input and needs no negative pairs. Collapse is avoided by asymmetry: a predictor on the context branch and a target encoder that is an exponential moving average (EMA) of the context encoder and receives no gradient [@grill2020byol; @assran2023ijepa]. With an action-conditioned predictor, a JEPA becomes a latent world model [@lecun2022path; @assran2025vjepa2].

This paper asks a narrow feasibility question. Can this principle be implemented with parametrized quantum circuits throughout, on data that are themselves quantum states? And do the trained circuits keep their behaviour on a superconducting quantum processor? With quantum-native data, the encoder acts on the state produced by the physical process, with no classical data loading.

The contribution is an implementation and a measured evaluation, not a performance result: (i) a JEPA whose encoder, EMA target encoder and action-conditioned predictor are all circuits, trained on spin-chain dynamics; (ii) ablations without EMA and stop-gradient, without predictor, and an untrained baseline, with collapse diagnostics and a dynamics probe; (iii) an evaluation of the same parameters on noiseless, noisy-simulated and real hardware, with a criterion fixed beforehand. All results are reported as measured, including those that did not go as expected.

# Related work

**JEPA and non-contrastive learning.** I-JEPA predicts EMA-encoder embeddings of masked image blocks with an L2 loss [@assran2023ijepa]; V-JEPA applies feature prediction to video [@bardes2024vjepa]; V-JEPA 2 adds an action-conditioned predictor used for planning [@assran2025vjepa2], the closest classical analogue of the predictor used here. Contrastive methods such as SimCLR need negative pairs [@chen2020simclr]. BYOL showed that a predictor plus an EMA target suffices without negatives, with a normalised (cosine) loss [@grill2020byol]; SimSiam showed that a stop-gradient alone can suffice, and that removing it leads to collapse [@chen2021simsiam]; VICReg uses explicit variance–covariance terms instead [@bardes2022vicreg]. I adopt the BYOL recipe (predictor, EMA target, cosine loss) because it needs neither negatives nor extra regularisers, which matters when every embedding costs measurements.

**Quantum machine learning.** CircuitJEPA is a variational quantum model [@cerezo2021variational; @biamonte2017quantum]. Lloyd et al. train quantum embeddings to separate classes, a supervised form of metric learning [@lloyd2020embeddings]. Jaderberg et al. introduce quantum self-supervised learning with small quantum networks inside a hybrid pipeline on classical images [@jaderberg2021quantum], and Edwards et al. insert a quantum module into SimCLR, MoCo v2 and BYOL backbones [@edwards2026width]. These works use classical data or classical backbones. Random deep circuits have barren plateaus [@mcclean2018barren], local costs on shallow circuits avoid them [@cerezo2021cost], and provable absence of barren plateaus often comes with classical simulability [@cerezo2025simulability]. The model here is small enough to be classically simulable, one reason why no advantage is claimed.

**Closest JEPA-like work.** Three recent works combine JEPA ideas with quantum ingredients. HQ-JEPA [@hossain2026hqjepa] is a classical JEPA for remote-sensing imagery with an added SWAP-test-based fidelity term; its encoders and predictor are classical. Quantum-JEPA [@liu2026quantumjepa] uses a quantum circuit as a Born-rule prior generating binary latent actions for an otherwise classical latent-action JEPA world model. QSWM [@jiang2026qswm] is quantum-*inspired*: a classical world model with density-matrix-like latents, evaluated on cellular automata. In CircuitJEPA the encoder, the EMA target encoder and the action-conditioned predictor are all parametrized circuits, the inputs are quantum states, and the embedding is a vector of Pauli expectations measured on a QPU. My search covered arXiv titles and abstracts and may miss very recent work; I did not find a prior architecture with these three components as circuits on quantum-native data, executed on hardware. That is the full extent of the novelty claim.

**Data and readout.** The data follow the standard Trotterized digital simulation [@lloyd1996universal]; circuits are built and run with Qiskit [@javadiabhari2024qiskit]. Embeddings are read out by Pauli-basis tomography; classical shadows [@huang2020shadows] are the scalable alternative.

# Method

**Data.** The data qubits form an open chain of $n = 3$ spins with $H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i$, $J = h = 1$, evolved by first-order Trotter steps $U_\delta = \prod_i e^{\,i h \delta X_i} \prod_{i} e^{\,i J \delta Z_i Z_{i+1}}$ ($R_{ZZ}$ and $R_X$ gates). Each sample starts from a product state with qubit $i$ in $R_Z(b_i) R_Y(a_i) \lvert 0 \rangle$, $a_i \sim \mathcal{U}[0, \pi]$, $b_i \sim \mathcal{U}[0, 2\pi]$. A sample is $(\psi_0, t_0, s)$ with $t_0 \in \{0, \dots, 3\}$ and $s \in \{1, 2, 3\}$ uniform; the context is $U_\delta^{t_0} \psi_0$ and the target $U_\delta^{t_0 + s} \psi_0$. The same circuits define the data in simulation and on hardware. I use 256 training samples (data seed 1234) and 64 test samples (data seed 5678), of which 18, 23 and 23 samples have $s = 1, 2, 3$.

**Encoder.** $E_\theta$ acts on the 3 data qubits plus 2 ancillas in $\lvert 0 \rangle$: 3 layers of $R_Y$, $R_Z$ on every qubit followed by a CZ chain, for 30 parameters in total. The latent is the reduced state of qubits 0 and 1; the other 3 qubits are traced out. The ancillas are there so that collapse is not excluded: with a single discarded qubit, a constant latent would be unreachable by a dimension argument, whereas with 3 discarded qubits that argument no longer excludes it. I did not check whether the ansatz can actually reach one.

**Predictor and target encoder.** $P_\phi$ is a 2-layer ansatz on the 2 latent qubits ($R_Y$, $R_Z$, then a CZ per layer; 8 parameters), applied $s$ times for time step $s$: the action is the number of steps. Its distance from the identity is $\lVert P - I \rVert := 1 - \lvert \mathrm{Tr}\, U_\phi \rvert / 4$. The target encoder $E_{\bar\theta}$ has the architecture of $E_\theta$, with $\bar\theta \leftarrow \tau \bar\theta + (1 - \tau)\theta$ after every step ($\tau = 0.99$) and no gradient.

**Embedding and loss.** The embedding of a latent state $\rho$ is its Bloch vector $r \in \mathbb{R}^{15}$, $r_P = \mathrm{Tr}(\rho P)$ over the non-identity two-qubit Paulis. The loss is $\mathcal{L} = 1 - \overline{\cos}(r_{\mathrm{pred}}, s_{\mathrm{target}})$ over the batch, with norms computed as $\sqrt{\lVert r \rVert^2 + \varepsilon^2}$, $\varepsilon = 10^{-8}$. I avoid the raw overlap $\mathrm{Tr}(\rho\sigma) = (1 + r \cdot s)/4$ because it depends on purity: it favours pure targets, and hardware noise would flatten it like collapse. The cosine is invariant to rescaling of either vector, in particular to global depolarisation, and is built from local observables rather than a global fidelity [@cerezo2021cost].

**Training and variants.** Training uses an exact statevector simulation in PyTorch (complex128) with automatic differentiation, Adam, and parameters initialised uniformly in $[0, 2\pi)$. Four variants are trained with seeds 0, 1, 2: **full**; **no_ema**, whose target branch shares $\theta$ and receives gradients, so both the EMA and the stop-gradient are removed; **no_predictor** ($P_\phi = I$); and **untrained** (random parameters).

# Experimental setup

**Configuration and tuning.** The final configuration is $\delta = 0.5$, $t_0 \le 3$, 3 encoder layers, 2 predictor layers, learning rate 0.02 (Adam), 150 epochs, $\tau = 0.99$, batch size 32, 256 training and 64 test samples. Only $\delta$ differs from the defaults. It was chosen with a sanity gate fixed beforehand: mean `full` exact test top-1 at least 0.3 and no collapsed `full` seed, at most 4 attempts, same settings for all variants. Attempt 1 used the defaults ($\delta = 0.25$), trained all four variants and failed (`full` top-1 0.245 ± 0.157); `no_predictor` beat `full` there. Knowing this, I set attempts 2–4 together and ran them as one batch: $\tau = 0.996$ and 200 epochs (0.271 ± 0.212, fail), $\delta = 0.5$ (0.469 ± 0.402, pass), and learning rate 0.05 with batch size 16 (0.214 ± 0.140, fail). The rule was to keep a passing setting; only attempt 3 passed. The gate is defined on the test split (there is no validation split), so the `full` numbers carry a selection bias over four settings.

**Metrics.** For $m$ test samples, $G_{ij} = \cos(r_{\mathrm{pred}, i}, s_{\mathrm{target}, j})$. *Top-1* is the fraction of rows with $\arg\max_j G_{ij} = i$ (chance $1/m$). The *diagonal gap* is the row mean of $g_i = G_{ii} - \mathrm{mean}_{j \ne i} G_{ij}$. A run is flagged *collapsed* when its gap is below 0.05 and its target total variance is below 5% of the same-seed untrained baseline. The *dynamics probe* is a ridge-regularised linear map (ridge $10^{-3}$) fitted on exact target embeddings of the training set to the true magnetisations $\langle Z_i \rangle(t_0+s)$. Its test MSE is reported for the *predicted* embedding and for the *un-advanced* context embedding ($E_\theta$ without predictor).

**Backends and readout.** The same parameters are evaluated on: **exact** (statevector, $m = 64$); **aer** (ideal Aer, 1000 shots, basis rz/sx/x/cz without connectivity constraints); **fake_fez** and **fake_marrakesh** (Aer with the device noise models of `qiskit-ibm-runtime` 0.52.1, 1000 shots); and **qpu** (`ibm_marrakesh` via the Sampler primitive with default options: no error mitigation, fractional gates not enabled). Shot-based backends use the first 16 test samples. For each sample the predicted, target and un-advanced circuits are measured in the 9 bases $\{X, Y, Z\}^{\otimes 2}$ of the latent qubits, and $G$ is computed classically. Circuits are transpiled at optimisation level 3 with a fixed seed. Software: Python 3.12.10, Qiskit 2.6.2, Qiskit Aer 0.18.2, qiskit-ibm-runtime 0.52.1, PyTorch 2.15.0.

**Pre-registered QPU criterion.** Before any training result was committed, I fixed the criterion in code (commit `41d5c8b`; an internal pre-registration, not a public registry), with the seed-0 pair (`full` vs `untrained`) as the registered pair. The full model's per-row gaps must exceed the baseline's under a paired one-sided sign-flip test with 10,000 resamples at $p < 0.01$; the smallest attainable value is $1/10{,}001 \approx 1.0 \times 10^{-4}$. Its top-1 must exceed chance under a one-sided binomial test at $p < 0.05$.

# Results

## Simulation

| variant | top-1 | diag. gap | total var. | mean $\lVert r\rVert$ | probe pred | probe unadv | $\lVert P-I\rVert$ |
|---|---|---|---|---|---|---|---|
| full | 0.469 ± 0.402 | 0.632 ± 0.210 | 1.562 ± 0.479 | 1.249 ± 0.232 | 0.0498 ± 0.0082 | 0.0830 ± 0.0289 | 0.453 ± 0.330 |
| no_ema | 0.391 ± 0.327 | 0.573 ± 0.219 | 1.527 ± 0.522 | 1.293 ± 0.201 | 0.0544 ± 0.0086 | 0.0838 ± 0.0372 | 0.523 ± 0.255 |
| no_predictor | 0.031 ± 0.054 | 0.175 ± 0.204 | 0.839 ± 0.138 | 1.123 ± 0.146 | 0.0474 ± 0.0012 | 0.0474 ± 0.0012 | 0.000 ± 0.000 |
| untrained | 0.000 ± 0.000 | -0.030 ± 0.051 | 1.279 ± 0.161 | 1.176 ± 0.074 | 0.0803 ± 0.0059 | 0.0918 ± 0.0118 | 0.894 ± 0.095 |

: Exact evaluation, $m = 64$, mean ± sample std over seeds 0, 1, 2 (chance top-1 1/64). No run was flagged as collapsed.

![Ablations by backend (mean ± std over seeds; one seed and no `no_predictor` run on the QPU). Dashed lines: chance $1/m$.](figures/ablation.png){width=100%}

**Seed variance is large.** The `full` exact top-1 is 59/64 (0.922) for seed 0, 21/64 (0.328) for seed 1 and 10/64 (0.156) for seed 2, so one seed dominates the mean; the diagonal gaps are 0.872, 0.536 and 0.486 and the final training losses 0.117, 0.442 and 0.466 (Figure 2). Each seed passes the same tests against its untrained baseline (sign-flip p = 1.0 × 10^-4^ for each; binomial p = 1.9 × 10^-100^, 2.5 × 10^-22^ and 6.1 × 10^-8^). `no_ema` is as dispersed: 20/64, 48/64 and 7/64.

![Training loss (mean ± std over seeds). A similar loss does not imply similar retrieval.](figures/loss_curves.png){width=55%}

**The predictor is needed for retrieval.** Without it, top-1 retrieval is 0.031 ± 0.054 (chance 1/64 ≈ 0.016) and the diagonal gap is 0.175 ± 0.204, although the training loss ends in the same range as `full`. At $\delta = 0.25$ (attempt 1) the opposite held: `no_predictor` reached an exact top-1 of 0.500 ± 0.204, against 0.245 ± 0.157 for `full`, plausibly (untested) because a small step makes "target ≈ context" a competitive shortcut.

**Removing the EMA and stop-gradient did not cause collapse.** `no_ema` reaches 0.391 ± 0.327 top-1 and 0.573 ± 0.219 diagonal gap, and its target variance (1.527 ± 0.522) does not shrink relative to `full` (1.562 ± 0.479). The collapse SimSiam reports without stop-gradient [@chen2021simsiam] did not occur here.

**Borderline cases.** `no_predictor` seed 2 has an exact diagonal gap of 0.0457, below 0.05; it is not flagged only because of the variance criterion (its target total variance is 0.720, against 1.158 for the untrained baseline). Its retrieval is 0/64, so I count it as near-collapsed, not healthy. `no_predictor` seed 0 is also non-retrieving (0/64) with a gap of 0.068, and its similarity matrix is uniformly high (Figure 4). Mean norms stay between 0.95 and 1.48 for all exact runs.

## Dynamics probe

![Dynamics probe by time step: predicted (solid) vs un-advanced (dashed) embedding. For `no_predictor` the two coincide.](figures/dynamics_probe.png){width=100%}

For `full` in exact simulation, the probe MSE is lower from the predicted embedding than from the un-advanced one (0.0498 ± 0.0082 against 0.0830 ± 0.0289), at every $s$ (0.0425, 0.0470, 0.0582 against 0.0614, 0.0992, 0.0838; Figure 3), for seeds 0 and 1 (0.0414 against 0.0936, and 0.0579 against 0.1052) but not seed 2 (0.0500 against 0.0504). However, **the probe does not separate `full` from `no_predictor`**: the latter, with near-chance retrieval, has a probe MSE of 0.0474 ± 0.0012, lower than `full`. The probe is therefore at most weak evidence that the predictor advances the embedding in time. $\lVert P - I \rVert$ of 0.453 ± 0.330 shows that the learned predictor is not the identity.

On shot-based backends, the `no_predictor` predicted and un-advanced circuits are identical, so their probe-MSE differences are pure sampling noise; they reach 0.0519 on aer, 0.0905 on fake_fez and 0.0742 on fake_marrakesh. Differences of `full` are of that size or smaller (for example 0.0475 on aer for seed 0), so I do not treat probe differences on shot-based backends, the QPU included, as significant.

## Noisy simulation

| backend | top-1 | diag. gap | total var. | mean $\lVert r\rVert$ |
|---|---|---|---|---|
| aer (noiseless, 1000 shots) | 0.604 ± 0.295 | 0.611 ± 0.214 | 1.467 ± 0.453 | 1.269 ± 0.219 |
| fake_fez | 0.625 ± 0.272 | 0.612 ± 0.213 | 1.216 ± 0.362 | 1.178 ± 0.200 |
| fake_marrakesh | 0.646 ± 0.253 | 0.614 ± 0.212 | 1.314 ± 0.398 | 1.214 ± 0.208 |

: `full` on the shot-based simulators ($m = 16$, chance 1/16), mean ± std over seeds.

Device noise changes top-1 and gap by less than the seed spread (for seed 0 the top-1 is 15/16 on all three), while it shrinks the total variance and mean norm. The top-1 of the other variants is also unchanged: `no_ema` gives 0.521 ± 0.321 on all three, and `untrained` gives 0.021 ± 0.036. A likely but undemonstrated explanation: the circuits are short (on average 19–27 CZ gates) and the cosine is purity-invariant, so a uniform contraction of the Bloch vectors changes $\lVert r \rVert$ but not $G$.

## QPU evaluation

**Runs.** The hardware runs took place on 2026-11-15 on `ibm_marrakesh` (IBM Heron) with $m = 16$, 9 bases and 1000 shots, which gives 432 circuits per model. The pre-registered pair was one submission (`ibm_marrakesh_20261115T100708+0000_6cc5cc3c`, 10:07 UTC): `full` seed 0 as job `dbso8j9abdes73frdplg`, `untrained` seed 0 as job `dbso8jujjaos73aked3g`. `no_ema` seed 0 was a second submission after the pair had completed (`ibm_marrakesh_20261115T101632+0000_a8e12a16`, 10:16 UTC, job `dbsod0gk09pc73b0ebpg`) and is not part of the criterion; `scripts/merge_qpu_eval.py` merges both submissions' raw counts, checking each run's parameter hash. Each job was charged 118 s of QPU time.

**Circuits and calibration.** For `full`, the transpiled circuits have a mean depth of 62.6 (maximum 84) and a mean of 21.4 two-qubit gates (maximum 27) for the predicted branch; for the target branch the mean depth is 77.8 (maximum 109) with a mean of 27.0 two-qubit gates (maximum 36); for the un-advanced branch, 48.2 (maximum 68) with a mean of 18.75 two-qubit gates (maximum 24). Only the first circuit's layout was recorded: [55, 54, 53, 39, 33] (latent qubits 55 and 54), the same for all models. Layouts are chosen per circuit; re-transpiling the 432 `full` seed-0 circuits against `FakeMarrakesh` with the same settings, 404 used one layout and 28 another, and CZ was the only two-qubit gate. I saved a calibration snapshot at about 10:13 UTC (after the pair, before `no_ema`). Its `last_update_date` is 2026-11-13 18:22:56+01:00, about 41 hours before the first job, and the quoted values carry dates between 54 and 57 hours before the first job, so they are context only. On the recorded layout it gives a median T1 of 176.4 µs, a median T2 of 184.1 µs, a median readout error of 0.0132 (0.0015 and 0.0039 on qubits 55 and 54), and the median CZ error on the four couplers of this chain is 0.00176.

| model (seed 0) | top-1 | diag. gap | total var. | mean $\lVert r\rVert$ |
|---|---|---|---|---|
| full, aer | 0.938 | 0.847 | 1.823 | 1.441 |
| full, fake_marrakesh | 0.938 | 0.847 | 1.634 | 1.378 |
| full, ibm_marrakesh | 0.938 | 0.827 | 1.218 | 1.241 |
| no_ema, ibm_marrakesh | 0.438 | 0.455 | 1.246 | 1.160 |
| untrained, ibm_marrakesh | 0.062 | 0.033 | 0.947 | 1.076 |

: Seed-0 models on the QPU, with `full` on two simulators for comparison ($m = 16$, chance 1/16).

**Pre-registered criterion: met.** The `full` seed-0 model retrieves 15 of 16 targets on the QPU, as on the simulators, against 1 of 16 for the untrained baseline. The diagonal-gap test gives p = 1.0 × 10^-4^ (the floor, below 0.01) and the binomial top-1 test gives p = 1.3 × 10^-17^ (below 0.05). By time step, top-1 is 1.000, 1.000 and 0.833 (5, 5 and 6 samples). `no_ema` gives 0.438 top-1 (7/16) (also 7/16 on all three shot-based simulators). Seed 0 is the best of the three seeds, so these numbers are not representative of the seed average (Table 2). Relative to aer, the QPU leaves top-1 unchanged and lowers the gap from 0.847 to 0.827, the total variance from 1.823 to 1.218 and the mean norm from 1.441 to 1.241, as with simulated noise.

![Similarity matrices $G$ of the seed-0 models on every backend; the diagonal holds the matching pairs.](figures/similarity_matrices.png){width=62%}

# Limitations

* **Scope.** 3 data qubits, 2 ancillas, a 2-qubit latent: classically simulable and trained entirely in simulation. Training with hardware gradients and shot noise was not tested. No advantage or scalability claim, and no comparison with a classical model.
* **Statistics.** Three seeds (`full` top-1 ranges from 0.156 to 0.922); 16 samples on hardware (5 or 6 per $s$).
* **Selection.** The gate passed only at attempt 3 of 4, is defined on test data, and $\delta$ was changed after attempt 1 had shown `no_predictor` beating `full` at $\delta = 0.25$. Conclusions are specific to this step size and carry a selection bias.
* **Best seed on hardware.** The QPU result is for the best seed, designated in advance; the `no_ema` QPU run was added afterwards.
* **Collapse.** Removing EMA and stop-gradient did not cause collapse, so this setup does not show what prevents it. Two `no_predictor` seeds are borderline (above).
* **Dynamics.** The probe does not separate `full` from `no_predictor`, and on shot-based backends its differences are within sampling noise. Retrieval may partly rely on conserved quantities, such as the X-parity $\prod_i X_i$, which $U_\delta$ conserves exactly, or, approximately, the energy.
* **Noise.** The small effect of noise on top-1 and gap is measured (one seed on the QPU); its explanation is a hypothesis. Calibration values are context only and refer to the first circuit's layout. No error mitigation was used.
* **Tomography cost.** A $k$-qubit latent needs $3^k$ bases and $4^k - 1$ expectation values, exponential in the latent size.

# Next steps

Planned work, with no predicted outcome: a short SPSA fine-tuning of the simulation-trained parameters on IBM hardware within the free-tier budget, and replacing tomography with classical shadows [@huang2020shadows].

# References {-}

::: {#refs}
:::
