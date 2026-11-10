# Pipeline Leak Detection: Synthetic Hydraulic Leak Localization and Optimization Dynamics

A research prototype for detecting and localizing pipeline leaks from **synthetic hydraulic simulations**, together with a set of optimization-dynamics experiments (capacity/noise sweeps, gradient propagation, and implicit regularization of gradient descent).

> **Read this first.**
>
> - All data are simulated (EPANET via WNTR) on a **single** topology (KY9). There is no field data and no cross-topology validation.
> - **Stage 1 accuracies are compromised by a row-wise train/test split** that places hourly rows of the same scenario in both sets (see [Row-wise split problem](#row-wise-split-problem-stage-1)). They are not independent-scenario results.
> - **Phase 2 excludes all zero-leak scenarios** because their grouping could not be reconstructed reliably (see [Phase 2 data](#phase-2-pooled-scenario-level-data)). Phase 2 therefore evaluates localization _conditional on a leak being present_, not leak/no-leak detection.
> - Scenario-held-out localization quality in Phase 2 is **weak** (best saved clean micro-average precision ≈ 0.0066, recall@3 ≈ 0.03).
> - The project does **not** contain an Adam-specific modified-loss derivation. Adam is compared with gradient flow only empirically.

---

## Table of contents

1. [Project progression and status](#1-project-progression-and-status)
2. [Hydraulic leak model](#2-hydraulic-leak-model)
3. [Datasets](#3-datasets)
4. [Models and objectives](#4-models-and-objectives)
5. [Stage 0: toy binary classifier](#5-stage-0-toy-binary-classifier)
6. [Stage 1: KY9 multivariate leak regression](#6-stage-1-ky9-multivariate-leak-regression)
7. [Phase 2: optimization dynamics](#7-phase-2-optimization-dynamics)
8. [Implicit regularization: GD, SGD, and Adam](#8-implicit-regularization-gd-sgd-and-adam)
9. [Summary of findings and limitations](#9-summary-of-findings-and-limitations)
10. [Reproducibility](#10-reproducibility)
11. [Repository structure](#11-repository-structure)
12. [Known inconsistencies](#12-known-inconsistencies)
13. [Background references](#13-background-references)

---

## 1. Project progression and status

The work proceeded in three stages.

| Stage       | Problem                                                                                                                                                   | Data                                                                  | Status                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Stage 0** | Binary leak classification on a 2-junction toy network                                                                                                    | 100,000 simulated operating conditions (80,000 train / 20,000 test)   | Run completed; test accuracy ≈ chance                                                           |
| **Stage 1** | Per-junction leak indicator and emitter-coefficient regression on the KY9 network (1,242 junctions)                                                       | 498 simulated scenarios × 24 hourly rows = 11,952 rows                | 20 optimizer/objective configurations completed; evaluated with a **row-wise** split            |
| **Phase 2** | Scenario-level capacity/noise sweep, gradient-propagation diagnostics, optimizer stability, GD/SGD modified-loss verification, Adam trajectory comparison | 8,952 rows = 373 leak scenarios × 24 h (zero-leak scenarios excluded) | Implemented with saved outputs, figures, and checkpoints; the directory is **untracked** in Git |

### Completed experiments (saved outputs exist)

- Stage 0 toy classifier training run.
- Stage 1: 20 configurations (4 objective/regularization groups × 5 optimizers), each with saved train/test loss plots and a final test accuracy.
- Phase 2: scenario reconstruction and its reliability report; capacity/noise sweep (Part 3); forward/backward propagation study (Part 4A); optimizer stability test (Part 4B); GD toy and leak-network flow comparison (Part 5A, 5B); SGD toy comparison (Part 5C); Adam trajectory comparison (Part 5D).

### Exploratory or auxiliary work (not authoritative)

- The notebooks in `EPANET Experiments/` (toy model experiment, topology exploration, KY9 exploration). Some refer to missing input files, obsolete paths, or topology sizes that differ from the current `ky9.inp` (see [Known inconsistencies](#12-known-inconsistencies)). They should not be treated as reference results.

### Not done

- Cross-topology evaluation, real-world or field validation, calibrated uncertainty, robustness to sensor failure.
- Re-evaluation of the Stage 1 models under a scenario-level split.
- Leak/no-leak detection under a scenario-independent split.
- An Adam-specific modified-loss derivation.

---

## 2. Hydraulic leak model

Hydraulics are computed by EPANET (through WNTR's `EpanetSimulator`); the code does not re-implement the solver. The following summarizes the standard formulation that the simulations rely on.

**Mass conservation at node $i$** (demand-driven analysis, DDA):

$$
\sum_{j} Q_{ji} - \sum_{k} Q_{ik} \;=\; d_i(t) + Q^{\mathrm{leak}}_i(t),
$$

where $Q_{ji}$ are link flows, $d_i(t)$ is the nodal demand, and $Q^{\mathrm{leak}}_i$ is the emitter flow (zero at non-leaking junctions).

**Energy relation along each link** $(i,j)$:

$$
h_i - h_j \;=\; \Delta h_{ij}(Q_{ij}),
$$

with the Hazen–Williams head-loss law used for pipes. Pumps and valves (PRVs in KY9) enter through their own characteristic or control relations. Tank levels evolve over the 24-hour horizon at hourly time steps.

**Emitter (leak) model.** With pressure head $P_i$ in metres, the leak flow at junction $i$ is

$$
Q^{\mathrm{leak}}_i = C_i\,P_i^{0.5},
$$

(emitter exponent 0.5). The generator computes the emitter coefficient from a physical hole size and discharge coefficient:

$$
A=\frac{\pi d^{2}}{4},\qquad C \;=\; C_d\,A\,\sqrt{2g},
$$

so that $Q^{\mathrm{leak}} = C_d A\sqrt{2gP}$. The coefficient $C$ is in SI units (flow in m³/s, pressure head in m). It is **not** a directly measured leak flow.

**Learning problem.** The model maps network-wide measurements to per-junction targets. The input vector is

$$
x = \big[\,\text{demand}_{1242}\;\big|\;\text{pressure}_{1242}\;\big|\;\text{flow}_{1270}\;\big|\;\text{time\_of\_day}\,\big]\in\mathbb R^{3755},
$$

and the targets are $y_j = C_j$ at a leaking junction $j$ and $y_j=0$ otherwise, with leak indicator $c_j=\mathbb 1[y_j>0]$, for $j=1,\dots,1242$.

> **Important detail.** In the generator, `REMOVE_LEAK_FLOW_FROM_DEMAND = False`: the demand measurement at a leaking junction **includes** the emitter flow. This should be taken into account when interpreting any model performance.

---

## 3. Datasets

### 3.1 Stage 0 toy data

- Network (`baby_network.inp`): two junctions, one reservoir, two pipes. Simulated with WNTR's `EpanetSimulator`.
- 100,000 independent operating conditions, NumPy seed 42. Demands and reservoir head are sampled from positive-truncated normal distributions.
- A leak is present with probability 0.5, at junction 1 or junction 3. The emitter coefficient is 40% uniform on $[0.01, 0.2]$, otherwise normal with mean 1 and standard deviation 0.25, lower-bounded at 0.2.
- Saved files: 80,000 training and 20,000 test rows, 10 columns each. Positive-label fractions: 50.0488% (train) and 49.8250% (test).
- The classifier uses only the pressures at junctions 1 and 3 and the flows in the two pipes. It does not use demand, reservoir head, leak location, or emitter coefficient.

### 3.2 Stage 1 KY9 simulation

Authoritative network input: `ky9.inp`.

| Component  | Count |
| ---------- | ----: |
| Junctions  | 1,242 |
| Pipes      | 1,270 |
| Reservoirs |     4 |
| Tanks      |    15 |
| Pumps      |    17 |
| Valves     |    56 |

The generator verifies these counts. Its saved output identifies the valves as PRVs, the pumps as POWER pumps, and the model as DDA with Hazen–Williams head loss and emitter exponent 0.5. The INP file's duration is zero; the generator overrides it to 24 hours with hourly hydraulic, pattern, and reporting time steps.

**Operating-condition randomization** (per scenario, on a copy of the base model):

- Global demand multiplier 0.75–1.30; per-junction log-normal variation with $\sigma=0.15$; perturbed demand patterns.
- Reservoir head offsets of approximately ±3 m, with smooth daily head-pattern variation.
- Initial tank levels sampled over 15–90% of the allowed range.
- Global and per-pipe roughness variation and minor-loss variation.
- Pump speed 0.90–1.05, with a 20% pump-outage probability.
- Valve settings and a 10% valve-closure probability.

**Leak sampling.** Leak counts are balanced by construction over 0, 1, 2, and 3 leaks. Locations are distinct junctions chosen only among junctions whose pressure stays at least 10 m throughout the no-leak run. Hole diameter is log-uniform on 2–15 mm; discharge coefficient is uniform on 0.60–0.80. For each leak scenario the generator first simulates without a leak, then applies emitters and reruns.

**Validity checks.** Non-finite simulations, disconnected scenarios, and scenarios with excessive negative pressure are rejected; closed links must carry negligible flow; the emitter-flow relation is verified for at least 75% of hourly steps.

**Saved generation output.**

| Quantity                                   |                    Value |
| ------------------------------------------ | -----------------------: |
| Scenarios planned                          |                      500 |
| Scenarios succeeded (24 hourly rows each)  |                      498 |
| Rejected attempts that were retried        |                      559 |
| Scenarios skipped after 12 failed attempts | 2 (scenarios 50 and 362) |
| Total rows                                 |                   11,952 |
| Zero / one / two / three-leak scenarios    |    125 / 124 / 125 / 124 |
| Wall-clock time                            | ≈ 2,626 s (about 44 min) |

The generator's raw output describes a table of shape `(11952, 3757)`: 3,755 input features plus two JSON metadata columns (`leak_locations`, `leak_C`). See [Known inconsistencies](#12-known-inconsistencies) for how this relates to the CSV files currently on disk.

### 3.3 Stage 1 files and preprocessing

| File                                        |  Rows | Columns | Role                                             |
| ------------------------------------------- | ----: | ------: | ------------------------------------------------ |
| `pipeline_leak_dataset(train).csv`          | 9,562 |   3,756 | Inputs plus serialized leak-coefficient metadata |
| `pipeline_leak_dataset(test).csv`           | 2,390 |   3,756 | Same                                             |
| `pipeline_leak_dataset_expanded(train).csv` | 9,562 |   4,997 | 3,755 inputs + 1,242 per-junction targets        |
| `pipeline_leak_dataset_expanded(test).csv`  | 2,390 |   4,997 | Same                                             |

Expanded targets are named `leak_C_<junction>` and are zero at non-leaking junctions. The Stage 1 notebook fills missing inputs with zero and standardizes test inputs with training-set means and standard deviations (zero standard deviations are replaced by one).

### Row-wise split problem (Stage 1)

The generator splits the data with `dataset_df.sample(frac=0.8, random_state=42)` and assigns the remaining rows to the test set. Each scenario contributes 24 correlated hourly rows with an **invariant target vector**, so rows of one scenario can appear in both train and test. The Phase 2 diagnostic found that **all 1,837 test leak rows had a same-scenario row in the training set.**

Consequently the Stage 1 test accuracies reflect, at least in part, recognition of scenarios seen during training. They must not be read as generalization to unseen scenarios. The Stage 1 models were not re-evaluated under a scenario-level split.

### Phase 2 pooled scenario-level data

Scenario IDs created in memory by the generator were not saved with the CSVs. The script `Phase 2/export_scenario_ids_phase1.py` pools the expanded train/test CSVs and reconstructs scenario groups:

- **Leak scenarios (≥ 1 leak):** identified **exactly** by their constant leak-coefficient vector.
- **Zero-leak scenarios:** no label fingerprint exists; the script attempts to infer groups from scenario-signature columns and validates the result.

The saved `scenario_reconstruction_report.json` states that the zero-leak reconstruction **failed its reliability checks**: held-out exact recovery was about 0.90, the self-training adjusted Rand index ranged from 0.856 to 0.931, and 36.87% of zero-leak rows were ambiguous. The configured policy is therefore `drop`.

The saved arrays contain only:

| Array                     | Shape        |
| ------------------------- | ------------ |
| `pooled_X.npy`            | (8952, 3755) |
| `pooled_Y.npy`            | (8952, 1242) |
| `pooled_scenario_ids.npy` | (8952,)      |

That is 373 leak scenarios × 24 hours (124 one-leak, 125 two-leak, 124 three-leak), consistent with the 3,000 rows (125 zero-leak scenarios × 24 h) that were removed from the 11,952.

**Consequence:** every Phase 2 result is conditional on a leak being present. The model never sees a no-leak row, and no leak/no-leak detection is evaluated.

Phase 2 preprocessing (training-pool statistics; NaN → 0; sample standard deviation; standard deviations below $10^{-6}$ set to 1) reports 143 near-constant columns and a maximum standardized magnitude of 1,104.2. Targets are divided by the median positive leak coefficient, $7.934\times10^{-5}$, so that the regression head operates at an $O(1)$ scale. As a result, **Phase 2 loss values are not comparable with Stage 1 loss values.**

---

## 4. Models and objectives

### 4.1 Architectures

All main models are fully connected networks with ReLU hidden activations (`nn.Linear`); the optimizers are custom implementations on top of PyTorch autograd.

| Experiment                            | Architecture                                   |                              Outputs |               Parameters |
| ------------------------------------- | ---------------------------------------------- | -----------------------------------: | -----------------------: |
| Stage 0 classifier                    | $4\to50\to50\to1$                              |                              1 logit |                    2,851 |
| Stage 1 homoscedastic                 | $3755\to256\to256\to2484$                      |           1,242 logits + 1,242 means |                1,665,716 |
| Stage 1 heteroscedastic               | $3755\to256\to256\to3726$                      | logits + means + 1,242 log-variances |                1,984,910 |
| Phase 2 capacity sweep                | $3755\to w\to2484$, $w=1\ldots2048$            |                       logits + means |       8,724 – 12,782,004 |
| Phase 2 propagation / optimizer study | $3755\to 64$ (repeated, depth 2–50) $\to 2484$ |                       logits + means | e.g. 480,884 at depth 20 |
| Phase 2 GD leak-network check         | $3755\to16\to2484$, **tanh** hidden activation |                       logits + means |                  102,324 |

For the one-hidden-layer sweep the parameter count is $P(w) = 3755\,w + w + 2484\,w + 2484 = 6240\,w + 2484$.

Stage 0/1 use `nn.init.kaiming_normal_` with zero biases. Phase 2 samples weights explicitly as

$$
W_{ij}\sim\mathcal N\!\left(0,\ \frac{2c}{\text{fan\_in}}\right),
$$

where $c=1$ corresponds to He scaling.

### 4.2 Stage 1 objectives

Let $z_{ij}$ be the leak logit, $\mu_{ij}$ the predicted coefficient, $v_{ij}$ a predicted log-variance, $p_{ij}=\sigma(z_{ij})$, and $c_{ij}=\mathbb 1[y_{ij}>0]$. All means are taken over row–junction pairs.

**Homoscedastic objective:**

$$
L_{\mathrm{homo}} \;=\; \alpha\,\mathrm{BCEWithLogits}(z,c)\;+\;\beta\;\mathrm{mean}\!\left[p\,(y-\mu)^2\right].
$$

**Heteroscedastic objective** ($v$ clamped to $[-4,4]$):

$$
L_{\mathrm{het}} \;=\; \alpha\,\mathrm{BCEWithLogits}(z,c)\;+\;\beta\;\mathrm{mean}\!\left[p\left(\frac{v}{2}+\frac{(y-\mu)^2}{2e^{v}}\right)\right].
$$

**L2 variants** add $\lambda\sum_\ell \lVert W_\ell\rVert_F^2$ with $\lambda=10^{-4}$ (biases excluded). The code uses $\alpha=\beta=1$. Classification is thresholded at $p\ge 0.5$; the regression term is softly gated by $p$.

### 4.3 Stage 1 optimizers and hyperparameters

Stage 1 implements full-batch GD, mini-batch SGD (batch size 64, reshuffled each epoch), SGD with momentum ($\beta=0.9$), RMSprop ($\rho=0.9$), and Adam ($\beta_1=0.9$, $\beta_2=0.999$), with $\epsilon=10^{-8}$. Each configuration runs 1,000 epochs with no learning-rate schedule; GD performs one update per epoch.

| Configuration          |     GD |    SGD | Momentum | RMSprop |   Adam |
| ---------------------- | -----: | -----: | -------: | ------: | -----: |
| Homoscedastic, no L2   |   0.01 |   0.01 |     0.01 |   0.001 |  0.001 |
| Heteroscedastic, no L2 | 0.0001 | 0.0001 |   0.0001 |  0.0001 | 0.0001 |
| Homoscedastic, L2      |  0.001 |  0.001 |    0.001 |   0.001 |  0.001 |
| Heteroscedastic, L2    | 0.0001 | 0.0001 |   0.0001 |  0.0001 | 0.0001 |

The stored loss curves are evaluated _before_ each epoch's updates, so the last stored point does not include that epoch's updates.

---

## 5. Stage 0: toy binary classifier

- Binary cross-entropy, full-batch GD, 10,000 epochs, learning rate 0.01.
- Stored test accuracy: **50.0096%** on a balanced test set (49.8250% positives).

This result is essentially chance. It establishes that the code ran; it does **not** establish useful toy leak detection. The train/test loss plot is saved in the notebook; no model checkpoint is present.

---

## 6. Stage 1: KY9 multivariate leak regression

Final test accuracies (saved in `Phase 1/Model Performances/accuracies.txt`):

| Group | Objective              |       GD |      SGD | Momentum |  RMSprop |     Adam |
| ----- | ---------------------- | -------: | -------: | -------: | -------: | -------: |
| G1    | Homoscedastic, no L2   | 56.6107% | 99.8748% | 99.8751% | 99.9390% | 99.9954% |
| G2    | Heteroscedastic, no L2 | 50.1985% | 58.1524% | 57.2644% | 27.7249% | 45.0766% |
| G3    | Homoscedastic, L2      | 52.4921% | 99.8711% | 99.8750% | 99.8751% | 99.8751% |
| G4    | Heteroscedastic, L2    | 51.8615% | 56.9473% | 59.0033% | 22.3471% | 43.0028% |

The 20 train/test loss plots are stored as `Phase 1/Model Performances/G<g>_<g>.<k>_<Optimizer>.png`, for example:

![Stage 1 G1 Adam](Phase%201/Model%20Performances/G1_1.5_Adam.png)

**How to read these numbers.**

- The heteroscedastic configurations (G2, G4) were substantially less reliable than the homoscedastic ones under these settings.
- The high homoscedastic accuracies are **not** evidence of strong localization:
  1. The test split shares scenarios with the training split (see [Row-wise split problem](#row-wise-split-problem-stage-1)).
  2. The accuracy metric averages over 1,242 junctions per row and is dominated by non-leaking junctions. As a reference calculation (derived from the leak-count mix in §3.2, _not_ a saved result): if accuracy is the fraction of correctly classified row–junction pairs, a predictor that always outputs "no leak" would score about $1 - \tfrac{746/498}{1242} \approx 99.88\%$, which is of the same order as the G1/G3 figures.
  3. No Stage 1 precision–recall, recall@k, or scenario-held-out metric was saved.

---

## 7. Phase 2: optimization dynamics

All Phase 2 results use the 8,952-row, leak-only pooled data described in §3.3 and are saved in `Phase 2/Main_Optimization_Dynamics_v2.ipynb`.

> Figures referenced below live in the untracked `Phase 2/` directory. Commit them for the image links to render on GitHub.

### 7.1 Capacity and noise sweep (Part 3)

**Setup.**

- One hidden ReLU layer, 17 widths from 1 to 2,048.
- Adam, 600 epochs, batch size 64, learning rate $10^{-3}$ with cosine decay to 1% of the base rate; up to two retries at one-third learning rate for failed fits.
- Eight models per width, each trained on a disjoint scenario-level subset of 37 scenarios (about 888 rows). All share one held-out test set of 75 scenarios (1,800 rows).
- Three conditions: **clean**; **20% label noise** (a fraction of training rows have their label vector replaced by that of a randomly chosen training row); **sensor noise** (Gaussian, standard deviation 0.5 in standardized feature units, added to training and test inputs). Test labels are always clean.

**Mean combined test loss over eight fits** (lower is better; rescaled-target units):

| Width | Parameters |        Clean | 20% label noise | Sensor noise (std 0.5) |
| ----: | ---------: | -----------: | --------------: | ---------------------: |
|     1 |      8,724 |     11.79037 |        14.72102 |               16.44442 |
|     2 |     14,964 |      6.17703 |         7.46302 |                8.87070 |
|     4 |     27,444 |      3.14331 |         1.88281 |                1.88257 |
|     8 |     52,404 |      1.97727 |         3.75013 |                0.84025 |
|    16 |    102,324 |      1.30830 |         1.71494 |                0.83807 |
|    32 |    202,164 |      0.66670 |         0.89709 |                0.50874 |
|    64 |    401,844 |      0.36612 |         0.39840 |                0.25670 |
|    96 |    601,524 |      0.12253 |         0.27703 |                0.16522 |
|   128 |    801,204 |      0.12866 |         0.20429 |                0.05297 |
|   192 |  1,200,564 |  **0.05863** |     **0.11203** |                0.05525 |
|   256 |  1,599,924 |      0.15933 |         1.11457 |                0.20750 |
|   384 |  2,398,644 |      2.81791 |         3.76107 |                1.08309 |
|   512 |  3,197,364 |      5.57066 |         9.03095 |                4.22073 |
|   768 |  4,794,804 | **15.19281** |    **32.85728** |                4.62413 |
| 1,024 |  6,392,244 |      5.09672 |         6.26199 |                0.07554 |
| 1,536 |  9,587,124 |      0.18380 |         0.27413 |                0.11401 |
| 2,048 | 12,782,004 |      0.25941 |         0.46550 |                0.15313 |

![Phase 2 capacity sweep, clean](Phase%202/part3_clean.png)
![Phase 2 capacity sweep, overlay](Phase%202/part3_overlay.png)

Also saved: `part3_label_noise.png`, `part3_sensor_noise.png`.

**Observations.**

- In the clean and label-noise conditions the loss falls to a minimum at width 192, rises sharply to a maximum at width 768, and falls again at larger widths. This is **double-descent-like** but is _not_ a clean demonstration of a canonical interpolation-threshold double descent.
- The loss after the second descent (widths 1,536–2,048) remains higher than the first minimum at width 192 in both conditions.
- No interpolation threshold for this multi-output problem was derived, so the location of the width-768 peak is not compared with a theoretical prediction. Even the smallest model (8,724 parameters) has far more parameters than training rows (≈ 888).
- Wide models frequently required learning-rate retries, so **optimization instability is a substantial confound**. Dispersion across the eight fits is not summarized here, and each tabulated mean may be influenced by individual unstable fits.
- The source audit states that the sensor-noise curve has its maximum at width 1 and lacks the interior peak. The tabulated values nevertheless rise from 0.2075 (width 256) to 4.6241 (width 768) before falling to 0.0755 at width 1,024. The interior behavior in that condition is reported here as tabulated and is not further characterized.

**Bias–variance decomposition and task metrics (clean condition).**

For $K$ models trained on disjoint training sets, the notebook decomposes squared error at each (test row, junction) pair using the exact identity

$$
\frac1K\sum_{k=1}^K\big(\hat f_k-y\big)^2 \;=\; \big(\bar f - y\big)^2 \;+\; \frac1K\sum_{k=1}^K\big(\hat f_k-\bar f\big)^2,
\qquad \bar f=\frac1K\sum_k \hat f_k,
$$

applied to two predictions: the leak probability $p$ against the indicator $c$ (Brier score), and the expected intensity $p\,\mu$ against $y$.

| Width | Brier bias² | Brier variance | Intensity bias² | Intensity variance |
| ----: | ----------: | -------------: | --------------: | -----------------: |
|   192 |    0.001587 |        5.74e-5 |         0.01098 |            0.00836 |
|   768 |    0.002218 |       0.004143 |           1.815 |              12.66 |

At width 768 variance exceeds bias² for both targets, consistent with the loss spike in the table above.

**Localization metrics are weak.** At width 192 (clean) the micro-average precision is **0.006644** and recall@3 (fraction of true leak junctions among the three highest-probability junctions, over rows with at least one leak) is **0.02878**. The best saved micro-AP values are approximately **0.00664** (clean), **0.00283** (label noise), and **0.00657** (sensor noise). For context, in the training pool the leak prevalence is only 0.00161 per output, and 753 of 1,242 junctions never leak in the pool. A matching constant-predictor baseline value is not recorded in the source audit.

### 7.2 Forward and backward propagation (Part 4A)

**Mean-field picture for ReLU networks.** Consider a depth-$L$ network with pre-activations $h_{k+1}=W_{k+1}\,\mathrm{ReLU}(h_k)+b_{k+1}$ and weights $W_{k}\sim\mathcal N(0,\,2c/n_{\mathrm{in}})$. Assuming zero-mean symmetric pre-activations and independence (large-width approximation),

$$
\mathbb E\big[h_{k+1}^2\big] \;=\; c\;\mathbb E\big[h_k^2\big]
\quad\Longrightarrow\quad
\mathrm{RMS}(h_k)\;\propto\; c^{(k-1)/2}.
$$

For the backward signal $\delta_k=\partial\mathcal L/\partial h_k$, with $\delta_k = D_k W_{k+1}^{\top}\delta_{k+1}$ and $D_k$ the ReLU activation mask (which halves the second moment),

$$
\mathbb E\big[\delta_k^2\big] \;=\; c\;\mathbb E\big[\delta_{k+1}^2\big]
\quad\Longrightarrow\quad
\mathrm{RMS}(\delta_k)\;\propto\; c^{(L-k)/2}.
$$

Hence $c<1$ attenuates (vanishing) and $c>1$ amplifies (exploding) both signals exponentially in depth, while $c=1$ (He scaling) is approximately variance-preserving. The weight gradient is $\partial\mathcal L/\partial W_k=\delta_k\,a_{k-1}^{\top}$, with $a_{k-1}=\mathrm{ReLU}(h_{k-1})$; under the two scalings above its RMS is expected to be roughly layer-independent while its overall magnitude scales with $c$ and $L$. This remark is a consequence of the stated scalings, not a separately verified result.

**Experiment.** Actual training-pool inputs, batch size 64, width 64, depths $L\in\{2,5,10,20,30,50\}$ and multipliers $c\in\{0.25,0.5,1,2,4\}$ (30 settings), three repeats each. Geometric means of the float64 RMS pre-activations, backpropagated gradients, and weight gradients are compared with the scaling guides above.

![Layer-wise propagation](Phase%202/part4a_layerwise.png)
![Depth scaling](Phase%202/part4a_depth_scaling.png)

The saved output reports "30 settings measured" and does not print a numerical summary table; no numerical slope or threshold is reported. The conclusions are therefore **qualitative** readings of the figures: signals decay for $c<1$, grow for $c>1$, and stay approximately flat for $c=1$. All measurements are at initialization; no full depth-by-depth training curves are reported.

### 7.3 Optimizer stability under initialization (Part 4B)

Setup: a 20-hidden-layer, width-64 network, 400 random mini-batch steps, three seeds, initialization scales $c\in\{0.5,1,2\}$, no gradient clipping. Each optimizer's learning rate is tuned once at He initialization and then held fixed.

| Quantity                                  |                                        Value |
| ----------------------------------------- | -------------------------------------------: |
| Selected SGD learning rate                | 0.1 (reported to lie at the search boundary) |
| Selected Adam learning rate               |                                    0.0316228 |
| Constant-predictor loss                   |                                      0.01482 |
| SGD late-loss score at its selected rate  |                                        0.688 |
| Adam late-loss score at its selected rate |                                    ≈ 0.01065 |

At learning rates of 0.316 and above, SGD diverges in the saved search. Adam reaches a very large loss at 0.316 and diverges at 1 and above. The Adam grid was augmented by a wider fallback grid, which is why the selected rate exceeds the originally listed candidates.

![Optimizer stability](Phase%202/part4b_optimizers.png)

The accompanying reading guide states that SGD's step is proportional to the gradient, whereas Adam divides by a running second-moment estimate, except where the $\epsilon$ floor dominates. These results apply to this architecture, loss, batch stream, and initialization. They are **not** a general ranking of optimizers.

---

## 8. Implicit regularization: GD, SGD, and Adam

This section follows the backward-error-analysis view of finite-step-size gradient methods. A finite step size makes the discrete iterates follow, to higher accuracy, the flow of a _modified_ loss rather than the flow of the original loss $L$.

**Setting.** $L(\phi)$ is smooth (at least $C^3$). Gradient flow is

$$
\frac{d\phi}{dt}=-\nabla L(\phi).
$$

The comparisons below use fixed-horizon flow time $t=n\alpha$ for $n$ steps of size $\alpha$.

### 8.1 Gradient descent

**Update:** $\phi_{n+1}=\phi_n-\alpha\nabla L(\phi_n)$.

**Derivation of the modified loss.** Seek a vector field $f=f_0+\alpha f_1+O(\alpha^2)$ whose time-$\alpha$ flow reproduces one GD step to $O(\alpha^3)$. Expanding the flow,

$$
\phi(t+\alpha)=\phi+\alpha f+\frac{\alpha^2}{2}\,f'f+O(\alpha^3)
=\phi+\alpha f_0+\alpha^2\Big(f_1+\tfrac12 f_0'f_0\Big)+O(\alpha^3).
$$

A GD step is $\phi+\alpha f_0$ with $f_0=-\nabla L$, so the $O(\alpha^2)$ term must vanish:

$$
f_1=-\tfrac12 f_0'f_0 .
$$

Since $f_0'=-\nabla^2L$, we have $f_0'f_0=\nabla^2L\,\nabla L$, and because $\nabla\lVert\nabla L\rVert^2=2\,\nabla^2L\,\nabla L$,

$$
f_1=-\tfrac12\nabla^2L\,\nabla L=-\nabla\!\Big(\tfrac14\lVert\nabla L\rVert^2\Big).
$$

Therefore the modified flow is $\dot\phi=-\nabla\widetilde L_{\mathrm{GD}}$ with

$$
\widetilde L_{\mathrm{GD}}(\phi)\;=\;L(\phi)+\frac{\alpha}{4}\,\lVert\nabla L(\phi)\rVert^{2}.
$$

**Interpretation and limits.**

- The extra term penalizes the squared gradient norm; it is a first-order ($O(\alpha)$) correction obtained by truncating an asymptotic series. It is not an exact identity at finite $\alpha$.
- The local error per step against the modified flow is $O(\alpha^3)$, so at a fixed horizon $T$ the global error is expected to be $O(\alpha^2)$, compared with $O(\alpha)$ against the flow of $L$ itself.
- The derivation assumes a smooth loss. This is why the leak-network check below uses tanh rather than ReLU, whose gradient is discontinuous.
- Whether this regularizer improves generalization was **not** tested here; the experiments below compare trajectories only.

### 8.2 Stochastic gradient descent

**Setting.** Fix a partition of the training data into $B$ equal-size batches with batch losses $L_b$, so that $\sum_b\nabla L_b=B\,\nabla L$. Write $g_b=\nabla L_b$, $g=\nabla L$, $H_b=\nabla^2L_b$, $H=\nabla^2L$. Each epoch visits the batches in a uniformly random order $\sigma$:

$$
\phi_k=\phi_{k-1}-\alpha\,g_{\sigma(k)}(\phi_{k-1}),\qquad k=1,\dots,B.
$$

**One-epoch expansion.** Expanding $g_{\sigma(k)}(\phi_{k-1})$ about $\phi_0$ and using $\phi_{k-1}-\phi_0=-\alpha\sum_{j<k}g_{\sigma(j)}+O(\alpha^2)$,

$$
\phi_B=\phi_0-\alpha\sum_{k}g_{\sigma(k)}+\alpha^2\sum_{j<k}H_{\sigma(k)}\,g_{\sigma(j)}+O(\alpha^3).
$$

**Average over the batch order.** Each ordered pair of distinct batches $(a,b)$ appears as "$a$ before $b$" with probability $\tfrac12$, so

$$
\mathbb E_\sigma\sum_{j<k}H_{\sigma(k)}g_{\sigma(j)}=\tfrac12\sum_{a\neq b}H_b\,g_a
=\tfrac12\Big[\Big(\sum_bH_b\Big)\Big(\sum_ag_a\Big)-\sum_bH_bg_b\Big]
=\tfrac12\Big[B^2Hg-\sum_bH_bg_b\Big],
$$

giving

$$
\mathbb E_\sigma[\phi_B]=\phi_0-\alpha Bg+\frac{\alpha^2}{2}\Big[B^2Hg-\sum_bH_bg_b\Big]+O(\alpha^3).
$$

**Matching a modified flow.** Let $\dot\phi=-\nabla\widetilde L$ with $\widetilde L=L+\alpha\,\Omega$. An epoch corresponds to flow time $B\alpha$, and

$$
\phi(B\alpha)=\phi_0-B\alpha\,g-B\alpha^2\,\nabla\Omega+\frac{(B\alpha)^2}{2}\,Hg+O(\alpha^3).
$$

Matching the $\alpha^2$ terms requires $B\,\nabla\Omega=\tfrac12\sum_bH_bg_b$, i.e. $\nabla\Omega=\nabla\big[\tfrac{1}{4B}\sum_b\lVert g_b\rVert^2\big]$, because $\nabla\lVert g_b\rVert^2=2H_bg_b$. Hence

$$
\widetilde L_{\mathrm{SGD}}=L+\frac{\alpha}{4B}\sum_{b=1}^{B}\lVert\nabla L_b\rVert^{2}.
$$

**Variance form.** Because $\sum_b(g_b-g)=0$, we have $\tfrac1B\sum_b\lVert g_b\rVert^2=\lVert g\rVert^2+\tfrac1B\sum_b\lVert g_b-g\rVert^2$, so

$$
\boxed{\;\widetilde L_{\mathrm{SGD}}=L+\frac{\alpha}{4}\lVert\nabla L\rVert^{2}+\frac{\alpha}{4B}\sum_{b=1}^{B}\lVert\nabla L_b-\nabla L\rVert^{2}\;}
$$

The last term is a mini-batch gradient-variance penalty absent from $\widetilde L_{\mathrm{GD}}$. For $B=1$ the expression reduces to $\widetilde L_{\mathrm{GD}}$.

**Scope.** The statement concerns the **mean** iterate over random batch orders, for a **fixed** partition and equal-size batches, sampled at epoch boundaries (flow time $B\alpha$ per epoch), to first order in $\alpha$.

### 8.3 Numerical verification (completed)

All comparisons measure the Euclidean parameter-space distance at a fixed horizon $T$ between the discrete iterate and a numerically integrated flow (`torchdiffeq`), with the step size $\alpha$ halved repeatedly. Fitted log–log slopes of distance versus $\alpha$ are reported. Expected values are slope $\approx1$ against the flow of $L$ and slope $\ge2$ against the modified flow.

**(A) GD on a two-parameter toy.**
Model $f(x;\phi)=\phi_0+\sin(\phi_1x)$, 16 observations generated from $(0.3,\,1.2)$ with Gaussian noise, mean squared error, start $(-0.5,\,0.8)$; largest Hessian eigenvalue at the start 2.0. Six step sizes from 0.125 down to 0.00390625 (halved each time), using 14 to 448 GD steps, which implies a horizon of $T=14\times0.125=1.75$.

| Comparison                                  | Fitted slope in $\alpha$ | Expected |
| ------------------------------------------- | -----------------------: | -------: |
| GD vs. flow of $L$                          |                 **1.00** |        1 |
| GD vs. flow of $\widetilde L_{\mathrm{GD}}$ |                 **1.97** |      ≥ 2 |

![GD toy](Phase%202/part5a_gd_toy.png)

**(B) GD on the leak network.**
128 randomly selected training-pool rows; a one-hidden-layer, width-16, **tanh** network (102,324 parameters); the homoscedastic leak loss; initialized near a constant predictor. Measured initial sharpness 0.5943; five halved step sizes from 0.4207 to 0.02629 at a fixed horizon of 16.83.

| Comparison                                  | Fitted slope in $\alpha$ | Expected |
| ------------------------------------------- | -----------------------: | -------: |
| GD vs. flow of $L$                          |                 **1.00** |        1 |
| GD vs. flow of $\widetilde L_{\mathrm{GD}}$ |                 **1.97** |      ≥ 2 |

This tests optimization trajectories of a leak-regression network. It is **not** a held-out predictive-performance experiment.

![GD leak network](Phase%202/part5b_gd_leaknet.png)

**(C) SGD on the toy.**
The same 16-point toy, a fixed partition into $B=4$ batches of four, the batch order reshuffled every epoch, and **200,000 independent shuffled runs**. The mean iterate is compared at horizon $T=4$ with step sizes 0.2, 0.1, 0.05, 0.025 (5, 10, 20, 40 epochs respectively).

| Comparison of the mean SGD endpoint   | Fitted slope in $\alpha$ |
| ------------------------------------- | -----------------------: |
| Flow of $L$                           |                     0.97 |
| Flow of $\widetilde L_{\mathrm{GD}}$  |                     0.96 |
| Flow of $\widetilde L_{\mathrm{SGD}}$ |                 **2.14** |

For this toy, the mean shuffled-SGD trajectory is matched far better by the SGD modified flow, which contains the mini-batch variance term, than by the GD modified flow. The finite number of shuffled runs introduces sampling noise in the mean iterate. SGD was verified on the toy only, not on the leak network.

![SGD toy](Phase%202/part5c_sgd_toy.png)

**Scope of the verification.** These are checks on one two-parameter toy and one small smooth leak network, with a fixed horizon and a limited range of step sizes. They confirm the expected error scalings in these settings. They do not by themselves show an effect on generalization.

### 8.4 Adam (separate, empirical only)

**Update** (bias-corrected, $t$ = step count):

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)\,g_t^{\odot2},
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\phi_t=\phi_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

**Why the GD analysis does not transfer.** The update is normalized elementwise. When the gradient varies slowly relative to the averaging windows and $\lvert g\rvert\gg\epsilon$, the per-coordinate step is of size $\sim\alpha$ regardless of the gradient magnitude (sign-like behavior). Under the time map $t=n\alpha$ this is a different dynamics from $\dot\phi=-\nabla L$. This is a heuristic explanation consistent with the observation below, not a derivation carried out in the project.

**Experiment (Part 5D).** Full-batch Adam and GD on the two-parameter toy at horizon $T=4$, with $\alpha\in\{0.05,0.02,0.01,0.005,0.002\}$ (80 to 2,000 steps).

| Method | Distance to the flow of $L$ at $T$                                                |
| ------ | --------------------------------------------------------------------------------- |
| GD     | decreases from $1.787\times10^{-3}$ to $7.106\times10^{-5}$ as $\alpha$ decreases |
| Adam   | stays at approximately 0.194–0.201 across all tested $\alpha$                     |

Under this time mapping, Adam does not approach the gradient flow of $L$, whereas GD does.

![Adam vs. flow](Phase%202/part5d_adam_vs_flow.png)

> **No Adam-specific modified-loss derivation was completed in this project.** A UDL-style analysis for Adam would require Adam's own continuous-time limit and a derivation of its modified loss; neither is provided. The Adam results above are an empirical trajectory comparison only.

---

## 9. Summary of findings and limitations

**Quantitative findings.**

1. The Stage 0 toy classifier reaches **50.0096%** test accuracy, which is chance on a balanced test set.
2. Stage 1 reports up to **99.9954%** accuracy, but this is measured with a row-wise split that leaks scenarios across train and test, and the metric is dominated by non-leaking junctions. No class-sensitive or scenario-held-out metric was saved for Stage 1.
3. Phase 2's scenario-held-out localization is weak: best saved clean micro-AP ≈ **0.00664** and recall@3 ≈ **0.03**, with zero-leak rows absent.
4. Phase 2 records a large, non-monotone, width-dependent test-loss spike (maximum at width 768 in the clean and label-noise conditions) with recovery at larger widths. This is double-descent-like but confounded by optimizer retries and not tied to a derived interpolation threshold.
5. GD and SGD modified-loss trajectory comparisons show the expected error-scaling slopes on a smooth toy and a smooth tanh leak network. The SGD mini-batch variance term is essential in the SGD toy comparison. Adam is compared with gradient flow only empirically.

**What the experiments do not establish.**

- Zero-leak (leak vs. no-leak) detection under a scenario-independent split.
- Generalization to unseen network topologies.
- Calibrated uncertainty or robustness to real sensor failures.
- Real-world utility. The project contains no field data and no external validation.
- General conclusions about optimizers or about generalization benefits of implicit regularization.

---

## 10. Reproducibility

### Dependencies (observed, not pinned)

NumPy, pandas, PyTorch, Matplotlib, WNTR, NetworkX; in Phase 2 additionally SciPy and `torchdiffeq`. The reconstruction script documents NumPy, pandas, and SciPy.

Observed from saved metadata and outputs:

- Notebook metadata records **Python 3.14.5** for most notebooks and **Python 3.11.6** for the older toy EPANET experiment.
- A saved exploratory output reports **WNTR 1.5.0**; the generator itself does not pin or report its WNTR version.
- Temporary EPANET reports identify **EPANET 2.2**.

There is no `requirements.txt`, `pyproject.toml`, environment lockfile, tested install command, or test suite. Phase 2's saved run reports CUDA; the GPU model, driver, and minimum memory are not documented. The Phase 1 code selects CUDA when available and otherwise CPU.

### Execution order

1. Generate or prepare the Stage 1 expanded CSVs (`Phase 1/EPANET/Dataset_Generator.ipynb`). The saved generation run took about 44 minutes. The exact procedure that produces the expanded CSVs from the generator output is not documented here.
2. Run `Phase 2/export_scenario_ids_phase1.py` to pool the CSVs, reconstruct scenario groups, validate them, and write the pooled arrays and `scenario_reconstruction_report.json`. Set the CSV paths at the top of the script.
3. Open `Phase 2/Main_Optimization_Dynamics_v2.ipynb` and run it top to bottom. Its source recommends first setting `SMOKE_TEST=True` to verify the setup, then using the full configuration.

Stage 0 and Stage 1 are run from `Phase 0 (Toy)/Main_binary_classifier.ipynb` and `Phase 1/Main_Multivariate_Regression.ipynb` respectively.

### Known obstacles to a clean rerun

- The script and several notebooks use absolute paths rooted at `C:\Data\pipeline-leakage-system`. The toy classifier/generator and one older exploration notebook instead refer to `C:\Data\Project` or a Linux `/home/...` path.
- The topology-exploration notebook references an INP file that is not present in the workspace.
- The Phase 2 capacity sweep is substantially more compute-intensive than the other experiments; its runtime was not recorded in the source audit.
- The large Phase 1 CSVs are present locally but ignored by Git; the Phase 2 arrays, checkpoints, and results are also untracked.

---

## 11. Repository structure

The workspace has no conventional Python package, application entry point, or C++/CUDA implementation. The only standalone `.py` file is the Phase 2 export/reconstruction script; the implementation is otherwise in notebooks.

```text
pipeline-leakage-system/
├── README.md
├── .gitignore
├── .vscode/
│   └── settings.json                         [untracked]
├── EPANET Experiments/                       [exploratory; not authoritative]
│   ├── EPANET_toy_model_experiment.ipynb
│   ├── exploring_KY9.ipynb
│   ├── Phase_1(Exploring Topologies).ipynb
│   └── temp.inp, temp.bin, temp.rpt
├── Phase 0 (Toy)/
│   ├── Main_binary_classifier.ipynb
│   ├── toy_dataset(train).csv
│   ├── toy_dataset(test).csv
│   └── EPANET/
│       ├── baby_network.inp
│       ├── dataset_generator.ipynb
│       └── temp.inp, temp.bin, temp.rpt
├── Phase 1/
│   ├── Main_Multivariate_Regression.ipynb
│   ├── pipeline_leak_dataset(train).csv          [ignored by Git]
│   ├── pipeline_leak_dataset(test).csv           [ignored by Git]
│   ├── pipeline_leak_dataset_expanded(train).csv [ignored by Git]
│   ├── pipeline_leak_dataset_expanded(test).csv  [ignored by Git]
│   ├── EPANET/
│   │   ├── Dataset_Generator.ipynb
│   │   └── ky9.inp
│   └── Model Performances/
│       ├── accuracies.txt
│       └── 20 loss plots: G1_1.1_GD.png ... G4_4.5_Adam.png
│           (groups G1-G4 x optimizers GD, SGD, SGDM, RMSprop, Adam)
└── Phase 2/                                  [untracked]
    ├── Main_Optimization_Dynamics_v2.ipynb
    ├── export_scenario_ids_phase1.py
    ├── scenario_reconstruction_report.json
    ├── pooled_columns.json
    ├── pooled_X.npy, pooled_Y.npy, pooled_scenario_ids.npy
    ├── part3_clean.pkl, part3_label_noise.pkl, part3_sensor_noise.pkl   (checkpoints)
    ├── part3_clean.png, part3_label_noise.png, part3_sensor_noise.png, part3_overlay.png
    ├── part4a_layerwise.png, part4a_depth_scaling.png, part4b_optimizers.png
    ├── part5a_gd_toy.png, part5b_gd_leaknet.png, part5c_sgd_toy.png, part5d_adam_vs_flow.png
    └── smoke_test/                           (smoke-run checkpoint and figures)
```

Phase 2 and the `.vscode` settings are untracked in the current Git worktree.

---

## 12. Known inconsistencies

These were identified during a read-only audit of the workspace and are reported here rather than resolved.

1. **Earlier README status.** A previous version of the README described the optimization-dynamics work as planned, whereas the untracked Phase 2 notebook has saved executed outputs, figures, and checkpoints. This README describes it as implemented but untracked.
2. **Stage 1 split.** The 80/20 split is row-wise over correlated hourly rows; all 1,837 test leak rows had a same-scenario training row.
3. **Phase 2 coverage.** Zero-leak scenarios are excluded because their reconstruction failed the reliability checks.
4. **He-initialization equation.** An earlier README gave the weight variance as $4/(D_h+D_h')$, which does not match the code: Stage 0/1 use Kaiming normal and Phase 2 uses $2c/\text{fan\_in}$.
5. **Generator output vs. CSV dimensions.** The saved generator output reports a raw table of shape `(11952, 3757)` (3,755 inputs + two metadata columns). The base CSVs currently on disk have **3,756** columns and the expanded CSVs have 4,997. The expanded files are shape-compatible with the Stage 1 regression notebook, but the saved generator narrative and the current files do not form a fully consistent provenance record.
6. **Stale exploratory notebooks.** `Phase_1(Exploring Topologies).ipynb` references `Phase 1\EPANET\Experiment_network.inp`, which is absent, and its saved output reports 1,096 junctions versus 1,242 in the current `ky9.inp` and generator checks. `exploring_KY9.ipynb` should therefore not be treated as authoritative for topology dimensions. The toy exploration notebook uses a Linux `/home/...` path and its saved outputs include implausibly large negative pressures after demand changes.
7. **Absolute paths.** Several notebooks use paths that differ between files (`C:\Data\pipeline-leakage-system`, `C:\Data\Project`, a Linux path).
8. **Performance claims.** Stage 1 accuracies are not independent-scenario results and lack class-sensitive localization metrics; Phase 2 clean scenario-level localization remains weak.
9. **Reproduction.** There are no pinned package versions, installation instructions, model checkpoints, or recorded hardware details, so a fresh environment cannot be reconstructed exactly from the repository alone.
10. **Sensor-noise curve.** The audit text and the tabulated sensor-noise losses differ in how they describe the interior behavior of that curve (see §7.1).

---

## 13. Background references

These are cited for context on the theory used in §8; they are not claimed to be the sources of the project's code.

- S. J. D. Prince, _Understanding Deep Learning_, MIT Press, 2023 (finite-step-size gradient descent and implicit regularization).
- D. G. T. Barrett and B. Dherin, "Implicit Gradient Regularization," ICLR 2021 (backward error analysis of gradient descent).
- S. L. Smith, B. Dherin, D. G. T. Barrett, and S. De, "On the Origin of Implicit Regularization in Stochastic Gradient Descent," ICLR 2021 (modified loss for SGD with random shuffling).
