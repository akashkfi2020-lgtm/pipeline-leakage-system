# Pipeline Leak Detection

This project explores machine-learning methods for detecting leaks in pipeline networks using simulated hydraulic data. It starts with a binary leak/no-leak classifier on a toy network and is being extended toward larger topologies and multivariate prediction.

## Project stages

### Stage 0 — Baseline leak detection _(completed)_

A baseline binary neural-network classifier was built to distinguish between leak and no-leak cases using simulated hydraulic data.

The implementation includes:

- A neural network built from scratch in PyTorch.

  **Architecture Construction:**

  <p align="center">
  <strong>h</strong><sub>1</sub> = a[β<sub>0</sub> + Ω<sub>0</sub><strong>x</strong>]
  <br><br>
  <strong>h</strong><sub>2</sub> = a[β<sub>1</sub> + Ω<sub>1</sub><strong>h</strong><sub>1</sub>]
  <br><br>
  <strong>h</strong><sub>3</sub> = a[β<sub>2</sub> + Ω<sub>2</sub><strong>h</strong><sub>2</sub>]
  <br><br>
  ⋮ &nbsp;&nbsp;&nbsp;&nbsp; ⋮ &nbsp;&nbsp;&nbsp;&nbsp; ⋮
  <br><br>
  <strong>h</strong><sub>K</sub> = a[β<sub>K−1</sub> + Ω<sub>K−1</sub><strong>h</strong><sub>K−1</sub>]
  <br><br>
  f[<strong>x</strong>, <strong>φ</strong>] = β<sub>K</sub> + Ω<sub>K</sub><strong>h</strong><sub>K</sub>
  </p>

- He initialization:

  <p align="center">
  σ<sub>Ω</sub><sup>2</sup> = 4 / (D<sub>h</sub> + D<sub>h</sub>′)
  </p>

- Binary Cross-Entropy as the loss function:

  <p align="center">
  L[<strong>φ</strong>] = ∑<sub>i=1</sub><sup>I</sup> { −(1 − y<sub>i</sub>) log[1 − sig(f[<strong>x</strong><sub>i</sub>, <strong>φ</strong>])]
  <br><br>
  − y<sub>i</sub> log[sig(f[<strong>x</strong><sub>i</sub>, <strong>φ</strong>])] }
  </p>

  where,

  <p align="center">
  sig[z] = 1 / (1 + exp[−z])
  </p>

- A custom vanilla Gradient Descent training loop using PyTorch autograd (full batch):

  <p align="center">
  <strong>φ</strong> ← <strong>φ</strong> − α ∇L[<strong>φ</strong>]
  </p>

- Input-feature standardization using training-set statistics, followed by evaluation on a separate test set.

The baseline notebook and toy-network data are in [`Phase 0 (Toy)`](<Phase 0 (Toy)>).

### Stage 1 — Larger topologies and multivariate regression _(in progress)_

The project is extending the baseline from a toy topology to larger, more realistic pipeline networks. The problem is also moving beyond binary detection toward multivariate regression for leak severity and location-dependent quantities.

Current and planned investigations include:

- Moving from Binary Classification (leak/no leak) to Multivariate Regression (leak location + magnitude)
- Studying model performance and hyperparameter selection across experimental settings.
- Examining the bias–variance tradeoff in the classical regime and double descent in the overparameterized regime.
- Investigating vanishing and exploding gradients and other training-related failure modes.
- Comparing optimizers, loss functions, and regularization techniques, including custom PyTorch implementations.

The Phase 1 notebook and datasets are in [`Phase 1`](<Phase 1>). Network-topology experiments are in [`EPANET Experiments`](<EPANET Experiments>).

### Future scope

- Incorporate more detailed fluid-mechanics physics, including flow phenomena such as turbulence.
- Develop a physics-informed ML architecture with custom loss terms constrained by governing fluid-mechanics differential equations.
- Extend the system toward real-time structural-health monitoring and leak detection.
- Investigate C++/CUDA implementations for computational acceleration and real-time deployment.

## Repository layout

- [`Phase 0 (Toy)`](<Phase 0 (Toy)>) — baseline binary classifier, toy-network datasets, and an EPANET dataset generator.
- [`Phase 1`](<Phase 1>) — larger-network experiments, multivariate-regression notebook, datasets, and an EPANET network model.
- [`EPANET Experiments`](<EPANET Experiments>) — exploratory notebooks for network topologies and EPANET simulations.
- [`All Implementations`](<All Implementations>) — implementation notebooks.

## Notes

The project is under active development. The notebooks and datasets document the experiments as they evolve; planned work and future scope above should not be read as completed functionality.
