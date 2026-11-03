# Pipeline Leak Detection

This project explores machine-learning methods for detecting leaks in pipeline networks using simulated hydraulic data. It starts with a binary leak/no-leak classifier on a toy network and is being extended toward larger topologies and multivariate prediction.

## Project stages

### Stage 0 — Baseline leak detection _(completed)_

A baseline binary neural-network classifier was built to distinguish between leak and no-leak cases using simulated hydraulic data.

The implementation includes:

- A neural network built from scratch in PyTorch.
  Architecture Construction:
  $$
  \begin{Bmatrix}
      \vec{h}_1 = a[\beta_0 + {\Omega_0 \vec{x}}] \\
      \vec{h}_2 = a[\beta_1 + {\Omega_1 \vec{h}_1}] \\
      \vec{h}_3 = a[\beta_2 + {\Omega_2 \vec{h}_2}] \\
      \vdots \qquad \vdots \qquad \vdots \\
      \vec{h}_K = a[\beta_{K-1} + {\Omega_{K-1} \vec{h}_{K-1}}] \\
      f[\vec{x}, \bold{\phi}] = \beta_K + {\Omega_K \vec{h}_K}
  \end{Bmatrix}
  $$
- He initialization:
  $$
  \sigma_\Omega^2 = \frac{4}{D_h + D_h'}
  $$
- Binary Cross-Entropy as the loss function:
  $$
  L[\bold{\phi}] = \displaystyle\sum_{i=1}^{I}{-(1-y_i) \log[1-\text{sig}[f[\vec{x}_i, \bold{\phi}]]] - y_i \log[\text{sig}[f[\vec{x}_i, \bold{\phi}]]]} \\
  \text{where,} \qquad
  \text{sig}[z] = \frac{1}{1+\exp[-z]}
  $$
- A custom vanilla Gradient Descent training loop using PyTorch autograd (full batch)
  $$
  \vec{\phi} \leftarrow \vec{\phi} - \alpha \nabla{L[\vec{\phi}]}
  $$
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
