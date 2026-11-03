# Pipeline Leak Detection

This project explores machine-learning methods for detecting leaks in pipeline networks using simulated hydraulic data. It starts with a binary leak/no-leak classifier on a toy network and is being extended toward larger topologies and multivariate prediction.

## Project stages

### Stage 0 — Baseline leak detection _(completed)_

A baseline binary neural-network classifier was built to distinguish between leak and no-leak cases using simulated hydraulic data.

The implementation includes:

- A neural network built from scratch in PyTorch.

  **Architecture Construction:**

  ![Neural Network Architecture](https://render.githubusercontent.com/render/math?math=%5Cbegin%7BBmatrix%7D%5Cvec%7Bh%7D_1%3Da%5B%5Cbeta_0%2B%5COmega_0%5Cvec%7Bx%7D%5D%5C%5C%5Cvec%7Bh%7D_2%3Da%5B%5Cbeta_1%2B%5COmega_1%5Cvec%7Bh%7D_1%5D%5C%5C%5Cvec%7Bh%7D_3%3Da%5B%5Cbeta_2%2B%5COmega_2%5Cvec%7Bh%7D_2%5D%5C%5C%5Cvdots%5Cquad%5Cvdots%5Cquad%5Cvdots%5C%5C%5Cvec%7Bh%7D_K%3Da%5B%5Cbeta_%7BK-1%7D%2B%5COmega_%7BK-1%7D%5Cvec%7Bh%7D_%7BK-1%7D%5D%5C%5Cf%5B%5Cvec%7Bx%7D%2C%5Cboldsymbol%7B%5Cphi%7D%5D%3D%5Cbeta_K%2B%5COmega_K%5Cvec%7Bh%7D_K%5Cend%7BBmatrix%7D)

- He initialization:

  ![He Initialization](https://render.githubusercontent.com/render/math?math=%5Csigma_%5COmega%5E2%3D%5Cfrac%7B4%7D%7BD_h%2BD_h%27%7D)

- Binary Cross-Entropy as the loss function:

  ![Binary Cross Entropy](https://render.githubusercontent.com/render/math?math=L%5B%5Cboldsymbol%7B%5Cphi%7D%5D%3D%5Csum_%7Bi%3D1%7D%5E%7BI%7D%5Cleft%5C%7B-%281-y_i%29%5Clog%5Cleft%5B1-%5Coperatorname%7Bsig%7D%5Bf%5B%5Cvec%7Bx%7D_i%2C%5Cboldsymbol%7B%5Cphi%7D%5D%5D%5Cright%5D-y_i%5Clog%5Cleft%5B%5Coperatorname%7Bsig%7D%5Bf%5B%5Cvec%7Bx%7D_i%2C%5Cboldsymbol%7B%5Cphi%7D%5D%5D%5Cright%5D%5Cright%5C%7D)

  where,

  ![Sigmoid Function](https://render.githubusercontent.com/render/math?math=%5Coperatorname%7Bsig%7D%5Bz%5D%3D%5Cfrac%7B1%7D%7B1%2B%5Cexp%5B-z%5D%7D)

- A custom vanilla Gradient Descent training loop using PyTorch autograd (full batch):

  ![Gradient Descent Update](https://render.githubusercontent.com/render/math?math=%5Cvec%7B%5Cphi%7D%5Cleftarrow%5Cvec%7B%5Cphi%7D-%5Calpha%20%5Cnabla%20L%5B%5Cboldsymbol%7B%5Cphi%7D%5D)

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
