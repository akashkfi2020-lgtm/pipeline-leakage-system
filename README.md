# Pipeline Leak Detection

This project explores machine-learning methods for detecting leaks in pipeline networks using simulated hydraulic data. It begins with a binary leak/no-leak classifier on a toy network and progressively extends toward larger network topologies and multivariate prediction of leak locations and intensities.

The project focuses on developing neural networks from scratch, studying their optimization and generalization properties, and eventually investigating the integration of machine learning with governing hydraulic physics.

## Project Overview

| Stage   | Focus                                                     | Status    |
| :------ | :-------------------------------------------------------- | :-------- |
| Stage 0 | Baseline binary leak detection on a toy network           | Completed |
| Stage 1 | Larger topologies and multivariate regression             | Completed |
| Stage 2 | Model capacity, generalization, and optimization dynamics | Planned   |

---

# Stage 0 — Baseline Leak Detection

<sub>Status: Completed</sub>

A baseline binary neural-network classifier was developed to distinguish between leak and no-leak cases using simulated hydraulic data.

### Implementation Details

**1. Neural Network Architecture**

A fully connected neural network was constructed from scratch in PyTorch.

$$
\begin{Bmatrix}
    \vec{h}_1 = a[\beta_0 + \Omega_0 \vec{x}] \\
    \vec{h}_2 = a[\beta_1 + \Omega_1 \vec{h}_1] \\
    \vec{h}_3 = a[\beta_2 + \Omega_2 \vec{h}_2] \\
    \vdots \qquad \vdots \qquad \vdots \\
    \vec{h}_K = a[\beta_{K-1} + \Omega_{K-1}\vec{h}_{K-1}] \\
    f[\vec{x}, \bold{\phi}] = \beta_K + \Omega_K \vec{h}_K
\end{Bmatrix}
$$

**2. He Initialization**

$$
\sigma_\Omega^2 = \frac{4}{D_h + D_h'}
$$

**3. Binary Cross-Entropy Loss**

$$
\boxed{
\begin{aligned}
L[\bold{\phi}]
={}&
\displaystyle\sum_{i=1}^{I}
\Big\{
-(1-y_i)
\log\left[
1-\text{sig}[f[\vec{x}_i,\bold{\phi}]]
\right]
\\
&-
y_i
\log\left[
\text{sig}[f[\vec{x}_i,\bold{\phi}]]
\right]
\Big\}
\\[6pt]
\text{where,}\qquad
\text{sig}[z]
={}&
\frac{1}{1+\exp[-z]}
\end{aligned}
}
$$

**4. Vanilla Gradient Descent**

A custom full-batch Gradient Descent training loop was implemented using PyTorch autograd.

$$
\boxed{
\vec{\phi}
\leftarrow
\vec{\phi}-\alpha\nabla L[\vec{\phi}]
}
$$

**5. Input Preprocessing and Evaluation**

- Input-feature standardization using training-set statistics.
- Evaluation on a separate test set.

**Implementation:** The baseline notebook and toy-network data are available in [`Phase 0 (Toy)`](./Phase%200%20%28Toy%29/).

---

# Stage 1 — Larger Topologies and Multivariate Regression

<sub>Status: Completed</sub>

Stage 1 extends the baseline classifier to larger pipeline networks and reformulates the problem as multivariate prediction of leak locations and intensities.

The implementation uses a custom Deep Neural Network (DNN), built entirely from scratch, to predict emitter coefficients across a large-scale hydraulic network.

### Dataset Specifications

| Parameter             |    Value |
| :-------------------- | -------: |
| Junctions             |     1242 |
| Pipes                 |     1270 |
| Input dimensionality  |     3755 |
| Target dimensionality |     1242 |
| Hidden layers         |        2 |
| Hidden-layer widths   | 256, 256 |

The 3755-dimensional input vector comprises demand, pressure, flowrate, and time-of-day features. The target is a 1242-dimensional vector of Emitter Coefficients representing leak intensities at individual junctions.

## 1. Network Architecture and Variables

### 1.1 Architecture Construction

The network consists of multiple fully connected layers with activation functions applied to the hidden layers.

$$
\begin{Bmatrix}
    \vec{h}_1 = a[\vec{\beta_0} + \Omega_0 \vec{x}] \\
    \vec{h}_2 = a[\vec{\beta_1} + \Omega_1 \vec{h}_1] \\
    \vec{h}_3 = a[\vec{\beta_2} + \Omega_2 \vec{h}_2] \\
    \vdots \qquad \vdots \qquad \vdots \\
    \vec{h}_K = a[\vec{\beta}_{K-1} + \Omega_{K-1}\vec{h}_{K-1}] \\
    f[\vec{x}, \bold{\phi}] = \vec{\beta}_K + \Omega_K \vec{h}_K
\end{Bmatrix}
$$

### 1.2 Network Outputs

Given an input batch $\mathcal{B}_b$, the network parameters $\phi$ map inputs $\mathbf{x}_i$ to a concatenated output vector, which is split into three components per junction $j$:

- **Logits ($z_{ij}$):** Used for binary classification (leak vs. no-leak).
- **Mean ($\mu_{ij}$):** Predicted emitter coefficient (leak intensity).
- **Log-Variance ($v_{ij}$):** Predicted uncertainty (only for heteroscedastic models).

From these, we derive:

**Leak Probability**

$$
p_{ij}
=
\text{sig}[z_{ij}]
=
\frac{1}{1+\exp[-z_{ij}]}
$$

**Binary Target**

$$
y^c_{ij} = \mathbb{I}[y_{ij} > 0]
$$

### 1.3 He Initialization

He/Kaiming Normal initialization is used for the network weights, with biases initialized to zero.

$$
\sigma_\Omega^2 = \frac{4}{D_h + D_h'}
$$

## 2. Loss Functions

A dual-objective loss is employed, combining:

- **Binary Cross-Entropy (BCE):** For leak localization.
- **Soft-Gated Regression:** For predicting leak intensities.

The regression objective is gated by the predicted leak probability $p_{ij}$, allowing the model to focus on junctions it believes are leaking. All losses are averaged over the batch.

### 2.1 Homoscedastic Loss (Constant Variance Assumed)

$$
\boxed{
\begin{aligned}
L[\phi] ={}&
\alpha \underbrace{
\frac{1}{|\mathcal{B}_b| \cdot 1242}
\sum_{i \in \mathcal{B}_b}
\sum_{j=1}^{1242}
\Big[
-(1-y^c_{ij})\log(1-p_{ij})
-y^c_{ij}\log(p_{ij})
\Big]
}_{\text{Binary Cross-Entropy}}
\\[8pt]
&+
\beta \underbrace{
\frac{1}{|\mathcal{B}_b| \cdot 1242}
\sum_{i \in \mathcal{B}_b}
\sum_{j=1}^{1242}
p_{ij}(y_{ij}-\mu_{ij})^2
}_{\text{Leak-Weighted Least Squares}}
\end{aligned}
}
$$

### 2.2 Heteroscedastic Loss (Input-Dependent Variance)

The variance is parameterized as $\sigma^2_{ij} = \exp(v_{ij})$. For numerical stability during training, $v_{ij}$ is clamped to $[-4.0, 4.0]$.

$$
\boxed{
\begin{aligned}
L[\phi] ={}&
\alpha L_{\text{cls}}[\phi]
\\[6pt]
&+
\beta \frac{1}{|\mathcal{B}_b| \cdot 1242}
\sum_{i \in \mathcal{B}_b}
\sum_{j=1}^{1242}
p_{ij}
\left[
\frac{1}{2}v_{ij}
+
\frac{(y_{ij}-\mu_{ij})^2}
{2\exp(v_{ij})}
\right]
\end{aligned}
}
$$

### 2.3 L2 Regularization

L2 regularization is added to the loss to penalize large weights, excluding biases.

$$
\boxed{
L_{L2}[\phi]
=
L[\phi]
+
\lambda \sum_l \|W_l\|_F^2
}
$$

## 3. Optimizers

Five custom optimizers were implemented using raw PyTorch autograd.

Let

$$
g_t
=
\displaystyle\sum_{i\in\mathcal{B}_t}
{\nabla_{\mathbf{\phi}}l_i[\mathbf{\phi_t}]}
$$

be the gradient of the batch loss.

### 3.1 Vanilla Gradient Descent

$$
\boxed{
\phi_{t+1}
\leftarrow
\phi_t-\alpha
\displaystyle\sum_{i=1}^{I}
{\nabla_{\mathbf{\phi}}l_i[\mathbf{\phi_t}]}
}
$$

Here, $\mathcal{B}_t$ is the full batch.

### 3.2 Stochastic Gradient Descent (SGD)

$$
\boxed{
\phi_{t+1}
\leftarrow
\phi_t-\alpha g_t
}
$$

Here, $\mathcal{B}_t$ is a mini-batch.

### 3.3 SGD with Momentum

$$
\boxed{
\begin{aligned}
\mathbf{m}_{t+1}
&\leftarrow
\beta\mathbf{m}_t+(1-\beta)g_t
\\[6pt]
\phi_{t+1}
&\leftarrow
\phi_t-\alpha\mathbf{m}_{t+1}
\end{aligned}
}
$$

### 3.4 RMSprop

$$
\boxed{
\begin{aligned}
\mathbf{v}_{t+1}
&\leftarrow
\rho\mathbf{v}_t+(1-\rho)g_t^2
\\[6pt]
\phi_{t+1}
&\leftarrow
\phi_t-
\alpha
\frac{g_t}{\sqrt{\mathbf{v}_{t+1}}+\epsilon}
\end{aligned}
}
$$

### 3.5 Adam

$$
\boxed{
\begin{aligned}
\mathbf{m}_{t+1}
&\leftarrow
\beta\mathbf{m}_t+(1-\beta)g_t
\\[6pt]
\mathbf{v}_{t+1}
&\leftarrow
\gamma\mathbf{v}_t+(1-\gamma)g_t^2
\\[8pt]
\hat{\mathbf{m}}_{t+1}
&=
\frac{\mathbf{m}_{t+1}}{1-\beta^{t+1}},
\qquad
\hat{\mathbf{v}}_{t+1}
=
\frac{\mathbf{v}_{t+1}}{1-\gamma^{t+1}}
\\[8pt]
\phi_{t+1}
&\leftarrow
\phi_t-
\alpha
\frac{\hat{\mathbf{m}}_{t+1}}
{\sqrt{\hat{\mathbf{v}}_{t+1}}+\epsilon}
\end{aligned}
}
$$

## 4. Experimental Setup and Hyperparameters

The repository evaluates **20 experimental configurations** (4 loss/regularization configurations $\times$ 5 optimizers).

### Architecture and Training

| Component              | Configuration                   |
| :--------------------- | :------------------------------ |
| Architecture           | `DNN(3755, [256, 256], Output)` |
| Homoscedastic output   | `2 * 1242`                      |
| Heteroscedastic output | `3 * 1242`                      |
| Weight initialization  | He/Kaiming Normal               |
| Bias initialization    | Zero                            |
| Epochs                 | 1000                            |
| Mini-batch size        | 64                              |
| L2 coefficient         | `1e-4`                          |

### Experimental Configurations

| Group | Regression Loss | L2 Regularization | Optimizers |
| :---- | :-------------- | :---------------- | :--------- |
| 1     | Homoscedastic   | No                | All five   |
| 2     | Heteroscedastic | No                | All five   |
| 3     | Homoscedastic   | Yes               | All five   |
| 4     | Heteroscedastic | Yes               | All five   |

### Learning Rates

| Configuration               | Optimizers        | Learning Rate |
| :-------------------------- | :---------------- | ------------: |
| Homoscedastic, without L2   | GD, SGD, Momentum |        `0.01` |
| Homoscedastic, without L2   | RMSprop, Adam     |       `0.001` |
| Heteroscedastic, without L2 | All five          |      `0.0001` |
| Homoscedastic, with L2      | All five          |       `0.001` |
| Heteroscedastic, with L2    | All five          |      `0.0001` |

The learning rate for heteroscedastic configurations was reduced to `0.0001` to improve numerical stability and prevent NaNs associated with early variance collapse.

### Evaluation and Artifacts

- **Training and Test Loss:** Loss trajectories are recorded across epochs for all experimental configurations.
- **Leak Detection Accuracy:** Test accuracy is calculated by thresholding the predicted leak probabilities at $p_{ij} \geq 0.5$ and comparing them against the binary targets $y^c_{ij}$.
- **Performance Plots:** Training/test loss trajectories are exported for all model configurations.
- **Accuracy Records:** Final test classification accuracies are saved to `accuracies.txt`.

## 5. Empirical Observations

The heteroscedastic regression loss

$$
L_{\text{het}}[\phi]
=
\sum_i
\left[
\frac{1}{2}\log(\sigma_i^2)
+
\frac{(y_i-f_1[\mathbf{x}_i,\phi])^2}{2\sigma_i^2}
\right]
$$

introduces a variance-dependent gradient scaling through $1/\sigma_i^2$. In our shared architecture, this can lead to unstable regression gradients and interfere with classification learning, particularly when predicted variances become excessively small.

Empirically, the standard least-squares loss

$$
L[\phi]
=
\sum_i
\left(y_i-f[\mathbf{x}_i,\phi]\right)^2
$$

proved substantially more effective and stable.

## 6. Completed Investigations

The following investigations, initially outlined during the development of the project, have been implemented:

- **Problem Reformulation:** Moving from binary classification (leak/no-leak) to multivariate regression (leak location and magnitude).
- **Model Performance and Hyperparameter Selection:** Studying training/test loss trajectories and test classification accuracy across different experimental configurations, with empirical selection of learning-rate settings.
- **Optimizer Comparison:** Comparing Vanilla Gradient Descent, SGD, SGD with Momentum, RMSprop, and Adam.
- **Loss Function Comparison:** Investigating homoscedastic and heteroscedastic regression objectives.
- **Regularization:** Comparing models with and without L2 regularization.
- **Custom Implementations:** Implementing the optimizer update rules and loss functions using PyTorch autograd rather than relying on built-in optimizer classes.

**Implementation and Data:**

- [`Phase 1`](./Phase%201/) — Larger-network experiments, multivariate-regression notebook, datasets, and EPANET network model.
- [`EPANET Experiments`](./EPANET%20Experiments/) — Exploratory notebooks for network topologies and EPANET simulations.

---

# Stage 2 — Model Capacity, Generalization and Optimization Dynamics

<sub>Status: Planned</sub>

Stage 2 will investigate the relationship between neural-network capacity, generalization, and optimization dynamics. The objective is to move beyond comparing fixed architectures and study how the underlying properties of neural networks evolve as their capacity increases.

### 1. Bias–Variance Trade-off and Double Descent

Investigate the effect of model capacity on training and test error by training the network across a series of model sizes.

- Systematically vary the number of hidden variables and corresponding trainable parameters.
- Record training and test errors for each model capacity.
- Study the classical bias–variance trade-off in the underparameterized regime.
- Identify the interpolation threshold, where the number of trainable weights approaches the number of training observations:

$$
N(\text{weights}) = N(\text{train})
$$

- Investigate the transition into the overparameterized regime and study the double-descent phenomenon, where generalization error may initially increase and subsequently decrease as model capacity grows.

The objective is to construct a model-capacity versus error plot, distinguishing the classical and modern regimes.

### 2. Continuous-Time Gradient Descent and Implicit Regularization

Investigate the optimization trajectory of neural networks by comparing continuous-time gradient flow with actual discrete Gradient Descent updates.

- Formulate and simulate the differential equation governing the continuous gradient descent trajectory.
- Compare the resulting continuous-time trajectory with the discrete parameter updates produced by Gradient Descent.
- Perform these comparisons across different model capacities.
- Investigate how the learning rate and model capacity influence the differences between continuous and discrete optimization trajectories.
- Analyze the implicit regularization induced by the optimization process and investigate how it influences the solutions learned by networks of different capacities.

The continuous-time gradient flow is described by:

$$
\frac{d\phi(t)}{dt}
=
-\nabla_\phi L[\phi(t)]
$$

The corresponding discrete Gradient Descent update is:

$$
\boxed{
\phi_{t+1}
\leftarrow
\phi_t-\alpha\nabla_\phi L[\phi_t]
}
$$

The objective is to understand how the optimization trajectory itself influences the solution obtained, beyond the effect of explicitly introduced regularization terms.

### 3. Vanishing and Exploding Gradient Analysis

Investigate the propagation of gradients through the network and characterize how gradient magnitudes evolve with model capacity.

- Record and analyze gradient magnitudes across network layers during training.
- Investigate the emergence of vanishing gradients as network capacity and architecture change.
- Identify conditions under which gradients become excessively large, resulting in exploding gradients and unstable updates.
- Compare gradient behaviour across different model capacities and training configurations.
- Relate gradient propagation to training stability and generalization behaviour.

### Summary of Planned Investigations

Current Phase 2 investigations include:

- Studying the bias–variance trade-off and double descent by training networks across a range of model capacities.
- Simulating continuous-time gradient flow and comparing it with actual discrete Gradient Descent trajectories.
- Investigating implicit regularization and its dependence on model capacity and optimization dynamics.
- Analyzing vanishing and exploding gradients across different model capacities.

---

# Future Scope

Beyond the investigations outlined in Phase 2, the project aims to explore the integration of machine learning with the underlying physics of pipeline networks.

Planned longer-term directions include:

- **Fluid Mechanics:** Incorporate more detailed fluid-mechanics physics, including flow phenomena such as turbulence.
- **Physics-Informed Machine Learning:** Develop a physics-informed ML architecture with custom loss terms constrained by governing fluid-mechanics differential equations.
- **Real-Time Monitoring:** Extend the system toward real-time structural-health monitoring and leak detection.
- **Computational Acceleration:** Investigate C++/CUDA implementations for computational acceleration and real-time deployment.

---

# Repository Layout

| Directory                                       | Description                                                                                          |
| :---------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| [`Phase 0 (Toy)`](./Phase%200%20%28Toy%29/)     | Baseline binary classifier, toy-network datasets, and an EPANET dataset generator.                   |
| [`Phase 1`](./Phase%201/)                       | Larger-network experiments, multivariate-regression notebook, datasets, and an EPANET network model. |
| [`EPANET Experiments`](./EPANET%20Experiments/) | Exploratory notebooks for network topologies and EPANET simulations.                                 |

---

# Notes

This project is under active development. The notebooks and datasets document the experiments as they evolve. Phase 2 and the longer-term directions described above represent planned investigations and should not be interpreted as completed functionality.
