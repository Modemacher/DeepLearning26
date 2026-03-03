# Universal Function Approximator – Explanation

## What is a Universal Function Approximator?

A **universal function approximator** is a model that can approximate *any continuous function* on a bounded interval as closely as desired, provided it has enough capacity (e.g., enough hidden neurons).

Informally:

> A neural network with one hidden layer and a non-polynomial activation function (such as tanh, sigmoid, or ReLU) can approximate any continuous function on a compact set arbitrarily well.

This result is known as the **Universal Approximation Theorem**.

---

## What Does “Approximate Any Function” Mean?

It does **not** mean:

- The network represents every function exactly.
- The network automatically learns everything.

It means:

Given:
- A continuous function $f(x)$
- A bounded interval (e.g., $[-2,2]$)
- Enough hidden neurons $K$

There exists a neural network $N(x)$ such that:

$$
\sup_x |f(x) - N(x)| < \varepsilon
$$

for any small $\varepsilon > 0$.

In other words, we can make the approximation error arbitrarily small.

---

# Assignment 1: Universal Function Approximator

## Goal of the Exercise

Compare three neural network architectures and analyze their function approximation capacity.

### Models:

1. $N_0$: One-layer network (linear only)
2. $N_1$: One-layer network with non-linear activation
3. $N_2$: Two-layer network (hidden layer + non-linearity)

They are trained via **gradient descent with weight decay**.

---

## Model 1: Linear Network $N_0$

$$
f(x) = w_0 + w_1 x
$$

This can only represent straight lines.

It is **not universal**.

---

## Model 2: One-layer + Nonlinearity $N_1$

$$
f(x) = \tanh(w_0 + w_1 x)
$$

Now the model produces a nonlinear S-shaped curve.

More flexible than linear, but still limited because it has only one neuron.

---

## Model 3: Two-layer Network $N_2$

$$
f(x) = v_0 + \sum_{k=1}^K v_k \tanh(w_{k0} + w_{k1} x)
$$

This is a linear combination of nonlinear basis functions.

Each hidden neuron creates a nonlinear “bump”.

Combining many of these bumps allows approximation of very complex functions.

This is why the two-layer network is a **universal function approximator**.

---

# Target Functions in the Assignment

### 1. $X_0(x) = \sin(2x)$, for $x \in [-2,2]$

Oscillatory function with repeated bends.

Linear model will fail.

---

### 2. $X_1(x) = e^{-x^2}$, for $x \in [-3,3]$

Smooth bell-shaped curve.

Moderate complexity.

---

### 3. Polynomial:

$$
X_2(x) = -x^5 - 3x^4 + 11x^3 + 27x^2 - 10x - 32
$$

Higher-degree polynomial with multiple curvature changes.

Most complex of the three.

---

# Why These Three Functions?

They increase in complexity:

| Function | Complexity |
|----------|------------|
| $\sin(2x)$ | Oscillatory |
| $e^{-x^2}$ | Smooth bump |
| 5th-degree polynomial | Multiple bends |

This demonstrates the increasing need for network capacity.

---

# Training via Gradient Descent with Weight Decay

The loss function is:

$$
L(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i) - y_i)^2 + \lambda \|\theta\|^2
$$

Two parts:

1. Mean Squared Error (data fit)
2. L2 Regularization (weight decay)

Weight decay penalizes large weights and prevents overfitting.

---

# Big Picture

A two-layer neural network works like:

$$
f(x) = \sum_{k=1}^K v_k \cdot \text{nonlinear feature}_k(x)
$$

Each hidden neuron creates a nonlinear feature.

Combining many nonlinear features allows approximation of arbitrary continuous functions.

That is the core idea behind deep learning.

---

# What This Assignment Teaches

- Why linear models are limited
- Why nonlinear activation functions are essential
- Why hidden layers increase expressive power
- How gradient descent works
- How regularization works
- Why neural networks are powerful function approximators

---

This assignment builds the mathematical foundation for understanding deep learning.