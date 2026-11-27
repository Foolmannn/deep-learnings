# RMSprop (Root Mean Square Propagation): Detailed Notes

## 1. Where it fits

RMSprop is an **adaptive learning-rate optimizer**. Plain gradient descent uses one learning rate for every parameter. RMSprop gives each parameter its own effective step size, based on how large its recent gradients have been.

It was proposed by Geoffrey Hinton in Lecture 6e of his Coursera course *Neural Networks for Machine Learning* (2012). It was never published in a paper, but it became very popular.

## 2. The problem it solves

**Problem 1: one learning rate doesn't suit all parameters.** In a loss surface shaped like a narrow ravine, the gradient is large along the steep direction and small along the flat one. A single learning rate then either:
- oscillates in the steep direction (if too large), or
- crawls in the flat direction (if too small).

**Problem 2: AdaGrad's learning rate decays to zero.** AdaGrad divides by the square root of the *sum of all past squared gradients*:

$$G_t = \sum_{i=1}^{t} g_i^2$$

This sum only grows, so the effective learning rate shrinks monotonically. Eventually training stalls, which is especially bad for non-convex deep networks.

**RMSprop's fix:** replace the ever-growing sum with an **exponentially decaying moving average** of squared gradients. Old gradients are forgotten, so the learning rate doesn't vanish.

## 3. The algorithm

For parameters $\theta$, loss $L$, learning rate $\eta$, decay rate $\beta$ (also written $\rho$ or $\gamma$), and small constant $\epsilon$:

1. Compute the gradient:
$$g_t = \nabla_\theta L(\theta_{t-1})$$

2. Update the moving average of squared gradients (element-wise):
$$s_t = \beta \, s_{t-1} + (1-\beta)\, g_t^2$$

3. Update the parameters:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t} + \epsilon}\; g_t$$

All operations are **element-wise**, so every parameter has its own $s_t$.

### Pseudocode
```
initialize s = 0
for each step t:
    g = gradient(loss, theta)
    s = beta * s + (1 - beta) * g**2
    theta = theta - lr * g / (sqrt(s) + eps)
```

## 4. Intuition

- $\sqrt{s_t}$ is the **RMS (root mean square) of recent gradients**, hence the name. It is a running estimate of each parameter's typical gradient magnitude.
- Dividing by it **normalizes** the step:
  - Parameters with consistently **large gradients** get their step **shrunk**, which damps oscillation in steep directions.
  - Parameters with consistently **small gradients** get their step **boosted**, which speeds progress in flat directions.
- The update is roughly $\eta \cdot \text{sign-like}(g)$. Each step is of size about $\eta$ regardless of gradient scale, so the optimizer is largely insensitive to gradient magnitude.
- $\epsilon$ prevents division by zero when $s_t \approx 0$.

## 5. Worked numeric example

Let $\beta = 0.9$, $\eta = 0.01$, $\epsilon \approx 0$, and suppose a parameter receives gradient $g_1 = 2$ at step 1, starting from $s_0 = 0$.

- $s_1 = 0.9 \cdot 0 + 0.1 \cdot 4 = 0.4$
- Update $= 0.01 \cdot \dfrac{2}{\sqrt{0.4}} \approx 0.01 \cdot 3.16 = 0.0316$

Plain SGD would have moved $0.01 \cdot 2 = 0.02$. The first step is *larger* than the learning rate because $s$ starts at zero and is **biased low** at the start (see the hyperparameter notes below).

If the gradient stays at 2 forever, $s \to 4$, so $\sqrt{s} \to 2$ and the step settles to $\eta \cdot 2/2 = \eta = 0.01$. With a constant gradient of 200 instead, $s \to 40000$ and the step is still $0.01$. That is the scale-invariance.

## 6. Hyperparameters

| Hyperparameter | Typical value | Notes |
|---|---|---|
| Learning rate $\eta$ | 0.001 (Hinton's suggestion) | Still the most important one to tune. PyTorch's default is 0.01. |
| Decay rate $\beta$ | 0.9 (Keras default), 0.99 (PyTorch `alpha`) | Larger means a longer memory, about $1/(1-\beta)$ steps. 0.9 averages over about 10 steps, 0.99 over about 100. |
| $\epsilon$ | $10^{-8}$ (PyTorch), $10^{-7}$ (Keras) | Numerical stability only. Setting it larger can help if training is unstable. |
| Momentum (optional) | 0 by default | Some implementations let you add momentum on top of RMSprop. |

**Note on bias:** RMSprop initializes $s_0 = 0$, so early estimates of $s_t$ are biased toward zero. That makes early steps too large, as in the example above. Adam fixes this with bias correction, and RMSprop does not.

## 7. Comparison with related optimizers

| Optimizer | Key idea | Weakness |
|---|---|---|
| **SGD** | Same step size for everything | Slow in ravines, sensitive to LR |
| **SGD + Momentum** | Accumulates a velocity (first moment) to smooth and accelerate | Still one global LR |
| **AdaGrad** | Divides by the sum of all past $g^2$ | LR decays to zero |
| **RMSprop** | Divides by the moving average of $g^2$ | No bias correction, no momentum by default |
| **Adam** | Momentum (1st moment) plus RMSprop (2nd moment), plus bias correction | Can generalize slightly worse than SGD in some vision tasks |

A useful way to remember it: **Adam ≈ Momentum + RMSprop + bias correction.**

## 8. Implementation

### From scratch (NumPy)
```python
import numpy as np

def rmsprop(grad_fn, theta, lr=0.001, beta=0.9, eps=1e-8, steps=1000):
    s = np.zeros_like(theta)
    for _ in range(steps):
        g = grad_fn(theta)
        s = beta * s + (1 - beta) * g**2
        theta = theta - lr * g / (np.sqrt(s) + eps)
    return theta
```

### PyTorch
```python
import torch.optim as optim
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)

# training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Keras / TensorFlow
```python
from tensorflow.keras.optimizers import RMSprop
opt = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-7)
```

## 9. Advantages and disadvantages

**Advantages**
- Fixes AdaGrad's vanishing learning rate.
- Handles non-stationary objectives well, which is why it suits RNNs and online learning.
- Works well with mini-batch training.
- Needs less manual LR tuning than SGD.
- Damps oscillation in steep directions.

**Disadvantages**
- Still needs a good global learning rate $\eta$.
- No bias correction, so early steps can be too large.
- No built-in momentum.
- Adaptive methods can converge to solutions that generalize slightly worse than well-tuned SGD+momentum on some tasks.
- Can be unstable if $\epsilon$ is too small and gradients are tiny.

## 10. When to use it

- **Recurrent networks** (RNN/LSTM/GRU), where gradients are noisy and vary in scale.
- **Reinforcement learning** (it was commonly used in early DQN and A3C work).
- **Non-stationary problems** where the loss landscape changes over time.
- As a solid alternative when Adam gives you trouble.

In practice, Adam or AdamW is the default today. Understanding RMSprop matters because it is the **second-moment half of Adam**.

## 11. Common exam and interview questions

1. **Why "RMS"?** It divides by the root of the mean of the squared gradients.
2. **How does it differ from AdaGrad?** It uses an exponential moving average instead of a cumulative sum, so the learning rate doesn't decay to zero.
3. **What does $\beta$ control?** How far back the memory of squared gradients reaches.
4. **Why is $\epsilon$ needed?** To avoid division by zero.
5. **How does Adam extend it?** It adds a momentum term and bias correction.

## 12. Quick summary

> RMSprop keeps a running average of each parameter's squared gradients and divides the gradient by its square root. This normalizes step sizes per parameter: large-gradient directions are damped, small-gradient directions are amplified, and unlike AdaGrad the learning rate never decays to zero.

If you'd like, I can turn this into a saved doc, add a small visualization of RMSprop vs SGD vs Adam on a toy loss surface, or write practice questions on it.