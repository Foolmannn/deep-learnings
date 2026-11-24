# Exponentially Weighted Moving Average (EWMA) in Deep Learning

## 1. Introduction

**Exponentially Weighted Moving Average (EWMA)** is a technique used in Deep Learning to calculate a moving average of values by assigning more importance (weight) to recent observations and gradually decreasing the importance of older observations.

It is commonly used in optimization algorithms such as\:

- Momentum-based Gradient Descent
- RMSprop
- Adam Optimizer

EWMA helps make learning more stable by reducing sudden fluctuations in gradients and parameter updates.

It is particularly useful when gradients vary significantly from one iteration to another.

### Why do we need EWMA?

During neural network training, the optimizer updates weights and biases using gradients calculated through backpropagation.

However, gradients can fluctuate significantly between consecutive iterations.

For example\:

| Iteration | Gradient |
|--------|--------|
| 1 | 10 |
| 2 | 2 |
| 3 | 12 |
| 4 | 3 |
| 5 | 11 |

Notice that the gradients change suddenly.

If we directly use these gradients to update the weights, the model may experience unstable updates, oscillations, or slow convergence.

EWMA solves this problem by maintaining a weighted average of previous gradients.

Instead of considering only the current gradient, it considers both\:

1. The current gradient.
2. The moving average of previous gradients.

---

## 2. Mathematical Formula of EWMA

The mathematical formula for Exponentially Weighted Moving Average is\:

$$
V_t = \beta V_{t-1}+(1-\beta)\theta_t
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $V_t$ | Current exponentially weighted moving average |
| $V_{t-1}$ | Previous moving average |
| $\beta$ | Exponential decay factor |
| $\theta_t$ | Current observed value (e.g., gradient) |
| $1-\beta$ | Weight assigned to the current observation |

**Understanding the formula\:**

The formula combines two components\:

- Previous moving average: $\beta V_{t-1}$
- Current observation: $(1-\beta)\theta_t$

The parameter $\beta$ controls how much importance is given to previous observations.

For example\:

- If $\beta=0.9$, the previous moving average receives 90% weight, while the current observation receives 10%.
- If $\beta=0.5$, the previous moving average receives 50% weight, while the current observation receives 50%.

Therefore, a higher beta value produces a smoother moving average because it gives more importance to historical observations.

## 3. How EWMA Works

EWMA works by continuously updating the moving average whenever a new observation becomes available.

The process follows these steps\:

1. Initialize the moving average $V_0=0$\.
2. Calculate the current gradient using backpropagation.
3. Multiply the previous moving average by $\beta$\.
4. Multiply the current gradient by $1-\beta$\.
5. Add both values to calculate the new moving average.
6. Repeat the process for subsequent iterations.

Shows the arithmetic mean as equal sharing across two to five integer stacks. Learners change the values and watch the same total redistribute into equal stacks, including fractional pieces when the mean is not a whole number.

Initial values:
- observation1: 2
- observation2: 4
- observation3: 7

The following example demonstrates the actual exponential weighting process.

### Step-by-step numerical example

Suppose we have the following gradients\:

$$
G=[10,20,30,40]
$$

Let\:

$$
\beta=0.9
$$

Initialize\:

$$
V_0=0
$$

**Iteration 1\:**

$$
V_1=0.9(0)+0.1(10)
$$

$$
\boxed{V_1=1}
$$

**Iteration 2\:**

$$
V_2=0.9(1)+0.1(20)
$$

$$
V_2=0.9+2
$$

$$
\boxed{V_2=2.9}
$$

**Iteration 3\:**

$$
V_3=0.9(2.9)+0.1(30)
$$

$$
V_3=2.61+3
$$

$$
\boxed{V_3=5.61}
$$

**Iteration 4\:**

$$
V_4=0.9(5.61)+0.1(40)
$$

$$
V_4=5.049+4
$$

$$
\boxed{V_4=9.049}
$$

The resulting moving averages are\:

| Iteration | Gradient | Moving Average |
|--------|--------|--------|
| 1 | 10 | 1.000 |
| 2 | 20 | 2.900 |
| 3 | 30 | 5.610 |
| 4 | 40 | 9.049 |

Notice that the moving average increases gradually rather than directly following the sudden changes in gradients.

This makes the updates smoother.

---

## 4. Why Is It Called Exponentially Weighted Moving Average?

It is called exponentially weighted because the contribution of older observations decreases exponentially over time.

Let's expand the EWMA formula\:

$$
V_t=\beta V_{t-1}+(1-\beta)\theta_t
$$

Substituting the previous moving average\:

$$
V_t=\beta[\beta V_{t-2}+(1-\beta)\theta_{t-1}]
+(1-\beta)\theta_t
$$

Expanding\:

$$
V_t=\beta^2V_{t-2}
+\beta(1-\beta)\theta_{t-1}
+(1-\beta)\theta_t
$$

Continuing this expansion\:

$$
V_t=(1-\beta)\theta_t
+\beta(1-\beta)\theta_{t-1}
+\beta^2(1-\beta)\theta_{t-2}
+\cdots
$$

This shows that\:

- Current observation receives weight $1-\beta$\.
- Previous observation receives weight $\beta(1-\beta)$\.
- Two-iterations-old observation receives weight $\beta^2(1-\beta)$\.
- Three-iterations-old observation receives weight $\beta^3(1-\beta)$\.

Therefore, the importance of older observations decreases exponentially.

**Example of exponential decay of weights**

Suppose $\beta=0.9$\.

| Observation age | Relative weight |
|--------|--------|
| Current | 10% |
| 1 iteration old | 9% |
| 2 iterations old | 8.1% |
| 3 iterations old | 7.29% |
| 4 iterations old | 6.56% |

*Contribution of observations to the moving average when beta = 0.9.*

| Category | Weight |
| --- | ---: |
| Current | 10% |
| 1 old | 9% |
| 2 old | 8.1% |
| 3 old | 7.29% |
| 4 old | 6.561% |

This is why recent observations have greater influence than older observations.

---

## 5. Role of Beta (β) in EWMA

Beta is one of the most important parameters in EWMA.

It determines how quickly the moving average responds to new observations.

Generally, beta is selected between 0 and 1.

### Effect of different beta values

| Beta value | Behavior |
|--------|--------|
| 0 | Considers only the current observation |
| 0.5 | Gives equal importance to the current observation and previous average |
| 0.9 | Produces a smoother moving average |
| 0.99 | Produces a very smooth moving average |
| Close to 1 | Gives more importance to historical observations |

**Low beta value\:**

- Responds quickly to new gradients.
- Less smoothing.
- More sensitive to fluctuations.

**High beta value\:**

- Responds slowly to new gradients.
- More smoothing.
- Less sensitive to sudden changes.

### Relationship between beta and effective window size

A useful approximation is\:

$$
N\approx\frac{1}{1-\beta}
$$

Where $N$ represents the approximate number of observations influencing the moving average.

For example\:

| Beta | Approximate window |
|--------|--------|
| 0.5 | 2 |
| 0.9 | 10 |
| 0.99 | 100 |
| 0.999 | 1000 |

This is an approximation, not a strict cutoff. EWMA does not completely discard older observations.

---

## 6. Bias Correction in EWMA

One important issue with EWMA is that its initial moving average can be biased toward zero.

This happens because we generally initialize the moving average as\:

$$
V_0=0
$$

During the initial iterations, the moving average may be smaller than the actual average of the observed values.

To solve this problem, we use **Bias Correction**\.

The bias-corrected EWMA formula is\:

$$
\boxed{\hat V_t=\frac{V_t}{1-\beta^t}}
$$

Where\:

- $\hat V_t$ \= Bias-corrected moving average.
- $V_t$ \= Original moving average.
- $\beta$ \= Decay factor.
- $t$ \= Current iteration number.

### Numerical example

Suppose\:

$$
\beta=0.9,\quad V_1=1
$$

The bias-corrected moving average is\:

$$
\hat V_1=\frac{1}{1-0.9^1}
$$

$$
=\frac{1}{0.1}
$$

$$
\boxed{\hat V_1=10}
$$

The original moving average was 1, but after bias correction, it becomes 10, which matches the first observed gradient.

**Why is bias correction important?**

- Reduces initialization bias.
- Improves the accuracy of early moving-average estimates.
- Helps optimizers such as Adam perform better during initial training iterations.

---

## 7. Applications of EWMA in Deep Learning

EWMA is an important mathematical concept used in several optimization algorithms.

### A. Momentum in Gradient Descent

Momentum uses EWMA to maintain a moving average of gradients.

Instead of updating weights using only the current gradient, it uses a combination of the current gradient and previous gradient information.

The formula is\:

$$
V_t=\beta V_{t-1}+(1-\beta)G_t
$$

The weight update is\:

$$
W_t=W_{t-1}-\alpha V_t
$$

Where\:

- $W_t$ \= Updated weights.
- $\alpha$ \= Learning rate.
- $G_t$ \= Current gradient.
- $V_t$ \= Moving average of gradients.

**Benefits\:**

1. Reduces oscillations during training.
2. Accelerates convergence in consistent gradient directions.
3. Makes weight updates smoother.
4. Helps the model move through shallow regions of the loss function.
![](../../images/momentum.jpg)
### B. RMSprop Optimizer

RMSprop uses EWMA to calculate the moving average of squared gradients.

Its formula is\:

$$
S_t=\beta S_{t-1}+(1-\beta)G_t^2
$$

Then the weight update is\:

$$
W_t=W_{t-1}-
\frac{\alpha}{\sqrt{S_t}+\epsilon}G_t
$$

Here\:

- $S_t$ \= Moving average of squared gradients.
- $G_t^2$ \= Squared current gradient.
- $\epsilon$ \= Small constant to prevent division by zero.

**Purpose\:**

RMSprop adjusts the learning rate for each parameter based on the magnitude of its recent squared gradients.

### C. Adam Optimizer

Adam (Adaptive Moment Estimation) combines two moving averages\:

1. First moment: EWMA of gradients.
2. Second moment: EWMA of squared gradients.

First moment\:

$$
M_t=\beta_1M_{t-1}+(1-\beta_1)G_t
$$

Second moment\:

$$
V_t=\beta_2V_{t-1}+(1-\beta_2)G_t^2
$$

After bias correction\:

$$
\hat M_t=\frac{M_t}{1-\beta_1^t}
$$

$$
\hat V_t=\frac{V_t}{1-\beta_2^t}
$$

Final weight update\:

$$
W_t=W_{t-1}-
\alpha\frac{\hat M_t}{\sqrt{\hat V_t}+\epsilon}
$$

Common default values\:

$$
\beta_1=0.9,\quad \beta_2=0.999
$$

Adam is widely used because it combines momentum with adaptive learning rates.

---

## 8. Python Implementation of EWMA

Let's implement EWMA using NumPy.

```python
import numpy as np

# Sample gradients
gradients = [10, 20, 30, 40, 50]

# Decay factor
beta = 0.9

# Initialize moving average
V = 0

# Calculate EWMA
for t, gradient in enumerate(gradients, start=1):

    V = beta * V + (1 - beta) * gradient

    # Bias correction
    V_corrected = V / (1 - beta**t)

    print(
        f"Iteration {t}: "
        f"EWMA = {V:.3f}, "
        f"Corrected = {V_corrected:.3f}"
    )
```

**Output\:**

```text
Iteration 1: EWMA = 1.000, Corrected = 10.000
Iteration 2: EWMA = 2.900, Corrected = 15.263
Iteration 3: EWMA = 5.610, Corrected = 20.701
Iteration 4: EWMA = 9.049, Corrected = 26.344
Iteration 5: EWMA = 13.144, Corrected = 32.090
```

The original EWMA is smoother and initially biased toward zero. Bias correction compensates for this initialization effect.

---

## 9. Advantages of EWMA

1. **Reduces fluctuations\:** Smooths sudden variations in gradients.
2. **Faster convergence\:** Helps momentum-based optimization move faster in consistent directions.
3. **Improved stability\:** Reduces unnecessary oscillations during training.
4. **Memory efficient\:** Stores only the previous moving average rather than all historical observations.
5. **Recent information priority\:** Gives greater importance to recent gradients.
6. **Adaptive optimization\:** Helps optimizers such as RMSprop and Adam adjust parameter updates.

## 10. Limitations of EWMA

1. **Parameter selection\:** Choosing an appropriate beta value can be difficult.
2. **Delayed response\:** A very high beta value may respond slowly to sudden changes.
3. **Initialization bias\:** Initial moving averages may be biased toward zero.
4. **Not a complete solution\:** EWMA alone cannot guarantee that a neural network will converge to the global minimum.
5. **Historical influence\:** Old observations continue to influence the moving average, although their contributions become very small.

---

## 11. Difference Between Simple Moving Average and Exponentially Weighted Moving Average

| Basis | Simple Moving Average (SMA) | Exponentially Weighted Moving Average (EWMA) |
|--------|--------|--------|
| Weight assignment | Equal weights | Exponentially decreasing weights |
| Recent observations | Same importance as older observations in the window | Greater importance |
| Older observations | Removed after leaving the window | Gradually lose influence |
| Memory | May require storing a window of values | Requires only the previous average |
| Responsiveness | Depends on window size | Controlled by beta |
| Deep Learning usage | Less common in gradient optimization | Widely used in optimizers |

## 12. Summary

Exponentially Weighted Moving Average (EWMA) is a technique that calculates a moving average by assigning greater importance to recent observations and exponentially decreasing the importance of older observations.

Its main formula is\:

$$
\boxed{V_t=\beta V_{t-1}+(1-\beta)G_t}
$$

The key points to remember are\:

- EWMA smooths gradient fluctuations during neural network training.
- Beta controls the influence of historical observations.
- A higher beta produces a smoother moving average.
- Bias correction removes the initial bias caused by zero initialization.
- Momentum, RMSprop, and Adam use EWMA in different ways.

**Most importantly\:** EWMA is not itself an optimizer. It is a mathematical technique used by optimization algorithms to make neural network training more stable and efficient.

![Mastering Gradient Descent: Optimizing Neural Networks with Precision. | by Omkar Hankare | Medium](https://images.openai.com/static-rsc-4/6cT5hdnhyEzMwrjY0A7F4WZk_udxG1fCpbHSBkR7fD2obMLmqW_66GXnXg3jOv25ptoL_XwveeJ2TsbokLV05DhMfhTx0Lrbsw3-mxm2bH9SZzazADuffmeLYBCM4igcnCUIpK6I9U_6kRsvUvWUzNIxkRTkfWfx4NsabuajL9_FbPmPdnhxtGVYZIAaUCe2?purpose=fullsize)

---
