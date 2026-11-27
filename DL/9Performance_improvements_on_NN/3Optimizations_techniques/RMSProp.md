# RMSprop (Root Mean Square Propagation) in Deep Learning

## 1. Introduction

**RMSprop (Root Mean Square Propagation)** is an adaptive learning rate optimization algorithm used to train artificial neural networks and deep learning models.

It was introduced by Geoffrey Hinton in his Coursera lecture on neural networks.

RMSprop is an improvement over traditional Gradient Descent and addresses some of the limitations of the AdaGrad optimization algorithm.

The main objective of RMSprop is to **adaptively adjust the learning rate for each parameter (weight and bias) during the training process.**

In traditional Gradient Descent, the same learning rate is generally applied to all parameters. However, different parameters may require different learning rates.

RMSprop solves this problem by maintaining an exponentially weighted moving average of the squared gradients.

### Key idea

- Parameters with consistently large gradients receive smaller updates.
- Parameters with small gradients receive relatively larger updates.
- The learning rate automatically adapts during training.
- It helps prevent excessively large parameter updates.

(Main concept) RMSprop divides the gradient by the square root of the moving average of squared gradients.

---

## 2. Why Do We Need RMSprop?

Before understanding RMSprop, let's understand the problems with traditional Gradient Descent and AdaGrad.

### A. Problem with traditional Gradient Descent

In traditional Gradient Descent, the weight update formula is\:

$$
w_{t+1}=w_t-\eta g_t
$$

Where\:

- $w_t$ \= Current weight
- $\eta$ \= Learning rate
- $g_t$ \= Gradient at the current step
- $w_{t+1}$ \= Updated weight

The problem is that the same learning rate is used for every parameter.

For example\:

Suppose a neural network has two weights\:

- Weight 1 has a large gradient of 100.
- Weight 2 has a small gradient of 0.01.

If the learning rate is 0.01\:

$$
\Delta w_1=0.01\times100=1
$$

$$
\Delta w_2=0.01\times0.01=0.0001
$$

The first parameter receives a relatively large update, whereas the second receives a very small update.

This can make optimization difficult, particularly when gradients have significantly different magnitudes.

### B. Problem with AdaGrad

AdaGrad attempts to solve this problem by maintaining the sum of squared gradients.

Its accumulated squared gradient is\:

$$
G_t=G_{t-1}+g_t^2
$$

Its update rule is\:

$$
w_{t+1}=w_t-\frac{\eta}{\sqrt{G_t}+\epsilon}g_t
$$

However, AdaGrad has a major limitation.

Since it continuously accumulates squared gradients, the value of $G_t$ keeps increasing.

As a result\:

1. The denominator becomes progressively larger.
2. The effective learning rate keeps decreasing.
3. Eventually, the learning rate may become extremely small.
4. Training can slow down significantly or stop making meaningful progress.

**Solution: RMSprop**

RMSprop solves this problem by replacing the cumulative sum of squared gradients with an exponentially weighted moving average.

Instead of remembering all previous squared gradients equally, it gives more importance to recent gradients.

---

## 3. Working Principle of RMSprop

RMSprop maintains a moving average of squared gradients.

It uses an exponential decay factor to control how much information from previous gradients is retained.

The algorithm works in the following steps\:

1. Calculate the gradient of the loss function.
2. Square the current gradient.
3. Calculate the exponentially weighted moving average of squared gradients.
4. Divide the current gradient by the square root of this moving average.
5. Update the weights and biases.

### Step 1: Calculate the gradient

First, calculate the gradient of the loss function with respect to the weight.

$$
g_t=\frac{\partial L}{\partial w_t}
$$

Here, $L$ represents the loss function.

### Step 2: Calculate the squared gradient

Square the current gradient\:

$$
g_t^2
$$

Squaring makes all values non-negative and allows RMSprop to track the magnitude of gradients.

### Step 3: Calculate the moving average of squared gradients

This is the most important step in RMSprop.

$$
\boxed{v_t=\beta v_{t-1}+(1-\beta)g_t^2}
$$

Where\:

- $v_t$ \= Moving average of squared gradients.
- $v_{t-1}$ \= Previous moving average.
- $\beta$ \= Decay rate.
- $g_t^2$ \= Current squared gradient.

The commonly used value of beta is\:

$$
\beta=0.9
$$

This means that the algorithm retains 90% of the previous moving average and incorporates 10% of the current squared gradient.

### Step 4: Calculate the adaptive gradient

The gradient is divided by the square root of the moving average.

$$
\boxed{\hat{g}_t=\frac{g_t}{\sqrt{v_t}+\epsilon}}
$$

Where\:

- $\hat{g}_t$ \= Normalized or scaled gradient.
- $v_t$ \= Moving average of squared gradients.
- $\epsilon$ \= Small constant used for numerical stability.

A commonly used value is\:

$$
\epsilon=10^{-8}
$$

### Step 5: Update the weights

Finally, update the weights using the adaptive gradient.

$$
\boxed{w_{t+1}=w_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}}
$$

Where\:

- $w_{t+1}$ \= Updated weight.
- $w_t$ \= Current weight.
- $\eta$ \= Learning rate.
- $g_t$ \= Current gradient.
- $v_t$ \= Moving average of squared gradients.

The same procedure is applied to biases.

## 4. Numerical Example of RMSprop

Let's understand RMSprop through a numerical example.

Suppose we have the following initial values\:

| Parameter | Value |
|--------|--------|
| Initial weight ($w_0$\) | 5 |
| Learning rate ($\eta$\) | 0.01 |
| Decay rate ($\beta$\) | 0.9 |
| Epsilon ($\epsilon$\) | 0.000001 |
| Initial moving average ($v_0$\) | 0 |
| Current gradient ($g_1$\) | 4 |

**Step 1: Calculate the squared gradient.**

$$
g_1^2=4^2=16
$$

**Step 2: Calculate the moving average.**

$$
v_1=\beta v_0+(1-\beta)g_1^2
$$

Substituting the values\:

$$
v_1=0.9(0)+0.1(16)
$$

$$
\boxed{v_1=1.6}
$$

**Step 3: Calculate the normalized gradient.**

$$
\hat{g}_1=\frac{g_1}{\sqrt{v_1}+\epsilon}
$$

$$
=\frac{4}{\sqrt{1.6}+0.000001}
$$

$$
\approx 3.1623
$$

**Step 4: Update the weight.**

$$
w_1=w_0-\eta\hat{g}_1
$$

$$
w_1=5-(0.01\times3.1623)
$$

$$
\boxed{w_1=4.968377}
$$

(Final result)

The weight changes from 5 to approximately 4.968377.

This demonstrates how RMSprop scales the gradient before updating the weight.

---

## 5. Understanding Exponential Moving Average in RMSprop

One of the most important concepts behind RMSprop is the **Exponentially Weighted Moving Average (EWMA)**\.

The moving average formula is\:

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

Let's understand what happens when the decay rate is 0.9.

$$
v_t=0.9v_{t-1}+0.1g_t^2
$$

This means\:

- 90% importance is given to the previous moving average.
- 10% importance is given to the current squared gradient.

For example, suppose the squared gradients are\:

$$
g_1^2=100,\quad g_2^2=64,\quad g_3^2=36
$$

Initially\:

$$
v_0=0
$$

**First iteration\:**

$$
v_1=0.9(0)+0.1(100)=10
$$

**Second iteration\:**

$$
v_2=0.9(10)+0.1(64)
$$

$$
v_2=15.4
$$

**Third iteration\:**

$$
v_3=0.9(15.4)+0.1(36)
$$

$$
v_3=17.46
$$

| Iteration | Squared gradient | Moving average |
|--------|--------|--------|
| 1 | 100 | 10 |
| 2 | 64 | 15.4 |
| 3 | 36 | 17.46 |

Notice that the moving average does not simply equal the current squared gradient. It considers both the previous history and the latest gradient.

This makes RMSprop responsive to recent changes while retaining some information about previous gradients.

---

## 6. Role of Important Hyperparameters in RMSprop

RMSprop has several hyperparameters that control its behavior.

### A. Learning Rate ($\eta$\)

The learning rate determines how much the weights are updated in each iteration.

Common values include\:

- 0.1
- 0.01
- 0.001
- 0.0001

A high learning rate can cause unstable training, whereas a very low learning rate can make training slow.

### B. Decay Rate ($\beta$\)

The decay rate determines how much importance is given to previous squared gradients.

Common values\:

| Beta | Behavior |
|--------|--------|
| 0.9 | More responsive to recent gradients |
| 0.99 | Retains a longer history |
| 0.999 | Retains an even longer history |

A higher beta means the moving average changes more slowly.

### C. Epsilon ($\epsilon$\)

Epsilon is a very small constant added to the denominator.

$$
\sqrt{v_t}+\epsilon
$$

Its main purposes are\:

1. Prevent division by zero.
2. Improve numerical stability.
3. Avoid extremely large updates when the moving average is very small.

A commonly used value is $10^{-8}$, although framework implementations can use different defaults.

### D. Momentum

Some RMSprop implementations also support momentum.

Momentum helps accelerate updates in consistent directions and reduce oscillations.

It introduces an additional momentum buffer into the update process.

### E. Centered RMSprop

Standard RMSprop tracks the moving average of squared gradients.

Centered RMSprop additionally tracks the moving average of gradients.

It estimates the gradient variance using\:

$$
v_t^{centered}=E[g^2]-E[g]^2
$$

The update becomes\:

$$
w_{t+1}=w_t-\eta
\frac{g_t}{\sqrt{v_t^{centered}}+\epsilon}
$$

This normalizes gradients using an estimate of their variance rather than their uncentered second moment.

---

## 7. Advantages of RMSprop

1. **Adaptive Learning Rate\:** Automatically adjusts the effective learning rate for individual parameters.
2. **Solves AdaGrad's Learning Rate Problem\:** Uses an exponentially weighted moving average instead of continuously accumulating squared gradients.
3. **Faster Convergence\:** Can improve optimization speed compared with basic Gradient Descent in suitable problems.
4. **Handles Different Gradient Scales\:** Works well when different parameters have gradients with significantly different magnitudes.
5. **Suitable for Non-Stationary Problems\:** Adapts to changing gradient magnitudes during training.
6. **Useful for Recurrent Neural Networks\:** Has historically been used for training RNNs and other sequence models.
7. **Less Manual Learning Rate Adjustment\:** Reduces the need to manually select a separate learning rate for every parameter.

## 8. Disadvantages of RMSprop

1. **Sensitive to Hyperparameters\:** Performance depends on the learning rate, decay rate, and epsilon.
2. **No Guarantee of Global Minimum\:** Like other gradient-based optimizers, it does not guarantee finding the global minimum for non-convex problems.
3. **Additional Memory Requirement\:** Stores a moving average of squared gradients for each parameter.
4. **May Not Always Generalize Better\:** A faster decrease in training loss does not necessarily mean better performance on unseen data.
5. **Can Require Learning Rate Tuning\:** Although it adapts learning rates, the initial learning rate still needs appropriate selection.

---

## 9. RMSprop vs SGD vs AdaGrad vs Adam

These are important optimization algorithms used in deep learning.

| Feature | SGD | AdaGrad | RMSprop | Adam |
|--------|--------|--------|--------|--------|
| Adaptive learning rate | No | Yes | Yes | Yes |
| Tracks squared gradients | No | Cumulative sum | Moving average | Moving average |
| Uses gradient momentum | Optional | No | Optional | Yes |
| Learning rate may decrease continuously | No automatic mechanism | Yes | Not necessarily | Not necessarily |
| Memory requirement | Low | Medium | Medium | Higher |
| Handles sparse gradients | Less effectively in basic form | Good | Can be useful | Good |

### Relationship between these optimizers

- **SGD\:** Updates weights using the current gradient.
- **AdaGrad\:** Adapts learning rates using accumulated squared gradients.
- **RMSprop\:** Improves AdaGrad by using an exponentially weighted moving average of squared gradients.
- **Adam\:** Combines momentum (moving average of gradients) with RMSprop-like scaling (moving average of squared gradients), along with bias correction.

## 10. Implementation of RMSprop in Deep Learning

RMSprop is available directly in popular deep learning frameworks such as TensorFlow/Keras and PyTorch.

### A. RMSprop using TensorFlow / Keras

**Example\:**

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create a neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model using RMSprop
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        epsilon=1e-7
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
```

**Explanation\:**

- `RMSprop()` selects RMSprop as the optimization algorithm.
- `learning_rate=0.001` specifies the initial learning rate.
- `rho=0.9` specifies the decay rate of the moving average.
- `epsilon=1e-7` provides numerical stability.
- `binary_crossentropy` is used as the loss function for binary classification.
- `accuracy` measures classification performance.

**Training the model\:**

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

### B. RMSprop using PyTorch

**Example\:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a neural network
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Define loss function
criterion = nn.BCELoss()

# Define RMSprop optimizer
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.9,
    eps=1e-8
)

# Training loop
for epoch in range(50):

    # Forward propagation
    outputs = model(X_train)

    # Calculate loss
    loss = criterion(outputs, y_train)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights using RMSprop
    optimizer.step()

    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
```

**Explanation\:**

- `model.parameters()` passes all trainable weights and biases to the optimizer.
- `lr=0.001` sets the learning rate.
- `alpha=0.9` controls the moving average decay.
- `zero_grad()` clears previously calculated gradients.
- `loss.backward()` calculates gradients using backpropagation.
- `optimizer.step()` updates the parameters using RMSprop.

**Important\:** Keras and PyTorch use slightly different default hyperparameters. For example, Keras commonly uses `rho=0.9`, while PyTorch uses `alpha=0.99`\. Both represent the decay factor in the moving average.  [Keras](https://keras.io/2/api/optimizers/rmsprop/?utm_source=chatgpt.com) [PyTorch main documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.RMSprop.html?utm_source=chatgpt.com)

---

## 11. Applications of RMSprop in Deep Learning

RMSprop is useful in different deep learning applications.

**1\. Recurrent Neural Networks (RNNs)**

RMSprop has historically been used for training RNNs because it can adapt to changing gradient magnitudes during sequence learning.

**2\. Long Short-Term Memory (LSTM)**

It can be used to train LSTM networks for sequential data, such as time-series forecasting and language processing.

**3\. Natural Language Processing (NLP)**

It can optimize neural networks used in language modeling, text classification, and sequence processing.

**4\. Computer Vision**

It can be used to optimize CNNs for image classification and other image-processing tasks.

**5\. Time-Series Forecasting**

It can help train models that predict future values based on historical data, such as weather and stock-price forecasting.

---

## 12. RMSprop Algorithm: Step-by-Step Summary

The complete RMSprop algorithm can be summarized as follows\:

**Initialize parameters**

Set weights, learning rate, decay rate and $v_0=0$\.

**Calculate gradients**

$\displaystyle g_t=\frac{\partial L}{\partial w_t}$

**Square the gradients**

$\displaystyle g_t^2$

**Update moving average**

$\displaystyle v_t=\beta v_{t-1}+(1-\beta)g_t^2$

**Normalize gradients**

$\displaystyle \hat g_t=\frac{g_t}{\sqrt{v_t}+\epsilon}$

**Update weights and biases**

$\displaystyle w_{t+1}=w_t-\eta\hat g_t$

**Repeat until convergence**

Continue until the stopping condition is reached.

## 13. Conclusion

RMSprop is an adaptive optimization algorithm that improves the training process of neural networks by adjusting the effective learning rate of individual parameters.

Its main contribution is the use of an exponentially weighted moving average of squared gradients instead of continuously accumulating squared gradients as in AdaGrad.

This prevents the effective learning rate from automatically shrinking indefinitely due to accumulated gradient history.

RMSprop is particularly useful for problems where gradient magnitudes change during training and has been widely used in recurrent neural networks and other deep learning architectures.

---

## 14. Important Points to Remember for Exams

1. RMSprop stands for **Root Mean Square Propagation**\.
2. It is an adaptive learning rate optimization algorithm.
3. It was introduced by Geoffrey Hinton.
4. It uses an exponentially weighted moving average of squared gradients.
5. It solves the continuously decreasing learning rate problem of AdaGrad.
6. It adjusts the effective learning rate separately for each parameter.
7. The commonly used decay rate is 0.9, although 0.99 is also common.
8. It is useful for training RNNs, LSTMs and other neural networks.
9. It requires additional memory to store moving averages.
10. Adam combines the ideas of momentum and RMSprop-like adaptive scaling.

**Most important formula\:**

$$
\boxed{v_t=\beta v_{t-1}+(1-\beta)g_t^2}
$$

$$
\boxed{w_{t+1}=w_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}}
$$

**Recommended references\:**

- Keras RMSprop documentation
- PyTorch RMSprop documentation

**My suggestion\:** Before moving to Adam, make sure you understand the difference between Momentum and RMSprop\:

- Momentum uses an average of **gradients** to improve the direction of updates.
- RMSprop uses an average of **squared gradients** to adapt the size of updates.
- Adam combines both approaches.

---
