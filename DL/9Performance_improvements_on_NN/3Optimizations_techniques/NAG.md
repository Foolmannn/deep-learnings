# Nesterov Accelerated Gradient (NAG) in Deep Learning

## 1. Introduction

**Nesterov Accelerated Gradient (NAG)** is an optimization algorithm used to train neural networks by updating their weights and biases in a way that accelerates convergence and improves learning efficiency.

NAG is an improved version of the **Momentum-based Gradient Descent** algorithm.

Both Momentum and NAG use the concept of momentum to accelerate gradient-based optimization. However, they differ in how they calculate gradients.

- **Momentum\:** Calculates the gradient at the current position of the parameters.
- **NAG\:** Calculates the gradient at an estimated future position of the parameters.

This small difference allows NAG to anticipate where the parameters are heading and make more informed updates.

NAG is particularly useful for\:

- Accelerating neural network training.
- Reducing oscillations during optimization.
- Improving convergence toward a minimum.
- Reducing overshooting in optimization.
- Handling complex loss functions.

## 2. Why Do We Need Nesterov Accelerated Gradient?

Before understanding NAG, let's understand a limitation of traditional Momentum Gradient Descent.

### Problem with Momentum

In Momentum Gradient Descent, the optimizer uses the previous update direction to determine the next update.

Suppose a neural network is minimizing a loss function.

The optimizer continuously moves toward the minimum using the accumulated momentum from previous gradients.

However, momentum can sometimes cause a problem\:

**It may move too far in a particular direction because it does not consider how the gradient might change at the next position.**

This can cause overshooting.
![](../../images/optimizers/nag_gd_momentum.jpg)
For example\:

1. The optimizer moves toward the minimum.
2. Momentum accelerates the movement.
3. The optimizer approaches the minimum with high velocity.
4. It may cross the minimum and move to the other side.
5. It then needs additional updates to correct its direction.

NAG attempts to solve this problem by looking ahead before calculating the gradient.

Instead of calculating the gradient only at the current position, it first estimates the future position and calculates the gradient there.

This helps the optimizer adjust its direction before making the actual update.

## 3. Working Principle of NAG

NAG works using two main ideas\:

1. **Look-ahead\:** Estimate where the parameters will move based on the previous momentum.
2. **Gradient correction\:** Calculate the gradient at the estimated future position and use it to correct the update.

Let's understand the process step by step.

### Step 1: Calculate the look-ahead position

First, the optimizer uses the previous velocity to estimate the future position of the parameters.

Instead of calculating the gradient at the current weights, it temporarily moves in the direction suggested by momentum.

### Step 2: Calculate the gradient at the look-ahead position

The loss function's gradient is calculated at the estimated future position.

This gradient provides information about the direction in which the loss function is increasing at that future position.

### Step 3: Update the velocity

The optimizer combines the previous velocity with the newly calculated gradient.

This helps maintain momentum while correcting the direction of movement.

### Step 4: Update the parameters

Finally, the optimizer updates the weights using the corrected velocity.
![](../../images/optimizers/nag.jpg)

In simple terms\:

**Momentum asks\:** Where am I moving, and how fast?

**NAG asks\:** Where am I going, and what will the gradient look like when I get there?

---

## 4. Mathematical Formulation of NAG

Let's understand the mathematical equations used in Nesterov Accelerated Gradient.

NAG uses three important variables\:

| Symbol | Meaning |
|--------|--------|
| $\theta_t$ | Current model parameters (weights and biases) |
| $v_t$ | Current velocity |
| $\eta$ | Learning rate |
| $\beta$ | Momentum coefficient |
| $\nabla J(\theta)$ | Gradient of the loss function |

Here, $J(\theta)$ represents the loss function.

### Equation 1: Look-ahead position

$$
\theta_{\text{lookahead}} =
\theta_t+\beta v_t
$$

This equation estimates the future position of the parameters using the previous velocity.

### Equation 2: Calculate the gradient

$$
g_t=\nabla J(\theta_t+\beta v_t)
$$

The gradient is calculated at the look-ahead position rather than the current position.

### Equation 3: Update velocity

$$
v_{t+1}=\beta v_t-\eta g_t
$$

Where\:

- $\beta v_t$ represents the contribution of previous momentum.
- $\eta$ represents the learning rate.
- $g_t$ represents the gradient at the look-ahead position.

### Equation 4: Update parameters

$$
\boxed{\theta_{t+1}=\theta_t+v_{t+1}}
$$

The model parameters are updated using the newly calculated velocity.

**Important\:** Some implementations use a different velocity convention, such as storing the negative gradient direction. The equations may look slightly different, but the underlying NAG principle remains the same.

---

## 5. Numerical Example of NAG

Let's understand NAG through a simple numerical example.

Suppose we have the following values\:

- Initial weight: $\theta_0=10$
- Initial velocity: $v_0=0$
- Learning rate: $\eta=0.1$
- Momentum coefficient: $\beta=0.9$
- Loss function\:

$$
J(\theta)=\theta^2
$$

The gradient of the loss function is\:

$$
\nabla J(\theta)=2\theta
$$

We will perform two iterations.

### Iteration 1

**Step 1: Calculate look-ahead position**

$$
\theta_{\text{lookahead}}=10+0.9(0)
$$

$$
=10
$$

**Step 2: Calculate gradient**

$$
g_0=2(10)=20
$$

**Step 3: Update velocity**

$$
v_1=0.9(0)-0.1(20)
$$

$$
v_1=-2
$$

**Step 4: Update weight**

$$
\theta_1=10+(-2)
$$

$$
\boxed{\theta_1=8}
$$

### Iteration 2

Now the current weight is 8, and the previous velocity is -2.

**Step 1: Calculate look-ahead position**

$$
\theta_{\text{lookahead}}=8+0.9(-2)
$$

$$
=6.2
$$

**Step 2: Calculate gradient at look-ahead position**

$$
g_1=2(6.2)
$$

$$
=12.4
$$

**Step 3: Update velocity**

$$
v_2=0.9(-2)-0.1(12.4)
$$

$$
=-1.8-1.24
$$

$$
v_2=-3.04
$$

**Step 4: Update weight**

$$
\theta_2=8+(-3.04)
$$

$$
\boxed{\theta_2=4.96}
$$

### Summary of iterations

| Iteration | Current weight | Look-ahead position | Gradient | New velocity | Updated weight |
|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 20 | \-2 | 8 |
| 2 | 8 | 6.2 | 12.4 | \-3.04 | 4.96 |

We can observe that NAG uses the previous velocity to estimate the next position and calculates the gradient at that estimated position.

This helps it adjust its movement more intelligently.

---

## 6. Difference Between Momentum and NAG

Both algorithms use momentum, but their gradient calculation methods are different.
![](../../images/optimizers/nag_vs_momentum.jpg)


| Basis | Momentum | Nesterov Accelerated Gradient |
|--------|--------|--------|
| Full name | Momentum Gradient Descent | Nesterov Accelerated Gradient |
| Gradient calculation | At current position | At look-ahead position |
| Approach | Uses previous velocity to accelerate movement | Uses previous velocity to anticipate future movement |
| Direction correction | Corrects using current gradient | Corrects using future-position gradient |
| Overshooting | May experience more overshooting | Can reduce overshooting |
| Convergence | Often faster than basic SGD | Can converge more efficiently in suitable problems |
| Complexity | Relatively simple | Slightly more complex |
| Main advantage | Accelerates gradient descent | Accelerates while anticipating future gradients |

### Important mathematical difference

**Momentum\:**

$$
v_{t+1}=\beta v_t-\eta\nabla J(\theta_t)
$$

$$
\theta_{t+1}=\theta_t+v_{t+1}
$$

**NAG\:**

$$
v_{t+1}=\beta v_t-
\eta\nabla J(\theta_t+\beta v_t)
$$

$$
\theta_{t+1}=\theta_t+v_{t+1}
$$

The main difference is the position at which the gradient is calculated.

---

## 7. Implementation of NAG in Deep Learning

NAG is supported by popular deep learning frameworks such as TensorFlow/Keras and PyTorch.

### A. Implementation using TensorFlow/Keras

In Keras, NAG can be implemented using the SGD optimizer with `momentum` and `nesterov` parameters.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile using Nesterov Accelerated Gradient
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
```

**Explanation\:**

- `learning_rate=0.01`\: Controls the size of each parameter update.
- `momentum=0.9`\: Determines how much previous velocity contributes to the current update.
- `nesterov=True`\: Enables Nesterov Accelerated Gradient.
- `binary_crossentropy`\: Loss function for binary classification.
- `metrics=['accuracy']`\: Measures classification accuracy.

To train the model\:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

### B. Implementation using PyTorch

PyTorch also supports NAG through the SGD optimizer.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create a neural network
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

# Define NAG optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

# Training example
for epoch in range(50):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

**Explanation\:**

- `momentum=0.9` enables momentum.
- `nesterov=True` enables Nesterov acceleration.
- `loss.backward()` calculates gradients using backpropagation.
- `optimizer.step()` updates the model parameters using NAG.

Note: For this example, `X_train` and `y_train` should be PyTorch tensors, with target values having the same shape as the model outputs.

---

## 8. Advantages of Nesterov Accelerated Gradient

1. **Faster Convergence\:** NAG can accelerate the optimization process by using previous momentum and anticipating future gradients.
2. **Reduced Overshooting\:** It can reduce excessive movement around the minimum by correcting the update direction using the look-ahead gradient.
3. **Improved Direction\:** It calculates gradients at an estimated future position, allowing more informed parameter updates.
4. **Reduced Oscillations\:** It can reduce unnecessary oscillations, particularly in directions where the loss function has steep curvature.
5. **Efficient Training\:** It can improve training efficiency, especially for problems where momentum-based optimization is beneficial.
6. **Easy Implementation\:** It is available directly in popular deep learning frameworks such as TensorFlow and PyTorch.

## 9. Disadvantages of Nesterov Accelerated Gradient

1. **Additional Computation\:** It requires gradient evaluation at a look-ahead position, making its implementation slightly more involved than standard momentum.
2. **Hyperparameter Sensitivity\:** Its performance depends on selecting suitable learning rates and momentum coefficients.
3. **Not Always Faster\:** NAG does not guarantee better performance for every neural network or optimization problem.
4. **Still Sensitive to Learning Rate\:** A very high learning rate can still cause instability or divergence.
5. **May Not Be Ideal for Every Problem\:** Adaptive optimizers such as Adam may perform better in certain problems, particularly when gradients have varying scales.

## 10. Applications of NAG in Deep Learning

NAG can be used in several deep learning applications\:

1. **Image Classification\:** Training CNN models for image recognition.
2. **Natural Language Processing\:** Optimizing neural networks used in language processing.
3. **Object Detection\:** Training deep learning models for object identification.
4. **Speech Recognition\:** Optimizing neural networks used in speech processing.
5. **Computer Vision\:** Improving the training of various computer vision architectures.
6. **Regression Problems\:** Optimizing neural networks used for numerical prediction.

---

## 11. NAG vs SGD vs Momentum vs Adam

These are important optimization algorithms in deep learning.

| Feature | SGD | Momentum | NAG | Adam |
|--------|--------|--------|--------|--------|
| Basic gradient descent | Yes | Yes | Yes | Yes |
| Uses momentum | No | Yes | Yes | Yes |
| Look-ahead gradient | No | No | Yes | No |
| Adaptive learning rate | No | No | No | Yes |
| Reduces oscillations | Limited | Yes | Yes | Yes |
| Computational complexity | Low | Low | Low to moderate | Moderate |
| Main benefit | Simplicity | Faster movement | Anticipatory correction | Adaptive updates |

## 12. Conclusion

Nesterov Accelerated Gradient (NAG) is an improved momentum-based optimization algorithm used to train deep learning models efficiently.

Unlike traditional Momentum Gradient Descent, which calculates gradients at the current parameter position, NAG calculates gradients at an estimated future position called the look-ahead position.

This allows the optimizer to anticipate changes in the gradient and correct its direction before updating the model parameters.

As a result, NAG can accelerate convergence, reduce oscillations, and improve optimization efficiency.

(Key concept to remember)

**NAG = Momentum + Look-ahead Gradient**

---

## 13. Quick Revision

| Question | Answer |
|--------|--------|
| What is NAG? | An improved momentum-based optimization algorithm. |
| Who introduced NAG? | Yurii Nesterov. |
| What is its main idea? | Looking ahead before calculating the gradient. |
| Where is the gradient calculated? | At the estimated future position. |
| What does momentum coefficient control? | Contribution of previous velocity. |
| What is the typical momentum value? | 0.9 (a common starting point). |
| What is the main advantage? | Faster and more informed optimization. |
| How is it enabled in Keras? | `nesterov=True` |
| How is it enabled in PyTorch? | `nesterov=True` |
