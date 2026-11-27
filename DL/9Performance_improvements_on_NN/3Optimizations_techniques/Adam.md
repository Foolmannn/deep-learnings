
# Adam Optimizer (Adaptive Moment Estimation) in Deep Learning

## 1. Introduction

**Adam (Adaptive Moment Estimation)** is one of the most popular and widely used optimization algorithms in Deep Learning. It is used to update the weights and biases of a neural network during training so that the model can minimize its loss function.

Adam combines the advantages of two optimization techniques\:

1. **Momentum\:** Uses the moving average of past gradients to accelerate learning and reduce oscillations.
2. **RMSprop\:** Uses the moving average of squared gradients to adapt the learning rate for each parameter individually.

By combining these two techniques, Adam provides an efficient and adaptive approach to optimizing neural networks.

Adam was introduced in 2014 by Diederik P. Kingma and Jimmy Ba in their research paper *Adam: A Method for Stochastic Optimization*\.

**In simple words\:** Adam remembers the direction in which the model has been learning and how large the previous gradients were. It uses this information to decide how much each weight should be updated.

## 2. Why Do We Need the Adam Optimizer?

Before understanding Adam, let's understand some limitations of traditional optimization algorithms.

### A. Problems with Gradient Descent

Gradient Descent updates the weights using the following formula\:

$$
W_{new}=W_{old}-\eta \frac{\partial L}{\partial W}
$$

Where\:

- $W$ \= Model weight
- $\eta$ \= Learning rate
- $L$ \= Loss function
- $\frac{\partial L}{\partial W}$ \= Gradient of the loss with respect to the weight

Some limitations include\:

- It uses the same learning rate for all parameters.
- Choosing an appropriate learning rate can be difficult.
- It may oscillate in certain directions.
- It can converge slowly in complex optimization problems.

### B. Problems with Momentum

Momentum improves Gradient Descent by considering previous gradients.

However\:

- It generally uses a fixed learning rate.
- It does not independently adapt the learning rate for each parameter.
- Choosing suitable momentum and learning-rate values is still important.

### C. Problems with RMSprop

RMSprop adapts the learning rate using the moving average of squared gradients.

However\:

- It does not explicitly maintain the moving average of gradients in the same way as Momentum.
- It does not combine both first-moment and second-moment estimates.

### Solution: Adam

Adam combines both approaches\:

- Momentum helps determine the direction of weight updates.
- RMSprop helps determine the appropriate update size for individual parameters.

As a result, Adam can achieve faster and more stable training in many deep-learning problems.

---

## 3. Working Principle of Adam Optimizer

Adam maintains two important moving averages during training.

**Adam Optimizer**

**First Moment Estimate**

Moving average of gradients

$m_t$

Momentum – Direction of updates

**Second Moment Estimate**

Moving average of squared gradients

$v_t$

RMSprop – Adaptive update size

**Bias Correction**

$\hat m_t,\hat v_t$

**Weight Update**

Update model parameters using corrected estimates

Let's understand each step in detail.

### Step 1: Calculate the Gradient

First, the neural network performs forward propagation and calculates the loss.

Then, backpropagation calculates the gradient of the loss with respect to each parameter.

$$
g_t=\frac{\partial L}{\partial W_t}
$$

Where\:

- $g_t$ \= Current gradient
- $L$ \= Loss function
- $W_t$ \= Model weight at iteration $t$

The gradient indicates how the loss changes when a particular weight changes.

### Step 2: Calculate the First Moment Estimate (Momentum)

Adam calculates the exponentially weighted moving average of the gradients.

It is called the **first moment estimate**\.

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

Where\:

- $m_t$ \= First moment estimate at current iteration.
- $m_{t-1}$ \= Previous first moment estimate.
- $g_t$ \= Current gradient.
- $\beta_1$ \= Exponential decay rate for the first moment.

The commonly used default value is\:

$$
\boxed{\beta_1=0.9}
$$

**How does it work?**

Suppose the gradients from three consecutive iterations are\:

$$
g_1=10,\quad g_2=8,\quad g_3=6
$$

Assume\:

$$
m_0=0,\quad \beta_1=0.9
$$

At the first iteration\:

$$
m_1=0.9(0)+0.1(10)
$$

$$
m_1=1
$$

At the second iteration\:

$$
m_2=0.9(1)+0.1(8)
$$

$$
m_2=1.7
$$

At the third iteration\:

$$
m_3=0.9(1.7)+0.1(6)
$$

$$
m_3=2.13
$$

The first moment estimate maintains information about previous gradients.

**Importance\:**

- Reduces unnecessary oscillations.
- Provides smoother gradient updates.
- Helps maintain a consistent update direction.
- Can accelerate convergence.

### Step 3: Calculate the Second Moment Estimate (RMSprop)

Adam calculates the exponentially weighted moving average of squared gradients.

It is called the **second moment estimate**\.

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Where\:

- $v_t$ \= Second moment estimate.
- $v_{t-1}$ \= Previous second moment estimate.
- $g_t^2$ \= Square of the current gradient.
- $\beta_2$ \= Exponential decay rate for the second moment.

The commonly used default value is\:

$$
\boxed{\beta_2=0.999}
$$

**How does it work?**

Using the same gradients\:

$$
g_1=10,\quad g_2=8,\quad g_3=6
$$

Assume\:

$$
v_0=0,\quad \beta_2=0.999
$$

At the first iteration\:

$$
v_1=0.999(0)+0.001(10^2)
$$

$$
v_1=0.1
$$

At the second iteration\:

$$
v_2=0.999(0.1)+0.001(8^2)
$$

$$
v_2=0.1639
$$

At the third iteration\:

$$
v_3=0.999(0.1639)+0.001(6^2)
$$

$$
v_3=0.1997361
$$

The second moment estimate tracks the magnitude of recent gradients.

**Importance\:**

- Adapts the learning rate for individual parameters.
- Prevents excessively large updates when gradients are large.
- Helps stabilize training.
- Reduces the impact of large gradient variations.

### Step 4: Bias Correction

This is one of the most important features of Adam.

Both moving averages $m_t$ and $v_t$ are initialized with zero\:

$$
m_0=0,\qquad v_0=0
$$

Because of this zero initialization, their initial estimates are biased toward zero, particularly during the early iterations.

To solve this problem, Adam applies **bias correction**\.

There are two bias-correction equations.

**First moment bias correction\:**

$$
\boxed{\hat m_t=\frac{m_t}{1-\beta_1^t}}
$$

**Second moment bias correction\:**

$$
\boxed{\hat v_t=\frac{v_t}{1-\beta_2^t}}
$$

Where\:

- $\hat m_t$ \= Bias-corrected first moment.
- $\hat v_t$ \= Bias-corrected second moment.
- $t$ \= Current iteration.
- $\beta_1,\beta_2$ \= Decay rates.

**Example\:**

Consider the first moment at iteration 1.

We already calculated\:

$$
m_1=1
$$

Given\:

$$
\beta_1=0.9,\qquad t=1
$$

Applying bias correction\:

$$
\hat m_1=\frac{1}{1-0.9^1}
$$

$$
\hat m_1=\frac{1}{0.1}
$$

$$
\boxed{\hat m_1=10}
$$

Similarly, for the second moment\:

$$
v_1=0.1
$$

Given\:

$$
\beta_2=0.999
$$

$$
\hat v_1=\frac{0.1}{1-0.999^1}
$$

$$
\hat v_1=\frac{0.1}{0.001}
$$

$$
\boxed{\hat v_1=100}
$$

Therefore, the corrected estimates are\:

$$
\hat m_1=10,\qquad \hat v_1=100
$$

**Why is bias correction important?**

- Corrects the initial bias caused by zero initialization.
- Provides more accurate moment estimates during early training.
- Helps Adam make appropriate updates from the beginning of training.

### Step 5: Update the Weights

After calculating the two bias-corrected estimates, Adam updates the neural network weights.

The final weight update equation is\:

$$
\boxed{
W_{t+1}=W_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
$$

Where\:

- $W_{t+1}$ \= Updated weight.
- $W_t$ \= Previous weight.
- $\eta$ \= Learning rate.
- $\hat m_t$ \= Corrected first moment.
- $\hat v_t$ \= Corrected second moment.
- $\epsilon$ \= Small constant for numerical stability.

The commonly used original Adam defaults are\:

$$
\eta=0.001
$$

$$
\beta_1=0.9
$$

$$
\beta_2=0.999
$$

$$
\epsilon=10^{-8}
$$

**Example\:**

Suppose\:

$$
W_t=5
$$

$$
\eta=0.001,\quad \hat m_t=10
$$

$$
\hat v_t=100,\quad \epsilon=10^{-8}
$$

Substituting into the equation\:

$$
W_{t+1}=5-0.001
\left(\frac{10}{\sqrt{100}+10^{-8}}\right)
$$

$$
W_{t+1}=5-0.001
\left(\frac{10}{10+10^{-8}}\right)
$$

Approximately\:

$$
W_{t+1}=5-0.001
$$

$$
\boxed{W_{t+1}=4.999}
$$

Therefore, the weight changes from 5 to approximately 4.999.

This process is repeated for all weights and biases during neural network training.

---

## 4. Complete Mathematical Formula of Adam

Let's combine all the steps into one place.

| Step | Operation | Formula |
|--------|--------|--------|
| 1 | Calculate gradient | $g_t=\nabla_W L$ |
| 2 | First moment | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ |
| 3 | Second moment | $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ |
| 4 | First moment correction | $\hat m_t=\frac{m_t}{1-\beta_1^t}$ |
| 5 | Second moment correction | $\hat v_t=\frac{v_t}{1-\beta_2^t}$ |
| 6 | Update weights | $W_{t+1}=W_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ |

**Complete algorithm\:**

1. Initialize the first moment $m_0=0$\.
2. Initialize the second moment $v_0=0$\.
3. Set the iteration counter $t=0$\.
4. Calculate the loss using forward propagation.
5. Calculate gradients using backpropagation.
6. Update the first and second moment estimates.
7. Apply bias correction to both estimates.
8. Update weights and biases.
9. Repeat until the model converges or reaches the specified number of epochs.

---

## 5. Understanding Adam Through a Real-Life Example

Imagine you are walking down a mountain to reach the lowest point.

Your objective is to reach the bottom as quickly and safely as possible.

Consider three situations\:

**1\. Momentum (First Moment)**

You remember the direction in which you were moving previously.

If you have consistently moved downhill, you continue moving in that direction instead of changing direction unnecessarily.

This helps reduce oscillations and maintain movement.

**2\. RMSprop (Second Moment)**

You observe how steep the mountain is.

- If the slope is very steep, you take smaller steps.
- If the slope is relatively gentle, you can take larger steps.

This helps adjust the size of your movements.

**3\. Adam (Combination)**

Adam combines both ideas.

- It remembers the direction of previous movements.
- It adjusts the size of each movement according to gradient magnitude.
- It corrects the initial estimates to make the updates more appropriate.

Therefore, Adam can move toward the lowest point efficiently.
![Optimisation Techniques II · Deep Learning](../../images/optimizers/rms_adam_sgd.jpg)

---

## 6. Hyperparameters of Adam Optimizer

Adam has four important hyperparameters.

| Hyperparameter | Default value | Purpose |
|--------|--------|--------|
| Learning rate ($\eta$\) | 0.001 | Controls the overall update size |
| Beta 1 ($\beta_1$\) | 0.9 | Controls the decay rate of gradient averages |
| Beta 2 ($\beta_2$\) | 0.999 | Controls the decay rate of squared-gradient averages |
| Epsilon ($\epsilon$\) | $10^{-8}$ | Provides numerical stability in the denominator |

*Note: These are the commonly cited original-paper defaults. Framework defaults can differ; for example, Keras uses an epsilon of $10^{-7}$ in its documented configuration.*  [cl.cam.ac.uk](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/papers/kingma_ICLR_2015.pdf?utm_source=chatgpt.com) [Keras](https://keras.io/2/api/optimizers/adam/?utm_source=chatgpt.com)

### A. Learning Rate

The learning rate controls how much the weights change during each update.

- Too high: Training may become unstable or overshoot the minimum.
- Too low: Training may become very slow.
- Appropriate: Helps the model converge efficiently.

### B. Beta 1

$$
\beta_1=0.9
$$

Controls the influence of previous gradients.

A higher value gives more importance to historical gradients, producing smoother updates.

### C. Beta 2

$$
\beta_2=0.999
$$

Controls the influence of previous squared gradients.

It helps Adam adapt the update size according to the magnitude of gradients.

### D. Epsilon

$$
\epsilon=10^{-8}
$$

A very small positive value added to the denominator.

Its purpose is to prevent division by zero and numerical instability.

---

## 7. Implementation of Adam Optimizer in Deep Learning

Adam is supported by popular deep-learning frameworks such as TensorFlow, Keras, and PyTorch.

### A. Adam using TensorFlow / Keras

Let's implement Adam in a simple neural network.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model using Adam
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
```

**Explanation\:**

- `learning_rate=0.001`\: Controls the learning speed.
- `beta_1=0.9`\: Controls the first moment estimate.
- `beta_2=0.999`\: Controls the second moment estimate.
- `epsilon=1e-7`\: Prevents numerical instability.
- `binary_crossentropy`\: Loss function for binary classification.

Training the model\:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

Here, Adam automatically updates the weights and biases during training.

### B. Adam using PyTorch

PyTorch also provides Adam through `torch.optim.Adam()`\.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define neural network
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x

# Create model
model = NeuralNetwork()

# Define loss function
criterion = nn.BCELoss()

# Define Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
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

**Important PyTorch operations\:**

| Operation | Purpose |
|--------|--------|
| `optimizer.zero_grad()` | Clears previous gradients |
| `loss.backward()` | Calculates gradients using backpropagation |
| `optimizer.step()` | Updates model parameters using Adam |

---

## 8. Advantages of Adam Optimizer

Adam provides several advantages that make it popular in Deep Learning.

**1\. Adaptive Learning Rate**

Adam automatically adjusts the effective update size for individual parameters based on their gradient history.

**2\. Faster Convergence**

By combining Momentum and RMSprop, Adam often converges faster than basic Gradient Descent.

**3\. Less Hyperparameter Tuning**

Adam generally performs well with its commonly used default hyperparameters.

**4\. Handles Noisy Gradients**

It uses moving averages to smooth gradient information, making it suitable for stochastic optimization.

**5\. Suitable for Large Models**

Adam works well with models containing millions of parameters.

**6\. Handles Sparse Gradients**

It is useful in problems where many parameters receive infrequent or sparse updates.

**7\. Computationally Efficient**

It requires relatively little additional computation for maintaining its moving averages.

## 9. Disadvantages of Adam Optimizer

Despite its advantages, Adam has some limitations.

**1\. Higher Memory Requirements**

Adam stores two additional values for every parameter\:

- First moment estimate ($m_t$\).
- Second moment estimate ($v_t$\).

Therefore, it requires more memory than basic SGD.

**2\. May Generalize Poorly in Some Problems**

In certain tasks, especially some computer vision applications, SGD with Momentum may achieve better final generalization performance than Adam.

**3\. Sensitive to Learning Rate**

Although Adam is relatively robust, an inappropriate learning rate can still cause unstable or slow training.

**4\. Convergence Issues**

The original Adam algorithm can fail to converge in some optimization problems. This motivated improvements such as AMSGrad.

**5\. Additional Computational Overhead**

Maintaining two moving averages and applying bias correction requires additional computations compared with basic SGD.

---

## 10. Difference Between SGD, Momentum, RMSprop and Adam

| Feature | SGD | Momentum | RMSprop | Adam |
|--------|--------|--------|--------|--------|
| Uses current gradient | Yes | Yes | Yes | Yes |
| Uses past gradients | No | Yes | Squared gradients | Yes |
| Adaptive learning rate | No | No | Yes | Yes |
| First moment | No | Yes | No | Yes |
| Second moment | No | No | Yes | Yes |
| Bias correction | No | No | Usually no | Yes |
| Memory requirement | Low | Low | Moderate | Higher |
| Convergence | Can be slow | Often faster | Often faster | Often fast |

### Important distinction

- **Momentum\:** Remembers the direction of previous gradients.
- **RMSprop\:** Adjusts update sizes using squared gradients.
- **Adam\:** Combines both techniques and applies bias correction.

---

## 11. Adam vs AdamW

AdamW is an improved variant of Adam that handles weight decay differently.

In standard Adam, adding L2 regularization to the loss makes the regularization gradient interact with Adam's adaptive scaling.

AdamW uses **decoupled weight decay**, applying weight decay separately from the adaptive gradient update.

| Adam | AdamW |
|--------|--------|
| Standard Adam optimization | Adam with decoupled weight decay |
| L2 regularization can interact with adaptive scaling | Weight decay is applied separately |
| Common general-purpose optimizer | Often preferred when explicit weight decay is needed |
| May require careful regularization configuration | Provides clearer weight-decay behavior |

AdamW is widely used in modern deep-learning models, including Transformer architectures.

## 12. Applications of Adam Optimizer

Adam is used in many deep-learning applications\:

1. **Computer Vision\:** Image classification, object detection and image segmentation.
2. **Natural Language Processing\:** Text classification, language modeling and machine translation.
3. **Recurrent Neural Networks\:** Training LSTM and GRU networks.
4. **Convolutional Neural Networks\:** Training CNN architectures.
5. **Generative Models\:** Training GANs and other generative networks.
6. **Transformers\:** Training attention-based neural networks.

## 13. When Should We Use Adam?

Adam is a good starting choice when\:

- You are training deep neural networks.
- Your dataset contains noisy or sparse gradients.
- You want faster initial training.
- You are working with CNNs, RNNs, LSTMs or Transformers.
- You do not know which optimizer to start with.

However, Adam is not always the best optimizer. It is worth comparing it with SGD with Momentum when final generalization performance is important.

## 14. Summary: How Adam Works

## Adam optimization workflow

1. **Forward Propagation**
   
   Calculate the model's predictions.
2. **Calculate Loss**
   
   Measure the difference between actual and predicted values.
3. **Backpropagation**
   
   Calculate gradients for weights and biases.
4. **First Moment**
   
   Calculate the moving average of gradients (Momentum).
5. **Second Moment**
   
   Calculate the moving average of squared gradients (RMSprop).
6. **Bias Correction**
   
   Correct both moment estimates.
7. **Update Parameters**
   
   Update weights and biases using Adam's update equation.
8. **Repeat**
   
   Continue training until the stopping condition is reached.

### Important points to remember for exams

1. Adam stands for **Adaptive Moment Estimation**\.
2. It was introduced by Diederik P. Kingma and Jimmy Ba.
3. It combines Momentum and RMSprop.
4. It maintains first and second moment estimates.
5. First moment represents the moving average of gradients.
6. Second moment represents the moving average of squared gradients.
7. It uses bias correction to compensate for zero initialization.
8. Its commonly used learning rate is 0.001.
9. It is computationally efficient and works well with large models.
10. It requires additional memory to store two moment estimates.

**Conclusion\:** Adam is an adaptive optimization algorithm that combines the advantages of Momentum and RMSprop. It uses the moving averages of gradients and squared gradients to determine the direction and magnitude of parameter updates. Its adaptive learning mechanism and bias correction make it one of the most widely used optimizers in Deep Learning.

## 15. References for Further Study

- Original research paper: Adam: A Method for Stochastic Optimization (Kingma & Ba)\.
- TensorFlow/Keras Adam documentation\.
- Original Adam algorithm and mathematical derivation\.

---




