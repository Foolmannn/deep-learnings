# Gradient Descent in Deep Learning (Detailed Study Notes)

Gradient Descent is one of the most fundamental optimization algorithms in Machine Learning and Deep Learning. It is the algorithm that allows a neural network to **learn from its mistakes by continuously adjusting its weights and biases to reduce prediction errors.**

To understand Gradient Descent properly, we will connect it with the concepts you have already studied\:

1. Perceptron and its learning process
2. Multi-Layer Perceptron (MLP)
3. Forward Propagation
4. Loss Functions
5. Backpropagation
6. Gradient Descent and its variants

---

# 1. What is Gradient Descent?

**Gradient Descent is an optimization algorithm used to minimize a loss function by iteratively updating model parameters in the direction opposite to the gradient.**

In simple words\:

- A neural network makes predictions.
- We calculate how wrong those predictions are using a loss function.
- We calculate how each weight and bias contributes to that error.
- We adjust the weights and biases to reduce the error.
- We repeat this process until the model learns a good relationship between inputs and outputs.

### Real-world analogy

Imagine standing on top of a mountain covered in fog. Your goal is to reach the lowest point in the valley.

You cannot see the entire mountain, but you can feel the slope beneath your feet.

You follow these steps\:

1. Determine which direction slopes downward.
2. Take a step in that direction.
3. Check the slope again.
4. Continue taking steps until you reach a low point.

This is conceptually similar to Gradient Descent.

![](./images/gd0.jpg)

In Deep Learning\:

| Mountain analogy | Deep Learning |
|--------|--------|
| Height of the mountain | Loss value |
| Position on mountain | Model parameters |
| Downhill direction | Negative gradient |
| Step size | Learning rate |
| Lowest point | Minimum loss |
| Repeated steps | Training iterations |

The objective is to find model parameters that produce the minimum possible loss.

---

# 2. Why Do We Need Gradient Descent?

Consider a neural network that predicts house prices.

Suppose the actual house price is\:

$$
y=5000000
$$

But the network predicts\:

$$
\hat{y}=4000000
$$

The prediction error is\:

$$
Error=y-\hat{y}
$$

$$
Error=1000000
$$

The network needs to improve its prediction.

However, a neural network might contain thousands or millions of weights and biases.

The important question is\:

**How do we know which weights should be increased or decreased, and by how much?**

This is where Gradient Descent comes in.

It uses derivatives to determine how changing a parameter affects the loss.

For example\:

- If increasing a weight increases the loss, we should generally decrease that weight.
- If increasing a weight decreases the loss, we should generally increase that weight.

The gradient provides this direction.

---

# 3. Understanding the Loss Function

Before understanding Gradient Descent mathematically, let's recall the Loss Function.

A loss function measures the difference between the actual output and predicted output.

For example, Mean Squared Error (MSE)\:

$$
L=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Where\:

- $n$ \= number of training samples
- $y_i$ \= actual value
- $\hat y_i$ \= predicted value

Our objective is\:

$$
\boxed{\min_{\theta} L(\theta)}
$$

Here, $\theta$ represents all trainable parameters of the neural network (weights and biases).

For simplicity, imagine a loss function depending on just one weight\:

$$
L(w)=(w-3)^2
$$

Our goal is to find the value of $w$ that minimizes this loss.

Clearly\:

$$
w=3
$$

gives\:

$$
L(3)=(3-3)^2=0
$$

But Gradient Descent does not need to know the answer beforehand. It finds a good value by repeatedly updating the weight.

---

# 4. Understanding the Gradient

This is the most important mathematical concept behind Gradient Descent.

A **gradient measures how quickly a function changes with respect to its parameters and in which direction it increases most rapidly.**

For a function with one variable, the gradient is simply its derivative.

Suppose\:

$$
L(w)=(w-3)^2
$$

Differentiating with respect to $w$\:

$$
\frac{dL}{dw}=2(w-3)
$$

This derivative tells us how the loss changes when we change the weight.

Let's examine different weight values.

| Weight $w$ | Loss $L(w)$ | Gradient $dL/dw$ |
|--------|--------|--------|
| 1 | 4 | \-4 |
| 2 | 1 | \-2 |
| 3 | 0 | 0 |
| 4 | 1 | \+2 |
| 5 | 4 | \+4 |

What can we understand?

- Negative gradient: Increasing the weight will decrease the loss locally.
- Positive gradient: Decreasing the weight will decrease the loss locally.
- Zero gradient: The function has a stationary point.
  
![](./images/gd1.jpg)

Notice that the gradient is zero at the minimum of this particular function.

**Important\:** A zero gradient does not always mean a global minimum. It can also indicate a local maximum or saddle point.

## 4.1 Gradient with multiple variables

A neural network does not have just one weight.

Suppose our loss function depends on two weights\:

$$
L(w_1,w_2)
$$

The gradient becomes a vector of partial derivatives\:

$$
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}
\end{bmatrix}
$$

Each partial derivative measures how the loss changes when one parameter changes while the others remain fixed.

For a neural network with many parameters\:

$$
\nabla L(\theta)=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots\\
\frac{\partial L}{\partial b_1}\\
\frac{\partial L}{\partial b_2}
\end{bmatrix}
$$

This gradient tells us how to adjust every trainable parameter.

# 5. The Gradient Descent Update Rule

Now we come to the central equation of Gradient Descent.

$$
\boxed{w_{\text{new}}=w_{\text{old}}-\alpha\frac{dL}{dw}}
$$

For multiple parameters\:

$$
\boxed{\theta_{\text{new}}=\theta_{\text{old}}-\alpha\nabla L(\theta)}
$$

Where\:

- $\theta_{\text{new}}$ \= updated parameter
- $\theta_{\text{old}}$ \= previous parameter
- $\alpha$ \= learning rate
- $\nabla L(\theta)$ \= gradient of the loss function

Let's understand each part.

### Why do we subtract the gradient?

Remember\:

- Positive gradient means the loss increases as the parameter increases.
- Negative gradient means the loss decreases as the parameter increases.

We want to move in the direction that decreases the loss.

Therefore, we subtract the gradient.

The gradient points toward the steepest increase, so its negative points toward the steepest decrease.

### Why do we multiply by the learning rate?

The gradient tells us the direction and steepness, but we also need to determine how large a step to take.

That is the job of the learning rate.

For example\:

$$
w_{\text{new}}=w_{\text{old}}-0.1\times gradient
$$

Here, 0.1 is the learning rate.

---

# 6. Learning Rate

The **learning rate** determines how much the model's parameters are updated during each iteration of Gradient Descent.

It is one of the most important hyperparameters in Deep Learning.

Let's understand it with our previous loss function\:

$$
L(w)=(w-3)^2
$$

Its derivative is\:

$$
\frac{dL}{dw}=2(w-3)
$$

Assume\:

- Initial weight = 0
- Learning rate = 0.1

Our update rule is\:

$$
w_{\text{new}}=w_{\text{old}}-0.1(2(w_{\text{old}}-3))
$$

### Iteration 1

Initial weight\:

$$
w_0=0
$$

Calculate the gradient\:

$$
gradient=2(0-3)=-6
$$

Update the weight\:

$$
w_1=0-0.1(-6)
$$

$$
\boxed{w_1=0.6}
$$

Calculate the new loss\:

$$
L(0.6)=(0.6-3)^2=5.76
$$

Initially, the loss was 9. Now it is 5.76.

The loss has decreased\!

### Iteration 2

Current weight\:

$$
w_1=0.6
$$

Gradient\:

$$
gradient=2(0.6-3)=-4.8
$$

Update\:

$$
w_2=0.6-0.1(-4.8)
$$

$$
\boxed{w_2=1.08}
$$

New loss\:

$$
L(1.08)=(1.08-3)^2=3.6864
$$

### Iteration 3

$$
gradient=2(1.08-3)=-3.84
$$

$$
w_3=1.08-0.1(-3.84)
$$

$$
\boxed{w_3=1.464}
$$

New loss\:

$$
L(1.464)=(1.464-3)^2=2.3593
$$

We are gradually moving toward the minimum at $w=3$\.

### Observe the training progress

| Iteration | Weight | Loss |
|--------|--------|--------|
| 0 | 0.000 | 9.0000 |
| 1 | 0.600 | 5.7600 |
| 2 | 1.080 | 3.6864 |
| 3 | 1.464 | 2.3593 |
| 4 | 1.771 | 1.5099 |
| 5 | 2.017 | 0.9664 |
| 10 | 2.678 | 0.1038 |
| 20 | 2.965 | 0.0013 |

Notice The weight approaches 3, and the loss approaches zero.

## 6.1 What happens when the learning rate is too small?

Suppose\:

$$
\alpha=0.001
$$

The model takes very small steps.

Consequences\:

- Training becomes very slow.
- It may require many iterations.
- Computational resources may be wasted.

## 6.2 What happens when the learning rate is too large?

Suppose\:

$$
\alpha=1.5
$$

For our example, the update becomes\:

$$
w_{\text{new}}=w-1.5(2(w-3))
$$

Starting at $w=0$\:

$$
w_1=9
$$

The minimum is at 3, but the weight jumps to 9.

The next update\:

$$
w_2=9-1.5(12)= -9
$$

The weight jumps from one side to the other, and the loss increases dramatically.

This is called *overshooting*\.

![](./images/gd2.jpg)


| Learning rate | Behavior |
|--------|--------|
| Too small | Very slow convergence |
| Appropriate | Stable convergence |
| Too large | Overshooting or oscillation |
| Extremely large | Divergence; loss may increase continuously |

---

# 7. How Gradient Descent Works in a Neural Network

Now let's connect Gradient Descent with the MLP and Backpropagation concepts you have already studied.

Consider a simple neural network\:
![](./images/gd3.jpg)

A neural network has\:

- Input layer
- Hidden layer(s)
- Output layer
- Weights
- Biases
- Activation functions

During training, the following process takes place.

## Step 1: Initialize weights and biases

Initially, the model assigns weights and biases, usually using an initialization strategy such as Xavier or He initialization.

For example\:

$$
W_1=0.5,\quad W_2=0.3
$$

These are initial values, not necessarily good values.

## Step 2: Forward propagation

The input travels through the network.

For a simple neuron\:

$$
z=wx+b
$$

Then an activation function is applied\:

$$
\hat y=f(z)
$$

The network produces a prediction.

## Step 3: Calculate loss

Compare the predicted output with the actual output.

For MSE\:

$$
L=(y-\hat y)^2
$$

The loss measures how poorly the network is performing.

## Step 4: Backpropagation

Now the network calculates the gradients of the loss with respect to its weights and biases.

For example\:

$$
\frac{\partial L}{\partial w_1}
$$

$$
\frac{\partial L}{\partial w_2}
$$

$$
\frac{\partial L}{\partial b_1}
$$

Backpropagation uses the chain rule to calculate these derivatives efficiently, starting from the output layer and moving backward through the network.

## Step 5: Gradient Descent

Now the calculated gradients are used to update the parameters.

$$
w_1=w_1-\alpha\frac{\partial L}{\partial w_1}
$$

$$
w_2=w_2-\alpha\frac{\partial L}{\partial w_2}
$$

$$
b_1=b_1-\alpha\frac{\partial L}{\partial b_1}
$$

## Step 6: Repeat the process

The updated parameters are used in the next forward propagation.

The entire process repeats until the model reaches a satisfactory loss or a stopping condition.
![](./images/gd4.jpg)

**An important distinction\:**

- Backpropagation calculates the gradients.
- Gradient Descent uses those gradients to update the parameters.

They are related, but they are not the same thing.

---

# 8. Mathematical Example: Gradient Descent in a Single Neuron

Let's understand how the entire process works mathematically.

Consider a single neuron with a linear activation\:

$$
\hat y=wx+b
$$

Assume\:

- Input $x=2$
- Actual output $y=10$
- Initial weight $w=2$
- Initial bias $b=1$
- Learning rate $\alpha=0.01$

We will use the following loss function\:

$$
L=(y-\hat y)^2
$$

### Step 1: Forward propagation

$$
\hat y=wx+b
$$

$$
\hat y=2(2)+1=5
$$

### Step 2: Calculate loss

$$
L=(10-5)^2
$$

$$
\boxed{L=25}
$$

### Step 3: Calculate the gradient with respect to weight

Using the chain rule\:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\times
\frac{\partial \hat y}{\partial w}
$$

We know\:

$$
L=(y-\hat y)^2
$$

Therefore\:

$$
\frac{\partial L}{\partial \hat y}
=-2(y-\hat y)
$$

And\:

$$
\frac{\partial \hat y}{\partial w}=x
$$

Combining\:

$$
\frac{\partial L}{\partial w}
=-2(y-\hat y)x
$$

Substitute the values\:

$$
=-2(10-5)(2)
$$

$$
\boxed{\frac{\partial L}{\partial w}=-20}
$$

### Step 4: Calculate the gradient with respect to bias

Similarly\:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial \hat y}
\times
\frac{\partial \hat y}{\partial b}
$$

Since\:

$$
\frac{\partial \hat y}{\partial b}=1
$$

We get\:

$$
\frac{\partial L}{\partial b}=-2(10-5)
$$

$$
\boxed{\frac{\partial L}{\partial b}=-10}
$$

### Step 5: Update weight and bias

Weight update\:

$$
w_{\text{new}}=2-0.01(-20)
$$

$$
\boxed{w_{\text{new}}=2.2}
$$

Bias update\:

$$
b_{\text{new}}=1-0.01(-10)
$$

$$
\boxed{b_{\text{new}}=1.1}
$$

### Step 6: Perform forward propagation again

$$
\hat y=2.2(2)+1.1
$$

$$
\hat y=5.5
$$

New loss\:

$$
L=(10-5.5)^2
$$

$$
\boxed{L=20.25}
$$

(Loss decreased)

The loss decreased from 25 to 20.25 after just one update.

This demonstrates how Gradient Descent improves a neural network's parameters.

---

# 9. Types of Gradient Descent

Depending on how much training data is used to calculate the gradient before updating the parameters, Gradient Descent is divided into three major types.

1. Batch Gradient Descent
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent

This distinction is particularly important when training deep neural networks with large datasets.

## 9.1 Batch Gradient Descent

Batch Gradient Descent calculates the loss using the entire training dataset and then updates the parameters.

Suppose we have 1,000 training samples.

The process is\:

1. Pass all 1,000 samples through the network.
2. Calculate predictions for all samples.
3. Calculate the average loss.
4. Calculate gradients using all samples.
5. Update weights and biases once.

Mathematically\:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}L_i(\theta)
$$

The gradient is\:

$$
\nabla J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\nabla L_i(\theta)
$$

Where $m$ is the total number of training samples.

**Advantages\:**

- Stable gradient updates.
- Generally smooth convergence.
- Uses the complete dataset to calculate each update.

**Disadvantages\:**

- Requires more memory for large datasets.
- Each update is computationally expensive.
- Training can be slow.

## 9.2 Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent updates the parameters using only one training sample at a time.

Suppose we have 1,000 samples.

The process is\:

1. Take the first training sample.
2. Perform forward propagation.
3. Calculate its loss.
4. Calculate its gradient.
5. Update weights.
6. Repeat for the next sample.

Mathematically\:

$$
\theta=\theta-\alpha\nabla L_i(\theta)
$$

Here, $L_i$ represents the loss for a single sample.

**Advantages\:**

- Updates happen frequently.
- Requires less memory.
- Can make progress quickly on large datasets.
- Noisy updates can sometimes help escape saddle points or shallow local minima.

**Disadvantages\:**

- Loss fluctuates considerably.
- Updates are noisy.
- May not settle smoothly around a minimum.

## 9.3 Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a combination of Batch Gradient Descent and SGD.

Instead of using the entire dataset or just one sample, it uses a small group of samples called a *mini-batch*\.

For example\:

- Total samples = 1,000
- Batch size = 100
- Number of batches = 10

The process\:

1. Divide the dataset into 10 batches.
2. Process the first 100 samples.
3. Calculate average loss and gradients.
4. Update the parameters.
5. Process the next 100 samples.
6. Repeat until all batches are processed.

Mathematically\:

$$
\theta=\theta-\alpha
\frac{1}{B}\sum_{i=1}^{B}\nabla L_i(\theta)
$$

Where $B$ is the batch size.

**Advantages\:**

- More stable than single-sample SGD.
- Faster and more memory-efficient than full-batch training.
- Works well with GPU parallel processing.
- Commonly used in modern Deep Learning.

**Disadvantages\:**

- Batch size requires tuning.
- Gradients are still somewhat noisy.
- Requires memory proportional to the batch size.

### Comparison

| Feature | Batch GD | SGD | Mini-Batch GD |
|--------|--------|--------|--------|
| Samples per update | Entire dataset | 1 | Small group |
| Update frequency | Low | High | Medium |
| Memory usage | High | Low | Moderate |
| Gradient stability | High | Low | Moderate |
| GPU efficiency | Depends on implementation | Often lower | Usually high |
| Common usage | Smaller datasets | Online learning | Deep Learning |

(Important) Mini-Batch Gradient Descent is the most commonly used approach for training deep neural networks.

---

# 10. Epoch, Batch and Iteration

These three terms are closely related to Gradient Descent and are frequently used during model training.

### Epoch

An epoch means that the entire training dataset has passed through the neural network once.

### Batch

A batch is a subset of the training dataset processed together before one parameter update.

### Iteration

An iteration is one parameter-update step.

Let's take an example\:

Suppose\:

- Total training samples = 1,000
- Batch size = 100
- Epochs = 10

Number of batches\:

$$
\frac{1000}{100}=10
$$

Therefore\:

- 1 epoch = 10 iterations
- 10 epochs = 100 iterations
![](./images/gd5.jpg)

**Remember\:**

$$
\text{Iterations per epoch}=
\left\lceil\frac{\text{Number of samples}}{\text{Batch size}}\right\rceil
$$

The ceiling is used when the last batch contains fewer samples than the specified batch size and is not dropped.

---

# 11. Gradient Descent Using Python (From Scratch)

Let's implement Gradient Descent using NumPy without using any Deep Learning library.

We will use a simple linear regression problem.

Our objective is to find the best weight and bias for\:

$$
\hat y=wx+b
$$

Suppose our dataset is\:

| Input (x) | Actual output (y) |
|--------|--------|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

We can observe that the underlying relationship is\:

$$
y=2x+1
$$

Our model does not know this relationship initially.

It must learn the appropriate weight and bias through Gradient Descent.

### Python implementation

```python

import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):

    # 1. Forward propagation
    y_pred = w * X + b

    # 2. Calculate loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)

    # 3. Calculate gradients
    dw = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # 4. Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {loss:.6f}, "
            f"Weight = {w:.4f}, "
            f"Bias = {b:.4f}"
        )

print("\nFinal parameters:")
print("Weight:", w)
print("Bias:", b)

# Prediction
new_input = 6
prediction = w * new_input + b

print("Prediction for x=6:", prediction)
```

### Understanding the code

Let's understand the important parts.

**1\. Initialization**

```python
w = 0.0
b = 0.0
```

We start with initial weight and bias values.

**2\. Forward propagation**

```python
y_pred = w * X + b
```

The model generates predictions for all training samples.

**3\. Loss calculation**

```python
loss = np.mean((y - y_pred) ** 2)
```

We calculate the average squared error.

**4\. Gradient calculation**

```python
dw = (-2 / n) * np.sum(X * (y - y_pred))
db = (-2 / n) * np.sum(y - y_pred)
```

These equations calculate how the average loss changes with respect to the weight and bias.

**5\. Parameter update**

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

We adjust both parameters in the direction that reduces the loss.

### Expected result

After enough training, the parameters should approach\:

$$
w\approx2,\qquad b\approx1
$$

For an input of 6\:

$$
\hat y=2(6)+1=13
$$

The model has successfully learned the relationship between the inputs and outputs.

---

# 12. Gradient Descent in Deep Learning Using PyTorch

In real-world Deep Learning, we generally do not calculate every gradient manually.

Libraries such as PyTorch and TensorFlow provide automatic differentiation.

Let's understand the PyTorch implementation.

```python

import torch
import torch.nn as nn
import torch.optim as optim

# Training data
X = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0]]
)

y = torch.tensor(
    [[3.0], [5.0], [7.0], [9.0], [11.0]]
)

# Create a neural network with one linear layer
model = nn.Linear(1, 1)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

# Training
epochs = 1000

for epoch in range(epochs):

    # 1. Forward propagation
    predictions = model(X)

    # 2. Calculate loss
    loss = criterion(predictions, y)

    # 3. Clear previous gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update parameters
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )

# Inspect learned parameters
for name, param in model.named_parameters():
    print(name, param.data)

# Make prediction
test_input = torch.tensor([[6.0]])

with torch.no_grad():
    prediction = model(test_input)

print("Prediction:", prediction.item())
```

### Understanding the important PyTorch functions

| Function | Purpose |
|--------|--------|
| `nn.Linear(1, 1)` | Creates a linear layer with one input and one output |
| `nn.MSELoss()` | Defines the loss function |
| `optim.SGD()` | Creates an SGD optimizer |
| `optimizer.zero_grad()` | Clears previous gradients |
| `loss.backward()` | Calculates gradients using backpropagation |
| `optimizer.step()` | Updates model parameters |

Notice these two lines\:

```python
loss.backward()
optimizer.step()
```

The first performs gradient calculation through backpropagation.

The second performs parameter updates through Gradient Descent.

This is how the two concepts work together in practice.

---

# 13. Problems and Challenges in Gradient Descent

Although Gradient Descent is fundamental to Deep Learning, it has several challenges.

Understanding these problems will help you understand why more advanced optimization algorithms such as Momentum, RMSprop and Adam were developed.

## 13.1 Local Minima

A local minimum is a point where the loss is lower than the loss at nearby points, but it may not be the lowest point in the entire loss landscape.

Imagine a mountain range with several valleys.

Gradient Descent might reach a small valley instead of the deepest valley.
![](./images/gd6.jpg)

**Why is this a problem?**

The model might stop improving at a solution that is not globally optimal.

However, in large neural networks, saddle points and flat regions are often more significant practical challenges than simply getting trapped in poor local minima.

## 13.2 Saddle Points

A saddle point is a point where the gradient is zero, but the point is neither a local minimum nor a local maximum.

Consider\:

$$
L(x,y)=x^2-y^2
$$

At\:

$$
(x,y)=(0,0)
$$

Both partial derivatives are zero\:

$$
\frac{\partial L}{\partial x}=2x=0
$$

$$
\frac{\partial L}{\partial y}=-2y=0
$$

However, moving along the x-axis increases the function, while moving along the y-axis decreases it.
![](./images/gd7.jpg)

This is called a saddle point because the surface curves upward in one direction and downward in another.

## 13.3 Vanishing Gradient Problem

This problem occurs when gradients become extremely small as they propagate backward through many layers.

For example\:

$$
gradient=0.0000001
$$

When gradients become very small\:

- Weight updates become extremely small.
- Earlier layers learn very slowly.
- Deep networks may struggle to learn useful representations.

It commonly occurs with sigmoid and tanh activation functions in deep networks.

**Solutions include\:**

- ReLU and its variants.
- Proper weight initialization.
- Batch normalization.
- Residual connections.
![](./images/gd8.jpg)

## 13.4 Exploding Gradient Problem

This is the opposite of vanishing gradients.

Gradients become extremely large during backpropagation.

For example\:

$$
gradient=1000000
$$

This can result in\:

- Extremely large weight updates.
- Unstable training.
- Loss values becoming very large.
- Numerical overflow.

**Solutions include\:**

- Gradient clipping.
- Proper weight initialization.
- Normalization.
- Appropriate learning rates.

## 13.5 Choosing the Right Learning Rate

An inappropriate learning rate can prevent successful training.

A very small learning rate makes learning slow, while a very large learning rate may cause divergence.

This is one reason adaptive optimization algorithms were developed.

---

# 14. Advanced Gradient Descent Optimization Algorithms

You now understand basic Gradient Descent. However, modern Deep Learning commonly uses improved optimization algorithms.

These algorithms are built around the same fundamental idea: calculating gradients and updating parameters.

The major algorithms include\:

1. Momentum
2. Nesterov Accelerated Gradient (NAG)
3. AdaGrad
4. RMSprop
5. Adam

Let's briefly understand each one.

## 14.1 Gradient Descent with Momentum

Normal Gradient Descent uses the current gradient to determine the update.

Momentum additionally considers previous gradients.

Think of a ball rolling down a hill.

As the ball continues moving in the same direction, it builds momentum.

Similarly, Momentum remembers previous update directions and uses them to accelerate training in consistent directions.

A simplified mathematical representation\:

$$
v_t=\beta v_{t-1}+\nabla L(\theta_t)
$$

$$
\theta_{t+1}=\theta_t-\alpha v_t
$$

Where\:

- $v_t$ \= velocity
- $\beta$ \= momentum coefficient
- $\alpha$ \= learning rate

**Benefits\:**

- Accelerates convergence.
- Reduces oscillations.
- Helps move through shallow regions.

## 14.2 Nesterov Accelerated Gradient (NAG)

NAG is an improvement over traditional Momentum.

Instead of calculating the gradient only at the current position, it calculates it at an approximate future position based on the current momentum.

This allows the optimizer to anticipate where it is heading and adjust its direction.

## 14.3 AdaGrad

AdaGrad adapts the learning rate separately for each parameter.

Parameters that receive large or frequent gradients get smaller effective learning rates, while parameters with relatively small gradients get larger effective learning rates.

This can be useful for sparse data.

However, its learning rate can become excessively small over time.

## 14.4 RMSprop

RMSprop addresses the continuously decreasing learning rate problem of AdaGrad.

It maintains an exponentially weighted average of squared gradients.

A simplified representation\:

$$
s_t=\beta s_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_{t+1}
=
\theta_t-\frac{\alpha}{\sqrt{s_t}+\epsilon}g_t
$$

Here, $g_t$ is the current gradient.

This allows the optimizer to adapt learning rates according to recent gradient magnitudes.

## 14.5 Adam (Adaptive Moment Estimation)

Adam combines the ideas of Momentum and RMSprop.

It maintains\:

- First moment: Exponentially weighted average of gradients.
- Second moment: Exponentially weighted average of squared gradients.

First moment\:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

Second moment\:

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

It also uses bias corrections\:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Final update\:

$$
\boxed{
\theta_{t+1}
=
\theta_t-
\frac{\alpha\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}
$$

**Why is Adam popular?**

- Adapts learning rates for individual parameters.
- Uses information from previous gradients.
- Works well across many deep learning tasks.
- Often provides good initial training performance without extensive tuning.

### Comparison of optimization algorithms

| Algorithm | Main idea | Key advantage |
|--------|--------|--------|
| SGD | Current gradient | Simple and memory-efficient |
| Momentum | Previous gradients | Reduces oscillations |
| NAG | Look-ahead gradient | Anticipates gradient direction |
| AdaGrad | Accumulated squared gradients | Adapts to sparse features |
| RMSprop | Recent squared gradients | Prevents excessive learning-rate decay |
| Adam | Momentum + adaptive scaling | Effective general-purpose optimizer |

These are not completely separate from Gradient Descent. They are different strategies for using gradients to update neural network parameters.

---

# 15. Gradient Descent vs Backpropagation

This is an important conceptual distinction for exams and interviews.

| Backpropagation | Gradient Descent |
|--------|--------|
| Calculates gradients | Updates parameters |
| Uses chain rule | Uses optimization update rules |
| Determines how loss changes with each parameter | Determines how parameters should change |
| Works backward through the network | Applies updates to weights and biases |
| Calculates derivatives | Uses derivatives to minimize loss |

### How they work together

![](./images/gd9.jpg)

The complete training cycle is\:

1. Forward propagation generates predictions.
2. Loss function calculates prediction error.
3. Backpropagation calculates gradients.
4. Gradient Descent updates weights and biases.
5. The next iteration begins.

This cycle continues for multiple epochs.

---

# 16. Important Terms to Remember

| Term | Meaning |
|--------|--------|
| Gradient | Direction and rate of steepest increase of a function |
| Loss | Measure of prediction error |
| Learning rate | Controls parameter update size |
| Parameter | Trainable weight or bias |
| Epoch | One complete pass through training data |
| Batch | Subset of training samples |
| Iteration | One parameter update |
| Convergence | Approaching a stable solution |
| Overshooting | Jumping past the minimum |
| Divergence | Moving away from a solution |
| Optimizer | Algorithm responsible for updating parameters |

---

# 17. Summary: Complete Deep Learning Training Process

Let's connect everything you have studied so far.

## Neural Network Training Pipeline

**1\. Input training data**

Features and target values

**2\. Forward Propagation**

Calculate predictions using weights and biases

**3\. Loss Calculation**

Measure the difference between actual and predicted values

**4\. Backpropagation**

Calculate gradients using the chain rule

**5\. Gradient Descent / Optimizer**

Update weights and biases using gradients

Repeat for subsequent batches and epochs

**6\. Trained Neural Network**

A model that has learned useful patterns from training data

The most important equations to remember are\:

**1\. Gradient Descent update rule**

$$
\theta_{\text{new}}=\theta_{\text{old}}-\alpha\nabla L(\theta)
$$

**2\. MSE Loss**

$$
L=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

**3\. Number of iterations per epoch**

$$
\text{Iterations}=
\left\lceil\frac{N}{B}\right\rceil
$$

---
