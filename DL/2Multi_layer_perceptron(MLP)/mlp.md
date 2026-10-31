# Multi-Layer Perceptron (MLP) in Deep Learning – Detailed Study Notes

A **Multi-Layer Perceptron (MLP)** is one of the fundamental architectures of Artificial Neural Networks (ANNs) and an important starting point for understanding Deep Learning.

We will cover MLP from its mathematical foundations to forward propagation, activation functions, loss functions, backpropagation, optimization, and implementation using Python.

# 1. What is a Multi-Layer Perceptron (MLP)?

A Multi-Layer Perceptron is a type of **feedforward artificial neural network** consisting of multiple layers of interconnected neurons.

Unlike a single-layer perceptron, which can learn only linearly separable patterns, an MLP can learn complex, nonlinear relationships between inputs and outputs.

An MLP generally consists of three types of layers\:

1. Input Layer
2. Hidden Layer(s)
3. Output Layer

### Example

Suppose we want to build a neural network that predicts whether a student will pass or fail based on\:

- Hours studied
- Attendance percentage
- Previous exam score

Our neural network receives these three input features and predicts the probability of passing.

For example\:

```text
Input:
Hours studied       = 5
Attendance          = 85%
Previous exam score = 70

Output:
Probability of passing = 0.92
```

The MLP learns the relationship between these inputs and the output through training.

## 1.1 Why do we need MLP?

Consider the limitations of a single-layer perceptron.

A single-layer perceptron calculates\:

$$
z = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

Then applies an activation function\:

$$
\hat y=f(z)
$$

It can create only a linear decision boundary for binary classification.

For example, it can solve\:

- AND gate
- OR gate
- Linearly separable classification problems

However, it cannot solve the XOR problem.

An MLP solves this limitation by introducing hidden layers and nonlinear activation functions.

The combination allows the network to learn nonlinear decision boundaries.

---

# 2. Architecture of MLP

Let's understand the three main components.

## 2.1 Input Layer

The input layer receives the input features from the dataset.

Suppose our dataset contains\:

| Feature | Value |
|--------|--------|
| Age | 22 |
| Study hours | 5 |
| Attendance | 85 |
| Previous score | 70 |

We have four input features, so our input layer contains four neurons.

Important points\:

- Each input neuron represents one feature.
- The input layer does not generally perform trainable mathematical transformations.
- It passes the input values to the first hidden layer.

If the input vector is\:

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}
$$

Then the input layer contains four neurons.

## 2.2 Hidden Layer

The hidden layer is where most of the learning and computation happens.

Each neuron receives inputs from the previous layer, multiplies them by corresponding weights, adds a bias, and applies an activation function.

Mathematically\:

$$
z_j=\sum_{i=1}^{n}w_{ij}x_i+b_j
$$

Then\:

$$
a_j=f(z_j)
$$

Where\:

- $x_i$ \= input from the previous layer
- $w_{ij}$ \= weight connecting input neuron $i$ to neuron $j$
- $b_j$ \= bias of neuron $j$
- $f$ \= activation function
- $a_j$ \= output of neuron $j$

### Why are hidden layers called hidden?

Because their outputs are not directly exposed as the final prediction.

They learn intermediate representations of the input data.

For example, when predicting student performance\:

- Some neurons might learn relationships involving attendance.
- Some might learn relationships involving study hours.
- Other neurons may combine these learned relationships to identify more complex patterns.

These are conceptual interpretations; individual neurons do not necessarily learn such clean, human-readable concepts.

## 2.3 Output Layer

The output layer produces the final prediction.

Its structure depends on the problem we are solving.

| Problem | Output neurons | Activation |
|--------|--------|--------|
| Binary classification | 1 | Sigmoid |
| Multiclass classification | Number of classes | Softmax |
| Regression | 1 | Linear |
| Multi-output regression | Number of targets | Linear (commonly) |

Example\:

For predicting whether a student passes or fails\:

$$
\hat y=\sigma(z)
$$

Suppose\:

$$
\hat y=0.92
$$

This means the model predicts a 92% probability of passing.

---

# 3. How neurons are connected in MLP

In a standard fully connected MLP, every neuron in one layer is connected to every neuron in the next layer.

This is also called a **Dense Layer** or Fully Connected Layer.

Consider this architecture\:

Suppose we have\:

- Input layer: 3 neurons
- Hidden layer 1: 4 neurons
- Hidden layer 2: 3 neurons
- Output layer: 1 neuron

The architecture can be represented as\:

$$
3\rightarrow4\rightarrow3\rightarrow1
$$

### Number of weights

The number of connections between two consecutive layers is\:

$$
W=n_{\text{previous}}\times n_{\text{current}}
$$

For our architecture\:

| Connection | Weight count |
|--------|--------|
| Input → Hidden 1 | $3\times4=12$ |
| Hidden 1 → Hidden 2 | $4\times3=12$ |
| Hidden 2 → Output | $3\times1=3$ |
| **Total** | **27** |

### Number of biases

Every neuron in a hidden or output layer has one bias.

$$
B=4+3+1=8
$$

Therefore\:

$$
\text{Total parameters}=W+B
$$

$$
=27+8=\boxed{35}
$$

**Important\:** The input layer does not have trainable weights or biases of its own.

---

# 4. Forward Propagation in MLP

Forward propagation is the process of passing input data through the neural network to generate a prediction.

This is the first major mathematical operation in MLP.

Let's understand it step by step.

Suppose we have a simple MLP\:

- 2 input neurons
- 2 hidden neurons
- 1 output neuron

Our input is\:

$$
X=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

### Step 1: Calculate the hidden layer weighted sum

Suppose the weights are\:

$$
W_1=
\begin{bmatrix}
0.2 & 0.4\\
0.3 & 0.5
\end{bmatrix}
$$

Bias\:

$$
b_1=
\begin{bmatrix}
0.1\\
0.2
\end{bmatrix}
$$

Using\:

$$
Z_1=W_1^TX+b_1
$$

$$
Z_1=
\begin{bmatrix}
0.2 & 0.3\\
0.4 & 0.5
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
0.2
\end{bmatrix}
$$

Therefore\:

$$
z_{11}=0.2(1)+0.3(2)+0.1=0.9
$$

$$
z_{12}=0.4(1)+0.5(2)+0.2=1.6
$$

So\:

$$
Z_1=
\begin{bmatrix}
0.9\\
1.6
\end{bmatrix}
$$

### Step 2: Apply activation function

Let's use ReLU.

$$
ReLU(z)=\max(0,z)
$$

Therefore\:

$$
A_1=
\begin{bmatrix}
ReLU(0.9)\\
ReLU(1.6)
\end{bmatrix}
$$

$$
A_1=
\begin{bmatrix}
0.9\\
1.6
\end{bmatrix}
$$

These are the outputs of our hidden neurons.

### Step 3: Calculate the output layer

Suppose\:

$$
W_2=
\begin{bmatrix}
0.6\\
0.7
\end{bmatrix}
$$

And\:

$$
b_2=0.1
$$

Then\:

$$
Z_2=W_2^TA_1+b_2
$$

$$
Z_2=0.6(0.9)+0.7(1.6)+0.1
$$

$$
Z_2=1.76
$$

### Step 4: Apply the output activation function

Since this is binary classification, we use sigmoid.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

$$
\hat y=\frac{1}{1+e^{-1.76}}
$$

$$
\boxed{\hat y\approx0.853}
$$

Our model predicts an 85.3% probability of the positive class.

This completes forward propagation.

---

# 5. Activation Functions in MLP

Activation functions are among the most important components of an MLP.

Without activation functions, an MLP with multiple layers would behave like a single linear transformation, regardless of how many layers it contains.

Let's understand why.

Suppose we have two layers\:

$$
Z_1=W_1X+b_1
$$

$$
Z_2=W_2Z_1+b_2
$$

Substituting\:

$$
Z_2=W_2(W_1X+b_1)+b_2
$$

$$
Z_2=W_2W_1X+W_2b_1+b_2
$$

This is still a linear transformation of the input.

Therefore, simply adding layers does not automatically make a neural network capable of learning nonlinear relationships.

**Activation functions introduce nonlinearity into the network.**

## 5.1 Sigmoid

The sigmoid function converts any real-valued input into a value between 0 and 1.

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Properties\:

- Output range: (0, 1)
- Smooth and differentiable.
- Commonly used in binary classification output layers.
- Can be interpreted as a probability when used with an appropriate classification model.

**Problem: Vanishing Gradient**

When the input is very large or very small, the derivative of sigmoid approaches zero.

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

This can make training deep networks difficult because gradients become extremely small.

## 5.2 Tanh

Tanh stands for hyperbolic tangent.

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Properties\:

- Output range: (-1, 1).
- Zero-centered.
- Smooth and differentiable.
- Can suffer from vanishing gradients.

Tanh is often preferable to sigmoid in some hidden-layer settings because its output is zero-centered.

## 5.3 ReLU (Rectified Linear Unit)

ReLU is one of the most commonly used activation functions in hidden layers.

$$
ReLU(x)=\max(0,x)
$$

Examples\:

| Input | ReLU output |
|--------|--------|
| \-5 | 0 |
| \-2 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

Properties\:

- Simple and computationally efficient.
- Helps reduce the vanishing gradient problem for positive activations.
- Produces sparse activations.
- Commonly used in hidden layers.

**Problem: Dying ReLU**

If a neuron consistently receives negative inputs, its output becomes zero and its gradient is also zero. It may stop learning.

## 5.4 Leaky ReLU

Leaky ReLU addresses the dying ReLU problem by allowing a small negative slope.

$$
f(x)=
\begin{cases}
x & x>0\\
\alpha x & x\leq0
\end{cases}
$$

Usually\:

$$
\alpha=0.01
$$

Example\:

$$
f(-5)=0.01(-5)=-0.05
$$

Unlike standard ReLU, negative inputs can still produce nonzero gradients.

## 5.5 Softmax

Softmax is commonly used in the output layer of multiclass classification networks.

It converts raw output scores (logits) into probabilities.

Formula\:

$$
Softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

Where\:

- $z_i$ \= score for class $i$\.
- $C$ \= total number of classes.

Example\:

Suppose our network produces three output scores\:

$$
Z=[2,1,0.1]
$$

After applying softmax, approximately\:

$$
P=[0.659,0.242,0.099]
$$

Therefore, the first class has the highest predicted probability.

### Activation function comparison

| Activation | Output range | Common use |
|--------|--------|--------|
| Sigmoid | $0, 1) | Binary output |
| Tanh | \(-1, 1) | Some hidden layers |
| ReLU | $$0, ∞) | Hidden layers |
| Leaky ReLU | \(-∞, ∞) | Hidden layers |
| Softmax | \(0, 1), sum = 1 | Multiclass output |
| Linear | \(-∞, ∞) | Regression output |

**Practical recommendation\:** Start with ReLU for hidden layers, sigmoid for binary classification, softmax for multiclass classification, and linear activation for regression.

---

# 6. Loss Functions in MLP

After forward propagation, the network generates a prediction.

But how does the network know whether its prediction is correct?

This is where the loss function comes in.

A loss function measures the difference between the actual target and the predicted output.

\[
Loss=L(y,\hat y)
$$

Where\:

- \(y$ \= actual value.
- $\hat y$ \= predicted value.

The objective of training is to minimize this loss.

## 6.1 Mean Squared Error (MSE)

MSE is commonly used for regression problems.

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Example\:

Actual values\:

$$
y=[10,20,30]
$$

Predictions\:

$$
\hat y=[12,18,27]
$$

Then\:

$$
MSE=\frac{(10-12)^2+(20-18)^2+(30-27)^2}{3}
$$

$$
=\frac{4+4+9}{3}
$$

$$
\boxed{MSE=5.67}
$$

## 6.2 Binary Cross-Entropy (BCE)

Used for binary classification.

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

For multiple samples\:

$$
L=-\frac{1}{n}\sum_{i=1}^{n}
[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)]
$$

Example\:

Actual label\:

$$
y=1
$$

Predicted probability\:

$$
\hat y=0.9
$$

Then\:

$$
L=-\log(0.9)
$$

$$
\boxed{L\approx0.105}
$$

A prediction of 0.9 for the correct positive class produces a relatively small loss.

If the model predicted 0.1 instead, its loss would be much higher.

## 6.3 Categorical Cross-Entropy

Used for multiclass classification.

$$
L=-\sum_{i=1}^{C}y_i\log(\hat y_i)
$$

For a single example with a one-hot encoded target, only the correct class contributes to the loss.

Example\:

Actual class\:

$$
y=[0,1,0]
$$

Predicted probabilities\:

$$
\hat y=[0.1,0.8,0.1]
$$

Then\:

$$
L=-\log(0.8)
$$

$$
\boxed{L\approx0.223}
$$

### Which loss function should you use?

| Problem | Output activation | Loss function |
|--------|--------|--------|
| Binary classification | Sigmoid | Binary cross-entropy |
| Multiclass classification | Softmax | Categorical cross-entropy |
| Multiclass with integer labels | Softmax | Sparse categorical cross-entropy |
| Regression | Linear | MSE or MAE |

---

# 7. Backpropagation in MLP

(Most important concept)

Backpropagation is the algorithm used to calculate how much each weight and bias contributes to the prediction error.

It allows the neural network to learn by calculating gradients of the loss function with respect to its parameters.

Remember\:

- Forward propagation calculates predictions.
- Loss calculation measures prediction error.
- Backpropagation calculates gradients.
- Optimization algorithms update weights and biases.

These processes work together during training.

## 7.1 Why do we need backpropagation?

Suppose our network predicts\:

$$
\hat y=0.85
$$

But the actual answer is\:

$$
y=1
$$

There is an error between the actual and predicted values.

We need to determine\:

- Which weights contributed to the error?
- How much did each weight contribute?
- In which direction should we change each weight?
- How much should we change it?

Backpropagation answers these questions using the chain rule of calculus.

## 7.2 Understanding the chain rule

Suppose\:

$$
y=f(g(x))
$$

Then\:

$$
\frac{dy}{dx}=
\frac{dy}{dg}\times\frac{dg}{dx}
$$

In an MLP, the loss depends on the output, the output depends on the hidden layer, and the hidden layer depends on the weights.

Therefore, we calculate derivatives backward through the network.

## 7.3 Mathematical derivation of backpropagation

Consider a simple network\:

$$
Input\rightarrow Hidden\rightarrow Output
$$

For simplicity, assume one neuron in each layer.

**Step 1: Forward propagation**

Hidden layer\:

$$
z_1=w_1x+b_1
$$

$$
a_1=f(z_1)
$$

Output layer\:

$$
z_2=w_2a_1+b_2
$$

$$
\hat y=\sigma(z_2)
$$

**Step 2: Calculate loss**

For binary classification, use binary cross-entropy\:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

**Step 3: Calculate output layer error**

For sigmoid activation combined with binary cross-entropy\:

$$
\frac{\partial L}{\partial z_2}=\hat y-y
$$

Let\:

$$
\delta_2=\hat y-y
$$

This is the output-layer error term.

**Step 4: Calculate output layer gradients**

Using the chain rule\:

$$
\frac{\partial L}{\partial w_2}
=
\frac{\partial L}{\partial z_2}
\times
\frac{\partial z_2}{\partial w_2}
$$

Since\:

$$
z_2=w_2a_1+b_2
$$

We have\:

$$
\boxed{\frac{\partial L}{\partial w_2}=\delta_2a_1}
$$

Similarly\:

$$
\boxed{\frac{\partial L}{\partial b_2}=\delta_2}
$$

**Step 5: Calculate hidden layer error**

The hidden layer also contributes to the final prediction.

Using the chain rule\:

$$
\delta_1=
\frac{\partial L}{\partial z_1}
$$

$$
\boxed{\delta_1=\delta_2w_2f'(z_1)}
$$

This tells us how much the hidden neuron's weighted sum contributes to the final loss.

**Step 6: Calculate hidden layer gradients**

Since\:

$$
z_1=w_1x+b_1
$$

We get\:

$$
\boxed{\frac{\partial L}{\partial w_1}=\delta_1x}
$$

And\:

$$
\boxed{\frac{\partial L}{\partial b_1}=\delta_1}
$$

We have now calculated gradients for all trainable parameters.

## 7.4 General backpropagation equations

For a network with multiple layers\:

Forward propagation\:

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}
$$

$$
A^{[l]}=f^{[l]}(Z^{[l]})
$$

Output error\:

$$
\delta^{[L]}=
\frac{\partial L}{\partial Z^{[L]}}
$$

Hidden layer error\:

$$
\delta^{[l]}=
(W^{[l+1]})^T\delta^{[l+1]}
\odot f'^{[l]}(Z^{[l]})
$$

Weight gradients\:

$$
\boxed{
\frac{\partial L}{\partial W^{[l]}}
=
\delta^{[l]}(A^{[l-1]})^T
}
$$

Bias gradients\:

$$
\boxed{
\frac{\partial L}{\partial b^{[l]}}
=
\delta^{[l]}
}
$$

Here\:

- $L$ \= final layer.
- $l$ \= current layer.
- $\odot$ \= element-wise multiplication.
- $T$ \= matrix transpose.

For batch training, gradients are averaged across samples.

---

# 8. Weight and Bias Updates – Gradient Descent

After backpropagation calculates gradients, the network needs to update its weights and biases.

This is done using an optimization algorithm.

The most basic optimization algorithm is Gradient Descent.

The weight update formula is\:

$$
\boxed{w_{\text{new}}=w_{\text{old}}-\alpha\frac{\partial L}{\partial w}}
$$

Similarly, for bias\:

$$
\boxed{b_{\text{new}}=b_{\text{old}}-\alpha\frac{\partial L}{\partial b}}
$$

Where\:

- $\alpha$ \= learning rate.
- $\frac{\partial L}{\partial w}$ \= gradient of loss with respect to weight.

### Example

Suppose\:

$$
w=0.5
$$

$$
\frac{\partial L}{\partial w}=0.2
$$

$$
\alpha=0.1
$$

Then\:

$$
w_{\text{new}}=0.5-0.1(0.2)
$$

$$
\boxed{w_{\text{new}}=0.48}
$$

The weight has decreased because its gradient was positive.

The purpose is to move the parameter in a direction that reduces the loss.

## 8.1 Different Gradient Descent approaches

There are three common approaches to updating parameters.

**1\. Batch Gradient Descent**

Uses the entire training dataset to calculate gradients before updating weights.

$$
w=w-\alpha\frac{1}{N}\sum_{i=1}^{N}
\frac{\partial L_i}{\partial w}
$$

Advantages\:

- Stable gradient estimates.
- Uses the complete dataset.

Disadvantages\:

- Can be slow for large datasets.
- Requires processing all samples before an update.

**2\. Stochastic Gradient Descent (SGD)**

Updates weights after processing each individual training sample.

$$
w=w-\alpha\frac{\partial L_i}{\partial w}
$$

Advantages\:

- Frequent updates.
- Can work well with large datasets.

Disadvantages\:

- Noisy updates.
- Loss may fluctuate during training.

**3\. Mini-Batch Gradient Descent**

Uses a small group of training samples to calculate gradients.

For example\:

- Dataset size = 10,000
- Batch size = 32
- Each update processes 32 samples.

$$
w=w-\alpha\frac{1}{B}
\sum_{i=1}^{B}\frac{\partial L_i}{\partial w}
$$

Mini-batch gradient descent is widely used for training neural networks.

## 8.2 Other optimization algorithms

| Optimizer | Main idea |
|--------|--------|
| SGD | Updates using sample or mini-batch gradients |
| Momentum | Uses previous update directions to smooth optimization |
| RMSprop | Adapts learning rates using moving averages of squared gradients |
| Adam | Combines momentum and adaptive learning rates |
| AdamW | Adam-style optimization with decoupled weight decay |

**Recommendation\:** Start with SGD to understand gradient descent, then learn Momentum and Adam.

---

# 9. Complete MLP Training Process

Now let's combine everything.

Training an MLP generally involves the following steps\:

### Step-by-step training

1. Initialize weights and biases.
2. Take a batch of training samples.
3. Perform forward propagation.
4. Calculate predictions.
5. Calculate the loss.
6. Perform backpropagation.
7. Calculate gradients.
8. Update weights and biases.
9. Repeat until the training process is complete.

### What is an epoch?

An epoch means that the model has processed the entire training dataset once.

Suppose\:

- Training samples = 1,000
- Batch size = 100

Then\:

$$
\text{Batches per epoch}=\frac{1000}{100}=10
$$

The model performs 10 parameter updates per epoch, assuming the dataset divides evenly into batches.

If we train for 50 epochs\:

$$
\text{Total updates}=10\times50=500
$$

### Important training terminology

| Term | Meaning |
|--------|--------|
| Epoch | One complete pass through training data |
| Batch size | Number of samples processed per update |
| Iteration | One optimizer update |
| Learning rate | Controls the size of parameter updates |
| Loss | Measures prediction error |
| Gradient | Measures how loss changes with parameters |
| Optimizer | Algorithm that updates parameters |

---

# 10. Implementing MLP Using NumPy From Scratch

This is an important practical exercise because it helps you understand what happens internally rather than depending entirely on a deep learning framework.

We will build an MLP with\:

- 2 input neurons
- 4 hidden neurons
- 1 output neuron
- ReLU activation in the hidden layer
- Sigmoid activation in the output layer
- Binary cross-entropy loss
- Gradient descent optimization

We will use the XOR dataset.

## 10.1 Why XOR?

XOR is a classic example of a nonlinear classification problem.

| Input 1 | Input 2 | Output |
|--------|--------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single-layer perceptron cannot correctly classify all four XOR combinations.

An MLP with a nonlinear hidden layer can learn this relationship.

## 10.2 Import libraries and prepare data

```python
import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)
```

## 10.3 Initialize weights and biases

```python
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Input to hidden
W1 = np.random.randn(
    input_neurons, hidden_neurons
) * np.sqrt(2 / input_neurons)

b1 = np.zeros((1, hidden_neurons))

# Hidden to output
W2 = np.random.randn(
    hidden_neurons, output_neurons
) * np.sqrt(1 / hidden_neurons)

b2 = np.zeros((1, output_neurons))
```

We use He-style initialization for the ReLU hidden layer.

The purpose of weight initialization is to give neurons useful starting values and help maintain reasonable activation and gradient magnitudes.

## 10.4 Define activation functions

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## 10.5 Forward propagation

```python
def forward_propagation(X):

    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2
```

## 10.6 Calculate loss

```python
def calculate_loss(y, y_pred):

    epsilon = 1e-8

    loss = -np.mean(
        y * np.log(y_pred + epsilon) +
        (1 - y) * np.log(1 - y_pred + epsilon)
    )

    return loss
```

We add a small epsilon to avoid taking the logarithm of zero.

## 10.7 Backpropagation

```python
def backward_propagation(X, y, Z1, A1, A2):

    m = X.shape[0]

    # Output layer gradient
    dZ2 = A2 - y

    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer gradient
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2
```

Notice\:

```python
dZ2 = A2 - y
```

This is the simplified output-layer gradient resulting from using sigmoid with binary cross-entropy.

## 10.8 Update weights

```python
def update_parameters(
    dW1, db1, dW2, db2, learning_rate
):

    global W1, b1, W2, b2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
```

## 10.9 Train the network

```python
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):

    # Forward propagation
    Z1, A1, Z2, A2 = forward_propagation(X)

    # Calculate loss
    loss = calculate_loss(y, A2)

    # Backpropagation
    dW1, db1, dW2, db2 = backward_propagation(
        X, y, Z1, A1, A2
    )

    # Update parameters
    update_parameters(
        dW1, db1, dW2, db2, learning_rate
    )

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.6f}")
```

## 10.10 Test the model

```python
_, _, _, predictions = forward_propagation(X)

print("Predictions:")
print(predictions)

print("\nBinary predictions:")
print((predictions >= 0.5).astype(int))
```

Expected binary predictions\:

```text
[[0]
 [1]
 [1]
 [0]]
```

The exact probabilities and training loss depend on initialization and training settings.

**What you have implemented\:** A complete neural network training pipeline, including forward propagation, loss calculation, backpropagation, and gradient descent, without using TensorFlow or PyTorch.

---

---

# 11. MLP Using Scikit-Learn

Now that you understand the internal implementation, let's see how to build an MLP using Scikit-Learn.

Scikit-Learn provides `MLPClassifier` and `MLPRegressor`\.

- `MLPClassifier`\: Classification problems.
- `MLPRegressor`\: Regression problems.

## 11.1 MLPClassifier

Let's solve the XOR problem again.

```python
import numpy as np

from sklearn.neural_network import MLPClassifier

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=10000,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```

Expected predictions\:

```text
[0 1 1 0]
```

### Understanding the parameters

| Parameter | Meaning |
|--------|--------|
| `hidden_layer_sizes` | Number of neurons in each hidden layer |
| `activation` | Hidden-layer activation function |
| `solver` | Optimization algorithm |
| `learning_rate_init` | Initial learning rate |
| `max_iter` | Maximum training iterations |
| `random_state` | Reproducibility |

For example\:

```python
hidden_layer_sizes=(64, 32, 16)
```

Means\:

- First hidden layer: 64 neurons
- Second hidden layer: 32 neurons
- Third hidden layer: 16 neurons

The number of input and output neurons is determined automatically by the training data.

## 11.2 MLPRegressor

Suppose we want to predict house prices using\:

- Area
- Number of bedrooms
- Number of bathrooms

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

We use `StandardScaler` because MLPs are sensitive to feature scales.

For example\:

- Area might range from 500 to 5000.
- Bedrooms might range from 1 to 5.
- Bathrooms might range from 1 to 4.

Features with substantially different scales can make optimization difficult.

Using standardization\:

$$
z=\frac{x-\mu}{\sigma}
$$

helps bring features onto comparable scales.

---

# 12. MLP Using TensorFlow and Keras

Scikit-Learn is useful for learning and traditional machine learning workflows.

For deeper neural network development, you should also learn TensorFlow/Keras and PyTorch.

Let's implement the same type of MLP using Keras.

## 12.1 Import libraries

```python
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## 12.2 Define the architecture

```python
model = Sequential([

    Dense(
        4,
        activation='relu',
        input_shape=(2,)
    ),

    Dense(
        1,
        activation='sigmoid'
    )

])
```

Architecture\:

$$
2\rightarrow4\rightarrow1
$$

The first layer has four hidden neurons, while the second layer produces the binary classification probability.

## 12.3 Compile the model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Here\:

- Optimizer: Adam
- Loss: Binary cross-entropy
- Metric: Accuracy

## 12.4 Train the model

```python
history = model.fit(
    X,
    y,
    epochs=1000,
    batch_size=4,
    verbose=0
)
```

## 12.5 Evaluate the model

```python
loss, accuracy = model.evaluate(X, y)

print("Loss:", loss)
print("Accuracy:", accuracy)
```

## 12.6 Make predictions

```python
predictions = model.predict(X)

print(predictions)

binary_predictions = (predictions >= 0.5).astype(int)

print(binary_predictions)
```

### Inspecting the model architecture

```python
model.summary()
```

This displays\:

- Layer names
- Output shapes
- Number of trainable parameters
- Total parameters

You should become comfortable reading `model.summary()` because it is one of the most useful ways to inspect neural network architectures.

---

# 13. Overfitting and Underfitting in MLP

MLPs can learn complex relationships, but they can also memorize training data.

This brings us to two important machine learning problems.

## 13.1 Underfitting

Underfitting occurs when the model cannot learn the underlying patterns in the training dataset.

Characteristics\:

- High training loss.
- High validation loss.
- Poor performance on training and testing data.

Possible causes\:

- Too few neurons.
- Too few hidden layers.
- Insufficient training.
- Inappropriate learning rate.
- Poor feature representation.

Solutions\:

- Increase model capacity.
- Train for more epochs when appropriate.
- Improve feature engineering.
- Tune hyperparameters.

## 13.2 Overfitting

Overfitting occurs when the model learns the training data too closely, including noise and irrelevant patterns.

Characteristics\:

- Low training loss.
- High validation loss.
- Good training performance but poor generalization.

### Techniques to prevent overfitting

**1\. L1 and L2 Regularization**

L2 regularization adds a penalty proportional to the squared weights.

$$
L_{\text{total}}=L_{\text{original}}+\lambda\sum_i w_i^2
$$

L1 regularization\:

$$
L_{\text{total}}=L_{\text{original}}+\lambda\sum_i|w_i|
$$

These discourage excessively large weights.

**2\. Dropout**

Dropout randomly disables a proportion of neurons during training.

For example\:

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(20,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

A dropout rate of 0.3 means approximately 30% of the eligible activations are randomly set to zero during each training step.

**3\. Early Stopping**

Stop training when validation performance stops improving.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    callbacks=[early_stop]
)
```

**4\. More Training Data**

More representative training data can improve generalization.

---

# 14. Hyperparameter Tuning in MLP

Hyperparameters are settings that we choose before or during model training rather than parameters directly learned through gradient descent.

Important MLP hyperparameters include\:

| Hyperparameter | Example values |
|--------|--------|
| Hidden layers | 1, 2, 3, 4 |
| Neurons | 16, 32, 64, 128 |
| Learning rate | 0.1, 0.01, 0.001 |
| Batch size | 16, 32, 64, 128 |
| Epochs | 50, 100, 200 |
| Activation | ReLU, Tanh |
| Optimizer | SGD, Adam |
| Dropout | 0.1, 0.2, 0.3, 0.5 |

There is no universally best MLP architecture.

You need to experiment using validation data and select the architecture that performs well on unseen samples.

One important distinction\:

- Weights and biases are learned parameters.
- Learning rate, batch size, number of layers, and neurons are hyperparameters.

---

# 15. Advantages and Disadvantages of MLP

## Advantages

1. Can learn nonlinear relationships.
2. Can solve complex classification and regression problems.
3. Supports multiple input and output features.
4. Can approximate a broad class of continuous functions under suitable conditions.
5. Can learn useful feature combinations automatically.
6. Relatively straightforward to implement.
7. Can serve as a foundation for more advanced neural network architectures.

## Disadvantages

1. Requires considerable training data for many complex tasks.
2. Can be computationally expensive.
3. Sensitive to feature scaling.
4. Can suffer from overfitting.
5. Hyperparameter tuning can be difficult.
6. Less interpretable than many traditional ML models.
7. Does not naturally exploit spatial structure in images or sequential structure in time-series data.
8. Deep networks can suffer from vanishing or exploding gradients.

---

# 16. MLP vs Single-Layer Perceptron

This is an important comparison because MLP is an extension of the perceptron architecture.

| Basis | Single-Layer Perceptron | Multi-Layer Perceptron |
|--------|--------|--------|
| Architecture | Input and output | Input, hidden and output |
| Hidden layers | None | One or more |
| Decision boundary | Linear | Can be nonlinear |
| Learning | Perceptron learning rule | Backpropagation and optimization |
| Activation | Traditionally step | ReLU, sigmoid, tanh, etc. |
| XOR problem | Cannot solve | Can solve |
| Complexity | Low | Higher |
| Applications | Simple classification | Complex classification and regression |

# 17. MLP vs Deep Neural Network

You will frequently encounter the terms ANN, MLP, and DNN.

They are related, but they are not interchangeable.

- **ANN\:** Broad term for artificial neural networks.
- **MLP\:** A feedforward, fully connected neural network architecture.
- **DNN\:** A neural network with multiple hidden layers.

An MLP with multiple hidden layers is a deep neural network.

For example\:

$$
Input\rightarrow H_1\rightarrow H_2\rightarrow H_3\rightarrow Output
$$

This is a deep MLP.

However, deep neural networks can also have other architectures, such as CNNs, RNNs, and Transformers.

---

# 18. Real-World Applications of MLP

MLPs are used in various machine learning applications.

| Application | Example |
|--------|--------|
| Financial prediction | Credit risk classification |
| Healthcare | Disease prediction using patient features |
| Banking | Fraud detection |
| Education | Student performance prediction |
| Business | Customer churn prediction |
| Manufacturing | Predicting equipment failures |
| Marketing | Customer purchase prediction |
| Computer vision | Classification of extracted image features |

MLPs are particularly useful when working with structured/tabular data containing numerical and categorical features.

For image-specific spatial learning, CNNs are generally more suitable. For language and sequential data, architectures such as RNNs and Transformers are more commonly used.

---

# 19. Important Concepts You Must Remember

Let's summarize the complete MLP learning process.

## MLP training pipeline

**1\. Input data**

Features passed into the network

**2\. Forward propagation**

Weighted sum + bias + activation

**3\. Prediction**

Output generated by the network

**4\. Loss calculation**

Measure prediction error

**5\. Backpropagation**

Calculate gradients using chain rule

**6\. Optimizer**

Update weights and biases

Repeat for multiple batches and epochs

The most important equations\:

| Operation | Equation |
|--------|--------|
| Weighted sum | $Z=WA+b$ |
| Activation | $A=f(Z)$ |
| MSE | $\frac{1}{n}\sum(y-\hat y)^2$ |
| Weight gradient | $\frac{\partial L}{\partial W}$ |
| Weight update | $W=W-\alpha dW$ |
| Bias update | $b=b-\alpha db$ |

# 20. What should you study next?

Now that you understand MLP, I recommend following this Deep Learning learning path\:

## Your next Deep Learning topics

1. **Perceptron and Perceptron Learning**
   
   (Foundation)
   
   Understand neurons, weights, bias, and learning rules.
2. **Multi-Layer Perceptron**
   
   (Current topic)
   
   Architecture, forward propagation, backpropagation, and optimization.
3. **Backpropagation and Gradient Descent – Advanced Mathematics**
   
   Derivatives, chain rule, computational graphs, and gradient calculations in matrix form.
4. **Weight Initialization**
   
   Xavier, He initialization, and why initialization matters.
5. **Vanishing and Exploding Gradients**
   
   Why gradients become unstable and how to solve these problems.
6. **Optimization Algorithms**
   
   SGD, Momentum, RMSprop, Adam, and AdamW in detail.
7. **Regularization in Deep Learning**
   
   Dropout, L1/L2, early stopping, and batch normalization.
8. **Convolutional Neural Networks (CNN)**
   
   Image classification, convolution, pooling, and feature extraction.
9. **Recurrent Neural Networks (RNN)**
   
   Sequential data and time-series learning.
10. **LSTM and GRU**
    
    Advanced sequence modeling.
11. **Transformers and Attention**
    
    The foundation of modern NLP and generative AI.

## Test your understanding

Try answering these questions before moving to the next topic.

**MLP knowledge check** (/5 answered)

**My recommendation\:** Before moving to CNN, become comfortable implementing an MLP from scratch using NumPy. In particular, understand matrix dimensions, forward propagation, the chain rule, and backpropagation. These concepts will make the rest of Deep Learning much easier.



---
