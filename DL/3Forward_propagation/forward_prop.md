
# Forward Propagation in Deep Learning (Detailed Study Notes)

Forward propagation is one of the most fundamental concepts in Deep Learning (DL). It is the process through which input data travels through a neural network, layer by layer, to produce an output or prediction.

When you give a neural network an input, the network performs mathematical operations using its weights, biases, and activation functions to generate a prediction.

For example\:

- Input: An image of a cat.
- Neural network: Processes the image through multiple layers.
- Output: Cat = 0.95, Dog = 0.05.

The process of generating these predictions is called **Forward Propagation**\.

Let's understand it from the fundamentals, including mathematical derivations, numerical examples, and Python implementation.

# 1. What is Forward Propagation?

Forward propagation (also called forward pass) is the process of passing input features through all the layers of a neural network to calculate the final output.

A typical neural network contains\:

1. Input layer
2. Hidden layers
3. Output layer

Each neuron performs three major operations\:

1. Receives inputs from the previous layer.
2. Calculates a weighted sum and adds a bias.
3. Applies an activation function.

The output of one layer becomes the input of the next layer.
![](nn.jpg)
The overall process can be represented as\:

$$
Input \rightarrow Weighted\ Sum \rightarrow
Activation \rightarrow Output
$$

This process continues until the network produces its final prediction.

# 2. Understanding the mathematical operations

Before understanding forward propagation in a complete network, we need to understand what happens inside an individual neuron.

## 2.1 Inputs (Features)

Suppose we want to predict whether a student will pass an examination based on\:

- $x_1$\: Hours studied
- $x_2$\: Hours slept

Our input vector is\:

$$
X =
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

Suppose\:

$$
X =
\begin{bmatrix}
5\\
8
\end{bmatrix}
$$

The input layer receives these values and passes them to the next layer.

Remember: The input layer generally does not perform trainable mathematical operations. It simply supplies the input features to the network.

## 2.2 Weights

Weights determine how important each input is when calculating a neuron's output.

Suppose\:

$$
W =
\begin{bmatrix}
0.6\\
0.3
\end{bmatrix}
$$

Here\:

- Weight 0.6 is associated with hours studied.
- Weight 0.3 is associated with hours slept.

A higher absolute weight generally means that an input has a stronger influence on the neuron's calculation, although the interpretation depends on other weights and the activation function.

Weights are learnable parameters. During training, the network adjusts them to reduce prediction errors.

## 2.3 Bias

Bias is an additional learnable parameter added to the weighted sum.

It allows the neuron to shift its activation independently of the input values.

Suppose\:

$$
b=0.5
$$

Without bias, the neuron would calculate only the weighted sum.

With bias\:

$$
z = w_1x_1+w_2x_2+b
$$

## 2.4 Weighted sum (Linear transformation)

This is the first mathematical operation performed by a neuron.

The formula is\:

$$
\boxed{z=\sum_{i=1}^{n}w_ix_i+b}
$$

For our example\:

$$
z=(5\times0.6)+(8\times0.3)+0.5
$$

$$
z=3+2.4+0.5
$$

$$
\boxed{z=5.9}
$$

This value is called the **pre-activation value** or weighted input.

Why pre-activation?

Because the activation function has not yet been applied.

## 2.5 Activation function

After calculating the weighted sum, we apply an activation function.

$$
a=f(z)
$$

Where\:

- $z$ \= weighted sum
- $f$ \= activation function
- $a$ \= activated output

Why do we need activation functions?

Without nonlinear activation functions, stacking multiple neural network layers would still result in a linear transformation.

For example\:

$$
Y=W_2(W_1X+b_1)+b_2
$$

This can be simplified into\:

$$
Y=W'X+b'
$$

Therefore, even a network with many layers would behave like a single linear model.

Activation functions introduce nonlinearity, allowing neural networks to learn complex patterns.

### Common activation functions

**1\. ReLU (Rectified Linear Unit)**

$$
f(x)=\max(0,x)
$$

- Negative values become zero.
- Positive values remain unchanged.
- Commonly used in hidden layers.

---

**2\. Sigmoid**

$$
f(x)=\frac{1}{1+e^{-x}}
$$

- Output ranges from 0 to 1.
- Commonly used for binary classification output.

---

**3\. Tanh**

$$
f(x)=\tanh(x)
$$

- Output ranges from -1 to 1.
- Zero-centered activation function.

---

**4\. Softmax**

$$
f(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

- Converts logits into probabilities.
- Commonly used for multiclass classification.

For our student example, let's use sigmoid.

$$
a=\frac{1}{1+e^{-5.9}}
$$

$$
\boxed{a\approx0.9973}
$$

If this neuron represents the probability of passing, its prediction is approximately 99.73%.

This completes the forward propagation of a single neuron.

---

# 3. Forward Propagation in a Multi-Layer Neural Network

Now let's move from a single neuron to a complete neural network.

Consider the following architecture\:

- Input layer: 2 neurons
- Hidden layer 1: 3 neurons
- Hidden layer 2: 2 neurons
- Output layer: 1 neuron

The information flows in this order\:

$$
X \rightarrow H_1 \rightarrow H_2 \rightarrow Y
$$

Each layer receives the previous layer's output and calculates its own output.

Let's understand the mathematical representation.

## 3.1 Input layer

Suppose we have two input features\:

$$
X =
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

For our numerical example\:

$$
X =
\begin{bmatrix}
2\\
3
\end{bmatrix}
$$

These values are passed to the first hidden layer.

## 3.2 First hidden layer

Suppose the first hidden layer contains three neurons.

Each neuron receives two inputs.

Therefore, we have six weights and three biases.

The weight matrix is\:

$$
W_1 =
\begin{bmatrix}
0.1 & 0.2\\
0.3 & 0.4\\
0.5 & 0.6
\end{bmatrix}
$$

Bias vector\:

$$
b_1 =
\begin{bmatrix}
0.1\\
0.2\\
0.3
\end{bmatrix}
$$

Notice the dimensions\:

$$
W_1: 3\times2
$$

$$
X: 2\times1
$$

$$
b_1: 3\times1
$$

We calculate the weighted sum\:

$$
Z_1=W_1X+b_1
$$

Substituting the values\:

$$
Z_1=
\begin{bmatrix}
0.1 & 0.2\\
0.3 & 0.4\\
0.5 & 0.6
\end{bmatrix}
\begin{bmatrix}
2\\
3
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
0.2\\
0.3
\end{bmatrix}
$$

Multiplying\:

$$
Z_1=
\begin{bmatrix}
0.2+0.6\\
0.6+1.2\\
1.0+1.8
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
0.2\\
0.3
\end{bmatrix}
$$

Therefore\:

$$
\boxed{
Z_1=
\begin{bmatrix}
0.9\\
2.0\\
3.1
\end{bmatrix}
}
$$

Now apply ReLU\:

$$
A_1=ReLU(Z_1)
$$

Since all values are positive\:

$$
\boxed{
A_1=
\begin{bmatrix}
0.9\\
2.0\\
3.1
\end{bmatrix}
}
$$

This is the output of the first hidden layer.

**Important\:** The output of the first hidden layer becomes the input of the second hidden layer.

## 3.3 Second hidden layer

Suppose the second hidden layer contains two neurons.

Its weight matrix is\:

$$
W_2=
\begin{bmatrix}
0.2 & 0.3 & 0.4\\
0.5 & 0.6 & 0.7
\end{bmatrix}
$$

Bias vector\:

$$
b_2=
\begin{bmatrix}
0.1\\
0.2
\end{bmatrix}
$$

Dimensions\:

$$
W_2:2\times3
$$

$$
A_1:3\times1
$$

$$
b_2:2\times1
$$

Calculate\:

$$
Z_2=W_2A_1+b_2
$$

Substitute\:

$$
Z_2=
\begin{bmatrix}
0.2&0.3&0.4\\
0.5&0.6&0.7
\end{bmatrix}
\begin{bmatrix}
0.9\\
2.0\\
3.1
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
0.2
\end{bmatrix}
$$

First neuron\:

$$
z_{2,1}=(0.2)(0.9)+(0.3)(2)+(0.4)(3.1)+0.1
$$

$$
=0.18+0.6+1.24+0.1
$$

$$
=2.12
$$

Second neuron\:

$$
z_{2,2}=(0.5)(0.9)+(0.6)(2)+(0.7)(3.1)+0.2
$$

$$
=0.45+1.2+2.17+0.2
$$

$$
=4.02
$$

Therefore\:

$$
Z_2=
\begin{bmatrix}
2.12\\
4.02
\end{bmatrix}
$$

Apply ReLU\:

$$
A_2=ReLU(Z_2)
$$

$$
\boxed{
A_2=
\begin{bmatrix}
2.12\\
4.02
\end{bmatrix}
}
$$

## 3.4 Output layer

Our output layer contains one neuron.

Suppose\:

$$
W_3=
\begin{bmatrix}
0.4&0.6
\end{bmatrix}
$$

$$
b_3=0.2
$$

Calculate the weighted sum\:

$$
Z_3=W_3A_2+b_3
$$

$$
Z_3=(0.4)(2.12)+(0.6)(4.02)+0.2
$$

$$
=0.848+2.412+0.2
$$

$$
\boxed{Z_3=3.46}
$$

Since we are doing binary classification, let's use sigmoid in the output layer.

$$
A_3=\sigma(Z_3)
$$

$$
A_3=\frac{1}{1+e^{-3.46}}
$$

$$
\boxed{A_3\approx0.9695}
$$

Our final prediction is approximately 96.95%.

If the threshold is 0.5\:

$$
\hat y=
\begin{cases}
1 & P\geq0.5\\
0 & P<0.5
\end{cases}
$$

Therefore\:

$$
\boxed{\hat y=1}
$$

The network predicts class 1.

---

# 4. General Mathematical Formula for Forward Propagation

Now let's generalize everything.

Suppose a neural network contains $L$ trainable layers.

For each layer $l$\:

### Step 1: Calculate the weighted sum

$$
\boxed{Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}}
$$

### Step 2: Apply the activation function

$$
\boxed{A^{[l]}=f^{[l]}(Z^{[l]})}
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $X$ | Input features |
| $W^{[l]}$ | Weight matrix of layer $l$ |
| $b^{[l]}$ | Bias vector of layer $l$ |
| $Z^{[l]}$ | Pre-activation values |
| $A^{[l]}$ | Activated output |
| $f^{[l]}$ | Activation function |
| $L$ | Number of trainable layers |

For the first layer\:

$$
A^{[0]}=X
$$

For the final layer\:

$$
\hat Y=A^{[L]}
$$

Therefore, the complete forward propagation process is\:

$$
X
\rightarrow Z^{[1]}
\rightarrow A^{[1]}
\rightarrow Z^{[2]}
\rightarrow A^{[2]}
\rightarrow \cdots
\rightarrow \hat Y
$$

# 5. Understanding Matrix Dimensions

This is extremely important when implementing neural networks using NumPy or PyTorch.

Suppose\:

- Input features = 4
- First hidden layer = 5 neurons
- Second hidden layer = 3 neurons
- Output layer = 2 neurons

The dimensions will be\:

| Parameter | Shape |
|--------|--------|
| Input $X$ | $4, 1) |
| \(W_1$ | $5, 4) |
| \(b_1$ | $5, 1) |
| \(Z_1$ | $5, 1) |
| \(A_1$ | $5, 1) |
| \(W_2$ | $3, 5) |
| \(b_2$ | $3, 1) |
| \(Z_2$ | $3, 1) |
| \(A_2$ | $3, 1) |
| \(W_3$ | $2, 3) |
| \(b_3$ | $2, 1) |
| \(Z_3$ | $2, 1) |
| \(A_3$ | $2, 1) |

**General rule\:**

If the previous layer contains \(n_{l-1}$ neurons and the current layer contains $n_l$ neurons\:

$$
W^{[l]}\in\mathbb R^{n_l\times n_{l-1}}
$$

$$
b^{[l]}\in\mathbb R^{n_l\times1}
$$

This guarantees that matrix multiplication is valid.

# 6. Forward Propagation with Multiple Training Examples (Batch Processing)

So far, we have considered only one training example.

However, in practical Deep Learning, we generally process multiple examples together using batches.

Suppose we have three students\:

| Student | Study hours | Sleep hours |
|--------|--------|--------|
| A | 2 | 3 |
| B | 4 | 5 |
| C | 6 | 7 |

Instead of processing each student separately, we can process all three together.

Our input matrix becomes\:

$$
X=
\begin{bmatrix}
2&4&6\\
3&5&7
\end{bmatrix}
$$

Here\:

- Rows represent features.
- Columns represent individual samples.

For the first hidden layer\:

$$
Z_1=W_1X+b_1
$$

Notice that the bias is broadcast across all three examples.

The resulting matrix contains the weighted sums for all neurons and all examples.
![](mat.jpg)
### Why is batch processing useful?

1. Efficient matrix operations.
2. Better hardware utilization, especially GPUs.
3. Faster training compared with processing every sample individually.
4. Allows vectorized computation using NumPy and PyTorch.

In practice, the common convention is\:

- Features-first representation: `(features, batch_size)` in mathematical derivations.
- Batch-first representation: `(batch_size, features)` in most PyTorch and scikit-learn workflows.

Both are valid, provided the matrix multiplication dimensions are consistent.

---

# 7. Implementing Forward Propagation Using Python and NumPy

Now let's implement everything ourselves without using TensorFlow or PyTorch.

We will use the exact same neural network from our numerical example.

Architecture\:

```text
Input layer:  2 neurons
Hidden 1:     3 neurons
Hidden 2:     2 neurons
Output layer: 1 neuron
```

### Step 1: Import NumPy

```python
import numpy as np
```

### Step 2: Define activation functions

```python
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Step 3: Define input data

```python
X = np.array([
    [2],
    [3]
])
```

### Step 4: Initialize weights and biases

For learning purposes, we are manually assigning weights and biases.

```python
# First hidden layer
W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

b1 = np.array([
    [0.1],
    [0.2],
    [0.3]
])

# Second hidden layer
W2 = np.array([
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7]
])

b2 = np.array([
    [0.1],
    [0.2]
])

# Output layer
W3 = np.array([
    [0.4, 0.6]
])

b3 = np.array([
    [0.2]
])
```

### Step 5: Perform forward propagation

```python
# First hidden layer
Z1 = W1 @ X + b1
A1 = relu(Z1)

# Second hidden layer
Z2 = W2 @ A1 + b2
A2 = relu(Z2)

# Output layer
Z3 = W3 @ A2 + b3
A3 = sigmoid(Z3)

print("Output:", A3)
```

Expected output\:

```text
Output: [[0.9695...]]
```

### Step 6: Complete implementation in one function

Let's organize the entire process into a function.

```python
def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]

    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # Hidden layer 1
    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    # Hidden layer 2
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    # Output layer
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    return A3
```

Parameters dictionary\:

```python
parameters = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2,
    "W3": W3,
    "b3": b3
}

prediction = forward_propagation(X, parameters)

print(prediction)
```

This is a basic implementation of forward propagation from scratch.

---

# 8. Forward Propagation in PyTorch

In real-world DL projects, we generally use frameworks such as PyTorch or TensorFlow.

Let's see how the same network can be implemented in PyTorch.

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
```

Create the model\:

```python
model = NeuralNetwork()
```

Input\:

```python
X = torch.tensor(
    [[2.0, 3.0]],
    dtype=torch.float32
)
```

Perform forward propagation\:

```python
prediction = model(X)

print(prediction)
```

Here is something important\:

```python
prediction = model(X)
```

Internally, PyTorch calls the `forward()` method and performs all the operations in sequence.

PyTorch automatically manages the gradients needed for backpropagation when gradient tracking is enabled.

---

# 9. Relationship Between Forward Propagation, Loss Function and Backpropagation

This is probably the most important connection to understand in Deep Learning.

Forward propagation alone does not train a neural network.

It only generates predictions.

To train the network, we need to calculate the error and update its parameters.

The complete training process looks like this\:

Let's understand each step.

## Step 1: Forward propagation

Pass the input through the network.

$$
\hat y=f(X,W,b)
$$

The network produces a prediction.

## Step 2: Calculate loss

Compare the predicted output with the actual output.

Suppose\:

$$
y=1
$$

$$
\hat y=0.9695
$$

For binary classification, we can use Binary Cross Entropy (BCE)\:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

Substituting\:

$$
L=-\log(0.9695)
$$

$$
L\approx0.031
$$

The loss is relatively small because the prediction is close to the actual value.

## Step 3: Backpropagation

Now the network needs to understand\:

- Which weights contributed to the prediction error?
- How much did each weight contribute?
- In which direction should each weight change?
- How much should each weight change?

Backpropagation uses the chain rule of calculus to calculate gradients.

For a weight\:

$$
\frac{\partial L}{\partial W}
$$

This represents how the loss changes when the weight changes.

## Step 4: Update weights

Using gradient descent\:

$$
\boxed{W_{new}=W_{old}-\alpha\frac{\partial L}{\partial W}}
$$

Where\:

- $W_{new}$ \= updated weight
- $W_{old}$ \= previous weight
- $\alpha$ \= learning rate
- $\frac{\partial L}{\partial W}$ \= gradient

The same principle applies to biases\:

$$
b_{new}=b_{old}-\alpha\frac{\partial L}{\partial b}
$$

## Step 5: Repeat

The updated weights are used in the next forward pass.

The process continues over multiple batches and epochs until the network learns a useful mapping between inputs and outputs.

---

# 10. Forward Propagation vs Backpropagation

These two concepts are closely related but perform different jobs.

| Forward Propagation | Backpropagation |
|--------|--------|
| Moves from input to output | Calculates gradients from output towards earlier layers |
| Generates predictions | Determines how parameters affect loss |
| Uses weights and biases | Uses derivatives and chain rule |
| Calculates activations | Calculates gradients |
| Computes loss when combined with a loss function | Uses loss to calculate gradients |
| Used during training and inference | Used during training |

One important distinction\:

**Forward propagation is used during both training and prediction (inference), whereas backpropagation is generally used only during training.**

# 11. Important Concepts You Should Understand

Before moving towards backpropagation, make sure you understand these concepts.

| Concept | What you should understand |
|--------|--------|
| Weights | How inputs influence neurons |
| Bias | Why additional learnable parameters are needed |
| Weighted sum | Matrix multiplication and addition |
| Activation function | Why nonlinear transformations are necessary |
| Pre-activation $Z$ | Output before activation |
| Activation $A$ | Output after activation |
| Matrix dimensions | How layers connect mathematically |
| Batch processing | How multiple examples are processed together |
| Output layer | How predictions are generated |
| Loss function | How prediction error is measured |

---

# 12. Practice Exercises

Try these exercises to check your understanding.

**Exercise 1: Single neuron calculation**

Given\:

$$
x_1=2,\quad x_2=3
$$

$$
w_1=0.5,\quad w_2=0.2
$$

$$
b=0.1
$$

Calculate the weighted sum $z$, then calculate the ReLU output.

1.1

1.7

2.1

2.7

---

**Exercise 2: Matrix dimensions**

A layer contains 4 input neurons and 3 output neurons. What should be the weight matrix shape?

4 × 3

3 × 4

4 × 4

3 × 3

---

**Exercise 3: Conceptual understanding**

What happens if we remove all nonlinear activation functions from a deep neural network?

The network becomes more powerful.

The entire network behaves like a linear transformation.

Forward propagation stops working.

Weights become unnecessary.

# 13. What Should You Learn Next?

Since you are currently learning Deep Learning after studying the Perceptron and MLP, I recommend following this sequence\:

1. **Forward Propagation**
   
   (Covered here)
   
   Understand how inputs travel through neural network layers.
2. **Loss Functions in Deep Learning**
   
   Understand MSE, MAE, Binary Cross Entropy, Categorical Cross Entropy, and why different tasks require different loss functions.
3. **Calculus for Deep Learning**
   
   Understand derivatives, partial derivatives, gradients, and the chain rule.
4. **Backpropagation in Detail**
   
   Understand how gradients are calculated layer by layer using the chain rule.
5. **Gradient Descent and Optimizers**
   
   Learn SGD, Mini-batch Gradient Descent, Momentum, RMSProp, and Adam.
6. **Training an MLP from Scratch**
   
   Combine forward propagation, loss calculation, backpropagation, and gradient descent in NumPy.

**My recommendation\:** Before jumping into backpropagation, make sure you can independently calculate $Z$, $A$, and the output of a multi-layer network using matrix multiplication. These are the foundations you will repeatedly use when learning how gradients flow backward through the network.

---
