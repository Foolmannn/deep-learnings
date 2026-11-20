# Activation Functions in Deep Learning (DL)

## 1. Introduction to Activation Functions

An **activation function** is a mathematical function applied to the output of a neuron in an artificial neural network (ANN) to determine whether and how strongly that neuron should be activated.

In simple terms, an activation function decides how much information a neuron should pass to the next layer of a neural network.

Activation functions introduce **non-linearity** into neural networks, allowing them to learn complex patterns and relationships in data.

### Why do we need activation functions?

Consider a neural network without activation functions.

A neuron calculates its output using\:

$$
z = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

Where\:

- $x_1,x_2,\ldots,x_n$ \= Input features
- $w_1,w_2,\ldots,w_n$ \= Weights
- $b$ \= Bias
- $z$ \= Weighted sum

If we do not apply an activation function, the output is simply\:

$$
a=z
$$

Even if we stack multiple layers, the network essentially performs a linear transformation.

For example\:

$$
y=W_2(W_1x+b_1)+b_2
$$

This can be simplified into another linear transformation\:

$$
y=Wx+b
$$

Therefore, adding more layers without activation functions does not allow the network to learn more complex non-linear relationships.

**Solution\:** Activation functions introduce non-linearity, enabling neural networks to learn complicated patterns such as image features, speech patterns, and relationships in natural language.

---

## 2. How Activation Functions Work

An activation function takes the weighted sum of inputs and transforms it into an output.

The general mathematical representation is\:

$$
a=f(z)
$$

Where\:

- $z$ \= Weighted sum of inputs
- $f$ \= Activation function
- $a$ \= Activated output

### Working process

**Input Features**

$x_1,x_2,x_3$

**Weighted Sum**

$\displaystyle z=\sum_{i=1}^{n}w_ix_i+b$

**Activation Function**

$\displaystyle a=f(z)$

**Output of Neuron**

Passed to the next layer

### Example

Suppose a neuron receives\:

- Input $x_1=2$
- Input $x_2=3$
- Weight $w_1=0.5$
- Weight $w_2=0.2$
- Bias $b=1$

First, calculate the weighted sum\:

$$
z=(2\times0.5)+(3\times0.2)+1
$$

$$
z=1+0.6+1=2.6
$$

Now, suppose we apply the ReLU activation function\:

$$
f(z)=\max(0,z)
$$

$$
f(2.6)=2.6
$$

Therefore, the final output of the neuron is **2.6**\.

---

# 3. Types of Activation Functions

Activation functions can be classified into several categories.

**Major categories**

**1\. Linear Activation Functions**

Linear

---

**2\. Non-Linear Activation Functions**

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- PReLU
- ELU
- SELU
- GELU
- Swish
- Mish

---

**3\. Output Activation Functions**

- Softmax
- Sigmoid (binary classification)
- Linear (regression)

Let's understand the important activation functions individually.

## 3.1 Linear Activation Function

The linear activation function produces an output directly proportional to its input.

### Mathematical formula

$$
f(x)=x
$$

Or more generally\:

$$
f(x)=ax
$$

Where $a$ is a constant.

### Characteristics

- Output can range from negative infinity to positive infinity.
- Does not introduce non-linearity.
- Simple and computationally inexpensive.
- Commonly used in the output layer of regression models.

### Graph
![](../../images/activation_func/linear_act.jpg)
### Advantages

1. Simple to implement.
2. Computationally efficient.
3. Suitable for predicting continuous numerical values.

### Disadvantages

1. Cannot learn complex non-linear relationships when used throughout a network.
2. Multiple linear layers collapse into a single linear transformation.
3. Not suitable as the only activation function in hidden layers of deep neural networks.

### Applications

- Linear regression.
- Predicting house prices.
- Predicting temperature.
- Predicting continuous numerical values.

---

## 3.2 Sigmoid Activation Function

The sigmoid activation function converts any real-valued input into a value between 0 and 1.

It produces an S-shaped curve.

### Mathematical formula

$$
f(x)=\frac{1}{1+e^{-x}}
$$

Where\:

- $x$ \= Input value.
- $e$ \= Euler's mathematical constant (approximately 2.718).

### Output range

$$
0<f(x)<1
$$

### Example

| Input (x) | Sigmoid output |
|--------|--------|
| \-5 | 0.0067 |
| \-2 | 0.1192 |
| \-1 | 0.2689 |
| 0 | 0.5 |
| 1 | 0.7311 |
| 2 | 0.8808 |
| 5 | 0.9933 |

### Graph
![](../../images/activation_func/sigmoid_act.jpg)


### Characteristics

1. Produces outputs between 0 and 1.
2. Has a smooth S-shaped curve.
3. Differentiable at every point.
4. Useful for representing probabilities.
5. Output is always positive.

### Derivative

The derivative of the sigmoid function is\:

$$
f'(x)=f(x)(1-f(x))
$$

Its maximum derivative is\:

$$
f'(x)\leq 0.25
$$

### Advantages

- Useful for binary classification.
- Produces probability-like outputs.
- Smooth and differentiable.
- Easy to implement.

### Disadvantages

**1\. Vanishing Gradient Problem**

For very large positive or negative inputs, the gradient approaches zero.

As a result, the earlier layers of a deep neural network may learn very slowly.

**2\. Not Zero-Centered**

The output is always positive, which can make optimization less efficient.

**3\. Computationally Expensive**

It requires exponential calculations compared with simpler activation functions such as ReLU.

### Applications

- Binary classification.
- Logistic regression.
- Binary output layers.
- Predicting whether an email is spam or not spam.

---

## 3.3 Tanh (Hyperbolic Tangent) Activation Function

The Tanh activation function is similar to sigmoid, but its output ranges from -1 to 1.

It produces an S-shaped curve centered around zero.

### Mathematical formula

$$
f(x)=\tanh(x)
$$

Another mathematical representation is\:

$$
f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

### Output range

$$
-1<f(x)<1
$$

### Example

| Input (x) | Tanh output |
|--------|--------|
| \-5 | \-0.9999 |
| \-2 | \-0.9640 |
| \-1 | \-0.7616 |
| 0 | 0 |
| 1 | 0.7616 |
| 2 | 0.9640 |
| 5 | 0.9999 |

### Graph
![](../../images/activation_func/tanh_act.jpg)
### Derivative

$$
f'(x)=1-\tanh^2(x)
$$

### Characteristics

- Output ranges between -1 and 1.
- Zero-centered activation function.
- Smooth and differentiable.
- Stronger gradients around zero compared with sigmoid.

### Advantages

1. Zero-centered output makes optimization easier in many cases.
2. Provides stronger gradients near zero than sigmoid.
3. Suitable for networks that need positive and negative activations.
4. Smooth and differentiable.

### Disadvantages

1. Suffers from the vanishing gradient problem.
2. Gradients become very small for large positive or negative inputs.
3. Requires exponential calculations.

### Applications

- Recurrent Neural Networks (RNNs).
- Hidden layers in some neural networks.
- Processing data where positive and negative activations are useful.

### Sigmoid vs Tanh

| Feature | Sigmoid | Tanh |
|--------|--------|--------|
| Output range | 0 to 1 | \-1 to 1 |
| Zero-centered | No | Yes |
| Shape | S-shaped | S-shaped |
| Vanishing gradient | Yes | Yes |
| Common use | Binary output | RNN hidden states |

---

## 3.4 ReLU (Rectified Linear Unit) Activation Function

ReLU is one of the most widely used activation functions in modern deep learning.

It outputs zero for negative input values and returns the input itself for positive values.

### Mathematical formula

$$
f(x)=\max(0,x)
$$

Or\:

$$
f(x)=
\begin{cases}
0 & x<0\\
x & x\geq0
\end{cases}
$$

### Example

| Input (x) | ReLU output |
|--------|--------|
| \-5 | 0 |
| \-2 | 0 |
| \-1 | 0 |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 5 | 5 |

### Graph
![](../../images/activation_func/relu_act.jpg)
### Derivative

$$
f'(x)=
\begin{cases}
0 & x<0\\
1 & x>0
\end{cases}
$$

At $x=0$, ReLU is not mathematically differentiable; implementations use a chosen subgradient, commonly zero.

### Characteristics

1. Produces zero for negative inputs.
2. Produces the same value for positive inputs.
3. Computationally efficient.
4. Helps reduce the vanishing gradient problem for positive inputs.
5. Produces sparse activations because negative values become zero.

### Advantages

- Simple and fast.
- Helps train deep neural networks efficiently.
- Reduces the vanishing gradient problem for positive activations.
- Computationally less expensive than sigmoid and tanh.
- Widely used in CNNs.

### Disadvantages

**1\. Dying ReLU Problem**

When a neuron continuously receives negative inputs, its output and gradient become zero.

As a result, the neuron may stop learning.

**2\. Not Zero-Centered**

ReLU outputs are non-negative.

**3\. Unbounded Positive Output**

It does not restrict large positive values.

### Applications

- Convolutional Neural Networks (CNNs).
- Image classification.
- Object detection.
- Hidden layers of deep neural networks.
- Computer vision applications.

---

## 3.5 Leaky ReLU Activation Function

Leaky ReLU is an improved version of ReLU designed to address the dying ReLU problem.

Unlike ReLU, which produces zero for all negative inputs, Leaky ReLU allows a small negative output.

### Mathematical formula

$$
f(x)=
\begin{cases}
x & x>0\\
\alpha x & x\leq0
\end{cases}
$$

Where\:

$$
\alpha=0.01
$$

is a commonly used negative slope.

### Example

| Input | ReLU | Leaky ReLU ($\alpha=0.01$\) |
|--------|--------|--------|
| \-5 | 0 | \-0.05 |
| \-2 | 0 | \-0.02 |
| \-1 | 0 | \-0.01 |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 5 | 5 | 5 |

### Graph
![](../../images/activation_func/leaky_relu_act.jpg)
### Characteristics

- Allows a small negative output.
- Reduces the dying ReLU problem.
- Computationally efficient.
- Differentiable at most points, with a slope change at zero.

### Advantages

1. Prevents the gradient from becoming zero for negative inputs.
2. Helps inactive neurons continue learning.
3. Computationally inexpensive.
4. Similar efficiency to ReLU.

### Disadvantages

1. The negative slope parameter must be selected.
2. Does not always perform better than standard ReLU.
3. Still produces unbounded positive outputs.

### Applications

- Deep neural networks.
- CNN architectures.
- Image processing.
- Generative Adversarial Networks (GANs), in some architectures.

---

## 3.6 PReLU (Parametric ReLU) Activation Function

PReLU is an extension of Leaky ReLU.

The main difference is that the negative slope is not fixed. Instead, it is a learnable parameter that is updated during training.

### Mathematical formula

$$
f(x)=
\begin{cases}
x & x>0\\
ax & x\leq0
\end{cases}
$$

Where $a$ is a learnable parameter.

### Difference between Leaky ReLU and PReLU

| Feature | Leaky ReLU | PReLU |
|--------|--------|--------|
| Negative slope | Fixed | Learnable |
| Parameter | Manually selected | Learned during training |
| Flexibility | Lower | Higher |
| Complexity | Lower | Slightly higher |

### Advantages

- Learns an appropriate negative slope.
- Reduces the dying ReLU problem.
- Provides additional flexibility.

### Disadvantages

- Introduces additional trainable parameters.
- May overfit in some situations.
- Requires additional training computation.

### Applications

- Deep CNNs.
- Image recognition.
- Computer vision architectures.

---

## 3.7 ELU (Exponential Linear Unit) Activation Function

ELU is another ReLU-based activation function that produces negative values for negative inputs using an exponential function.

It attempts to improve learning by allowing negative outputs and making the mean activation closer to zero.

### Mathematical formula

$$
f(x)=
\begin{cases}
x & x>0\\
\alpha(e^x-1) & x\leq0
\end{cases}
$$

Where $\alpha$ controls the negative saturation value.

### Characteristics

- Positive inputs are unchanged.
- Negative inputs produce smooth negative values.
- Negative outputs approach $-\alpha$\.
- Differentiable everywhere.

### Graph
![](../../images/activation_func/elu_act.jpg)
### Advantages

1. Allows negative outputs.
2. Helps reduce the dying ReLU problem.
3. Can produce mean activations closer to zero.
4. Smooth for negative inputs.

### Disadvantages

1. More computationally expensive than ReLU.
2. Requires exponential calculations.
3. Requires selecting the alpha parameter.

### Applications

- Deep neural networks.
- CNN architectures.
- Image classification tasks.

---

## 3.8 SELU (Scaled Exponential Linear Unit) Activation Function

SELU is a self-normalizing activation function designed to help neural networks maintain stable activation distributions during training under specific architectural and initialization assumptions.

It is based on ELU but includes scaling factors.

### Mathematical formula

$$
f(x)=\lambda
\begin{cases}
x & x>0\\
\alpha(e^x-1) & x\leq0
\end{cases}
$$

Where commonly\:

$$
\alpha\approx1.6733
$$

$$
\lambda\approx1.0507
$$

### Characteristics

- Helps maintain mean and variance of activations.
- Supports self-normalizing neural networks under suitable conditions.
- Uses exponential calculations.

### Advantages

1. Helps stabilize activation distributions.
2. Can reduce the need for explicit normalization layers in certain architectures.
3. Supports deep feedforward networks under its assumptions.

### Disadvantages

1. Requires specific initialization, typically LeCun normal.
2. Works best with suitable network architectures, such as AlphaDropout.
3. Not universally suitable for every neural network.

### Applications

- Self-normalizing neural networks.
- Deep feedforward networks.
- Some tabular learning architectures.

---

## 3.9 Softmax Activation Function

Softmax is commonly used in the output layer of a multi-class classification neural network.

It converts a vector of raw scores (logits) into probabilities that sum to 1.

### Mathematical formula

$$
f(x_i)=\frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
$$

Where\:

- $x_i$ \= Score of the current class.
- $n$ \= Total number of classes.
- $e^{x_i}$ \= Exponential of the current score.

### Example

Suppose a neural network predicts three classes\:

- Cat = 2.0
- Dog = 1.0
- Bird = 0.1

After applying Softmax, the approximate outputs are\:

| Class | Probability |
|--------|--------|
| Cat | 0.659 |
| Dog | 0.242 |
| Bird | 0.099 |
| **Total** | **1.000** |
![](../../images/activation_func/softmax_act.jpg)
### Characteristics

1. Converts logits into probability distributions.
2. All outputs are between 0 and 1.
3. The sum of all probabilities is 1.
4. Higher scores generally receive higher probabilities.
5. Commonly used with categorical cross-entropy loss.

### Advantages

- Useful for multi-class classification.
- Produces interpretable probability distributions.
- Helps identify the most likely class.
- Differentiable and suitable for gradient-based learning.

### Disadvantages

1. Sensitive to large differences between logits, resulting in highly concentrated probabilities.
2. Requires more computation than simple threshold-based outputs.
3. Probabilities may not be well calibrated.
4. Not generally used in hidden layers.

### Applications

- Image classification.
- Handwritten digit recognition.
- Multi-class text classification.
- Object classification.

---

## 3.10 GELU (Gaussian Error Linear Unit)

GELU is a smooth activation function that combines input values with a probability-like weighting based on the Gaussian distribution.

It is widely used in modern Transformer architectures.

### Mathematical formula

$$
f(x)=x\Phi(x)
$$

Where $\Phi(x)$ represents the cumulative distribution function of the standard normal distribution.

An approximate formula is\:

$$
f(x)\approx\frac{x}{2}
\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)\right)\right]
$$

### Characteristics

- Smooth and non-linear.
- Allows small negative outputs.
- Does not sharply cut off all negative values like ReLU.
- Commonly used in Transformer feedforward layers.

### Advantages

1. Smooth activation function.
2. Works well in many Transformer architectures.
3. Provides flexible activation behavior.
4. Can improve performance in some deep learning tasks.

### Disadvantages

1. More computationally expensive than ReLU.
2. Uses more complex mathematical calculations.
3. Does not guarantee better performance in every architecture.

### Applications

- Transformer models.
- BERT.
- Natural Language Processing.
- Deep neural networks.

---

## 3.11 Swish Activation Function

Swish is a smooth, non-monotonic activation function developed by researchers at Google.

It combines the input with its sigmoid value.

### Mathematical formula

$$
f(x)=x\cdot\sigma(x)
$$

Where\:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

### Characteristics

- Smooth and non-monotonic.
- Allows small negative outputs.
- Does not completely suppress negative values.
- Can improve performance in certain deep architectures.

### Advantages

1. Smooth gradient behavior.
2. Allows negative values.
3. Performs well in some deep neural networks.
4. Can outperform ReLU in certain tasks.

### Disadvantages

1. More computationally expensive than ReLU.
2. Requires sigmoid computation.
3. Performance improvement is architecture-dependent.

### Applications

- EfficientNet.
- Deep CNNs.
- Image classification.
- Computer vision.

---

## 3.12 Mish Activation Function

Mish is a smooth, non-monotonic activation function that combines the input with the hyperbolic tangent of Softplus.

### Mathematical formula

$$
f(x)=x\tanh(\ln(1+e^x))
$$

Where\:

$$
\ln(1+e^x)
$$

is the Softplus function.

### Characteristics

- Smooth and non-monotonic.
- Allows negative outputs.
- Unbounded above.
- Provides smooth gradient behavior.

### Advantages

1. Smooth activation and gradient behavior.
2. Allows information from negative inputs.
3. Can perform well in some computer vision models.

### Disadvantages

1. More computationally expensive than ReLU.
2. Requires additional mathematical calculations.
3. Not always better than simpler alternatives.

### Applications

- Object detection.
- Image classification.
- Deep CNNs.

---

# 4. Comparison of Important Activation Functions

| Activation | Output range | Zero-centered | Common application |
|--------|--------|--------|--------|
| Linear | $(-\infty,\infty)$ | Yes | Regression output |
| Sigmoid | $0, 1) | No | Binary classification |
| Tanh | \(-1, 1) | Yes | RNNs |
| ReLU | \[0, ∞) | No | CNN hidden layers |
| Leaky ReLU | \((-\infty,\infty)$ | No | Deep networks |
| PReLU | $(-\infty,\infty)$ | No | CNNs |
| ELU | $(-\alpha,\infty)$ | No | Deep networks |
| SELU | Approximately $(-\lambda\alpha,\infty)$ | No | Self-normalizing networks |
| Softmax | $0, 1) | No | Multi-class output |
| GELU | Approximately \((-\infty,\infty)$ | No | Transformers |
| Swish | Approximately $(-\infty,\infty)$ | No | EfficientNet |
| Mish | Approximately $(-\infty,\infty)$ | No | Object detection |

---

# 5. Problems Associated with Activation Functions

Activation functions are essential for deep learning, but some of them introduce challenges during training.

### 5.1 Vanishing Gradient Problem

The vanishing gradient problem occurs when gradients become extremely small during backpropagation.

It is commonly associated with sigmoid and tanh activation functions when they operate in their saturated regions.

**How it happens\:**

1. During backpropagation, gradients are calculated using derivatives of activation functions.
2. Sigmoid and tanh derivatives become very small for large positive or negative inputs.
3. When these small gradients are multiplied across many layers, the gradients become increasingly smaller.
4. Earlier layers receive very small updates.
5. Consequently, learning becomes slow or may stop.

**Solutions\:**

- Use ReLU or its variants.
- Apply suitable weight initialization.
- Use batch normalization where appropriate.
- Use residual connections.

### 5.2 Dying ReLU Problem

Dying ReLU occurs when a ReLU neuron continuously receives negative inputs.

Since ReLU produces zero for negative inputs, its gradient also becomes zero.

As a result, the neuron may stop learning.

**Solutions\:**

- Use Leaky ReLU.
- Use PReLU.
- Use ELU.
- Apply suitable weight initialization and learning rates.

### 5.3 Exploding Gradient Problem

Exploding gradients occur when gradients become excessively large during backpropagation.

This can lead to unstable weight updates and training failure.

**Solutions\:**

- Gradient clipping.
- Proper weight initialization.
- Suitable learning rates.
- Batch normalization or other normalization methods where appropriate.

---

# 6. Implementation of Activation Functions in Python

Let's understand how to implement activation functions using TensorFlow/Keras and PyTorch.

## 6.1 Using TensorFlow / Keras

First, import the required library\:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

### Example 1: ReLU

```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.summary()
```

Here\:

- First hidden layer uses ReLU.
- Second hidden layer uses ReLU.
- Output layer uses linear activation by default.

### Example 2: Sigmoid

```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])
```

Used for binary classification.

### Example 3: Tanh

```python
model = models.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(10,)),
    layers.Dense(1)
])
```

### Example 4: Softmax

```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(3, activation='softmax')
])
```

Used for classification involving three classes.

### Example 5: Leaky ReLU

```python
model = models.Sequential([
    layers.Dense(64, input_shape=(10,)),
    layers.LeakyReLU(negative_slope=0.01),
    layers.Dense(3, activation='softmax')
])
```

---

## 6.2 Using PyTorch

First, import PyTorch\:

```python
import torch
import torch.nn as nn
```

### Example 1: ReLU

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.output(x)
```

### Example 2: Sigmoid

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

### Example 3: Tanh

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)
```

### Example 4: Softmax

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
    nn.Softmax(dim=1)
)
```

Note: When using `nn.CrossEntropyLoss()` in PyTorch, do not add Softmax to the model's output layer. `CrossEntropyLoss` applies the required log-softmax operation internally.

### Example 5: Leaky ReLU

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.LeakyReLU(0.01),
    nn.Linear(64, 1)
)
```

---

# 7. Choosing the Right Activation Function

Different activation functions are suitable for different tasks.

| Situation | Recommended activation |
|--------|--------|
| Hidden layers in general deep networks | ReLU |
| Binary classification output | Sigmoid |
| Multi-class classification output | Softmax |
| Regression output | Linear |
| RNN hidden states | Tanh (traditional RNNs) |
| Transformer feedforward layers | GELU or ReLU variants |
| Dying ReLU problem | Leaky ReLU or PReLU |
| Self-normalizing networks | SELU |

### Typical neural network architecture

**Input Layer**

Input features

**Hidden Layer 1**

(ReLU)

**Hidden Layer 2**

(ReLU)

**Output Layer**

(Sigmoid / Softmax / Linear)

The output activation depends on the problem\:

- Binary classification → Sigmoid.
- Multi-class classification → Softmax.
- Regression → Linear.

---

# 8. Important Differences to Remember

| Sigmoid vs ReLU | Explanation |
|--------|--------|
| Output | Sigmoid: 0 to 1; ReLU: 0 to infinity |
| Computation | Sigmoid is more expensive |
| Gradient | Sigmoid can suffer from vanishing gradients |
| Negative inputs | Sigmoid gives positive values; ReLU gives zero |
| Usage | Sigmoid for binary output; ReLU for hidden layers |

| ReLU vs Leaky ReLU | Explanation |
|--------|--------|
| Negative inputs | ReLU outputs zero |
| Negative inputs | Leaky ReLU outputs a small negative value |
| Gradient | ReLU gradient is zero for negative inputs |
| Gradient | Leaky ReLU has a small negative gradient |
| Main difference | Leaky ReLU helps address dying ReLU |

---

# 9. Conclusion

Activation functions are one of the most important components of deep neural networks. They introduce non-linearity, allowing neural networks to learn complex patterns and relationships from data.

Different activation functions serve different purposes\:

1. **Sigmoid\:** Used mainly for binary classification outputs.
2. **Tanh\:** Produces zero-centered outputs and is used in traditional RNNs.
3. **ReLU\:** A popular choice for hidden layers because of its simplicity and computational efficiency.
4. **Leaky ReLU\:** Helps address the dying ReLU problem.
5. **Softmax\:** Converts logits into a probability distribution for multi-class classification.
6. **GELU and Swish\:** Modern activation functions used in advanced deep learning architectures.

The selection of an appropriate activation function can significantly affect the learning speed, stability, and performance of a neural network.

---



