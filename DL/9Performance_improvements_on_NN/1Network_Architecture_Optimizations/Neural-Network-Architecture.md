
## **2. Network Architecture Optimization**

Improving the structure and design of the neural network.

1. **Choosing the Number of Layers:** Determining the appropriate network depth.
2. **Choosing the Number of Neurons:** Finding the optimal number of neurons in each layer.
3. **Activation Function Selection:** Using ReLU, Leaky ReLU, GELU, etc.
4. **Weight Initialization:** Techniques such as Xavier and He initialization.
5. **Batch Normalization:** Normalizing layer activations to improve training stability.

---

# 2. Network Architecture Optimization

## 1. Introduction

Network Architecture Optimization is the process of improving the structure and design of a neural network to achieve better performance, faster training, and improved generalization.

The architecture of a neural network determines how information flows from the input layer through hidden layers to the output layer.

A neural network's performance depends not only on the training data and optimization algorithm but also on how the network is designed.

For example, a network with too few layers may fail to learn complex patterns, while a network with too many layers may become computationally expensive and difficult to train.

**Main objectives of Network Architecture Optimization\:**

- Improve prediction accuracy.
- Reduce overfitting and underfitting.
- Improve training speed and stability.
- Learn complex patterns efficiently.
- Reduce unnecessary computational requirements.

There are five major techniques\:

1. Choosing the Number of Layers.
2. Choosing the Number of Neurons.
3. Activation Function Selection.
4. Weight Initialization.
5. Batch Normalization.

---

# 2.1 Choosing the Number of Layers

## What is Network Depth?

Network depth refers to the number of layers in a neural network, particularly its hidden layers.

A neural network consists of\:

- Input layer
- One or more hidden layers
- Output layer

The input layer receives the data, hidden layers learn patterns and representations, and the output layer produces the final prediction.

### A. Shallow Neural Network

A shallow neural network contains one or a small number of hidden layers.

Example\:

```text
Input Layer
     |
Hidden Layer
     |
Output Layer
```

It is suitable for relatively simple problems where patterns are not extremely complex.

**Advantages\:**

- Requires less computational power.
- Faster to train.
- Easier to understand and implement.
- Often sufficient for simple datasets.

**Disadvantages\:**

- May struggle with complex patterns.
- Has limited hierarchical feature-learning capability.
- May underfit complicated problems.

### B. Deep Neural Network

A deep neural network contains multiple hidden layers.

Example\:

```text
Input Layer
     |
Hidden Layer 1
     |
Hidden Layer 2
     |
Hidden Layer 3
     |
Hidden Layer 4
     |
Output Layer
```

Each hidden layer can learn different levels of representation.

For example, in image recognition\:

- Early layers learn simple features such as edges and lines.
- Middle layers learn textures and shapes.
- Deeper layers learn complex structures such as eyes, faces, or objects.

This hierarchical learning allows deep networks to solve complex tasks.

**Advantages\:**

- Learns complex relationships.
- Extracts hierarchical features.
- Can achieve excellent performance on large and complex datasets.
- Useful for image recognition, speech processing, and NLP.

**Disadvantages\:**

- Requires more computational resources.
- Takes longer to train.
- Can overfit when training data is insufficient.
- May suffer from vanishing or exploding gradients.

### C. How to Choose the Number of Layers?

There is no universal rule for selecting the correct number of hidden layers.

The choice depends on\:

1. **Problem complexity\:** Complex tasks may require deeper networks.
2. **Dataset size\:** Larger datasets can often support more complex architectures.
3. **Computational resources\:** Deeper networks require more memory and computation.
4. **Training performance\:** Training and validation performance help identify whether the network is too simple or too complex.
5. **Experimentation\:** Different architectures should be tested and compared.

| Problem | Possible architecture |
|--------|--------|
| Simple classification | 1–2 hidden layers |
| Tabular prediction | A few hidden layers |
| Image recognition | Deep CNN |
| Language processing | Transformer or other deep sequence architecture |
| Complex pattern recognition | Deep neural network |

These are illustrative choices, not fixed requirements.

### D. Underfitting and Overfitting

Choosing the number of layers affects the model's ability to learn.

**Underfitting\:** The network is too simple to learn the underlying patterns.

- High training error.
- High validation error.

**Overfitting\:** The network learns training data too closely, including noise.

- Very low training error.
- High validation error.

The objective is to find a suitable architecture that learns important patterns without memorizing the training data.

---

# 2.2 Choosing the Number of Neurons

## What are Neurons?

A neuron is the basic computational unit of a neural network.

It receives inputs, multiplies them by weights, adds a bias, and applies an activation function.

The mathematical representation is\:

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

$$
a=f(z)
$$

Where\:

- $x_i$ \= Input values.
- $w_i$ \= Weights.
- $b$ \= Bias.
- $f$ \= Activation function.
- $a$ \= Neuron output.

The number of neurons in a layer determines how many computational units are available to learn representations at that stage.

### A. Too Few Neurons

If a hidden layer contains too few neurons, the network may not have enough capacity to learn the patterns in the data.

Consequences\:

- Underfitting.
- Poor prediction accuracy.
- Inability to represent complex relationships.

### B. Too Many Neurons

If a network contains too many neurons, it may have excessive learning capacity.

Consequences\:

- Increased computational cost.
- More memory requirements.
- Greater risk of overfitting.
- Longer training time.

Therefore, the objective is to select enough neurons to learn the important patterns without unnecessarily increasing model complexity.

### C. Methods for Choosing Neurons

**1\. Trial and Error**

Start with a reasonable number of neurons and experiment with different configurations.

For example\:

```text
Model A: 16 neurons
Model B: 32 neurons
Model C: 64 neurons
Model D: 128 neurons
```

Compare their training and validation performance.

**2\. Gradually Decreasing Neurons**

A network may use more neurons in earlier hidden layers and fewer in later layers.

Example\:

```text
Input: 20 features

Hidden Layer 1: 128 neurons
Hidden Layer 2: 64 neurons
Hidden Layer 3: 32 neurons

Output: 1 neuron
```

This creates a funnel-shaped architecture.

However, decreasing neuron counts is not mandatory. Some architectures use constant-width layers, while others increase their width.

**3\. Hyperparameter Tuning**

Try different neuron configurations systematically using\:

- Grid Search.
- Random Search.
- Bayesian Optimization.

**4\. Monitor Validation Performance**

Use validation loss and relevant validation metrics to determine whether increasing neurons actually improves generalization.

### D. Example Using Python (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation="relu", input_shape=(20,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.summary()
```

Here\:

- Input contains 20 features.
- First hidden layer contains 128 neurons.
- Second hidden layer contains 64 neurons.
- Third hidden layer contains 32 neurons.
- Output layer contains one neuron for binary classification.

---

# 2.3 Activation Function Selection

## What is an Activation Function?

An activation function is a mathematical function applied to the output of a neuron.

It introduces non-linearity into a neural network, allowing it to learn complex relationships between inputs and outputs.

Recall the neuron equation\:

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

The activation function produces\:

$$
a=f(z)
$$

Without suitable nonlinear activation functions, stacking multiple layers of ordinary linear transformations is equivalent to a single linear transformation.

Therefore, the network would be limited in its ability to learn nonlinear patterns.

## A. Linear Activation Function

The linear activation function returns the input without modification.

$$
f(x)=x
$$

**Linear activation**

$y=x$

$$
f(x)=x
$$

**Advantages\:**

- Simple.
- Useful in regression output layers.

**Disadvantages\:**

- Does not introduce non-linearity.
- Not suitable as the only activation function in a deep hidden network.

## B. Sigmoid Activation Function

The sigmoid function converts any real-valued input into a value between 0 and 1.

Formula\:

$$
f(x)=\frac{1}{1+e^{-x}}
$$

**Sigmoid activation**

$y=\frac{1}{1+e^{-x}}$

$$
0<f(x)<1
$$

**Characteristics\:**

- Output range is between 0 and 1.
- Produces smooth output.
- Often used for binary classification output layers.

**Advantages\:**

- Useful for probability-like outputs.
- Differentiable.
- Easy to interpret.

**Disadvantages\:**

- Suffers from the vanishing gradient problem in saturated regions.
- Output is not zero-centered.
- Can slow down training in deep networks.

## C. Tanh Activation Function

Tanh stands for hyperbolic tangent.

Formula\:

$$
f(x)=\tanh(x)
$$

$$
f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

**Tanh activation**

$y=\tanh(x)$

$$
-1<f(x)<1
$$

**Characteristics\:**

- Output range is between -1 and +1.
- Zero-centered.
- Smooth and differentiable.

**Advantages\:**

- Zero-centered output.
- Often preferable to sigmoid in some hidden-layer settings.

**Disadvantages\:**

- Can suffer from vanishing gradients.
- Saturates for large positive and negative inputs.

## D. ReLU (Rectified Linear Unit)

ReLU is one of the most commonly used activation functions in deep neural networks.

Formula\:

$$
f(x)=\max(0,x)
$$

This means\:

$$
f(x)=
\begin{cases}
0 & x<0\\
x & x\geq 0
\end{cases}
$$

**ReLU activation**

$y=\max(0,x)$

**Characteristics\:**

- Negative inputs produce zero.
- Positive inputs pass through unchanged.
- Computationally efficient.

**Advantages\:**

- Helps reduce the vanishing gradient problem for positive activations.
- Computationally simple.
- Usually trains faster than sigmoid and tanh in many deep networks.

**Disadvantages\:**

- Can suffer from the dying ReLU problem.
- Negative inputs have zero gradient.

### What is the Dying ReLU Problem?

When a ReLU neuron receives negative inputs, its output and gradient are zero.

If a neuron remains in this region during training, its weights may stop updating through that activation path.

Such a neuron is called a dead or dying ReLU neuron.

## E. Leaky ReLU

Leaky ReLU is a modified version of ReLU that allows a small negative output when the input is negative.

Formula\:

$$
f(x)=
\begin{cases}
\alpha x & x<0\\
x & x\geq0
\end{cases}
$$

Here, $\alpha$ is a small positive constant, often 0.01.

**Leaky ReLU activation**

$y=\max(0.01x,x)$

**Advantages\:**

- Reduces the risk of dying ReLU neurons.
- Allows gradients to flow for negative inputs.
- Simple to implement.

**Disadvantages\:**

- The negative slope must be selected.
- It does not guarantee that dead neurons or training problems will disappear.

## F. GELU (Gaussian Error Linear Unit)

GELU is an activation function commonly used in modern deep learning architectures, particularly Transformers.

Formula\:

$$
GELU(x)=x\Phi(x)
$$

Where $\Phi(x)$ represents the cumulative distribution function of the standard normal distribution.

An approximate form is\:

$$
GELU(x)\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right)
\right]
$$

**GELU activation**

$y=\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right]$

**Characteristics\:**

- Smooth activation function.
- Allows negative values to contribute gradually.
- Commonly used in Transformer-based models.

**Advantages\:**

- Smooth nonlinear behavior.
- Effective in many modern deep learning architectures.
- Can provide good performance in large neural networks.

**Disadvantages\:**

- More computationally expensive than ReLU.
- Does not necessarily outperform ReLU in every task.

## G. Softmax Activation Function

Softmax converts a vector of raw scores into a probability distribution.

Formula\:

$$
P(y_i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Where\:

- $z_i$ \= Score for class $i$\.
- $K$ \= Number of classes.

Example\:

Suppose a network produces these raw scores\:

```text
Cat   = 2.0
Dog   = 1.0
Bird  = 0.5
```

After applying softmax, the output becomes approximately\:

```text
Cat   = 0.629
Dog   = 0.231
Bird  = 0.140
```

The probabilities sum to 1.

**Advantages\:**

- Useful for multiclass classification.
- Produces normalized class probabilities.
- Allows selection of the most probable class.

**Disadvantages\:**

- Standard softmax assumes mutually exclusive classes.
- It is not suitable by itself for independent multilabel predictions.

## H. Comparison of Activation Functions

| Activation | Output range | Common usage |
|--------|--------|--------|
| Linear | $(-\infty,+\infty)$ | Regression output |
| Sigmoid | $0, 1) | Binary classification output |
| Tanh | \(-1, 1) | Some hidden layers and recurrent networks |
| ReLU | $$0, +∞) | Hidden layers |
| Leaky ReLU | \((-\infty,+\infty)$ | Hidden layers |
| GELU | Approximately $(-\infty,+\infty)$ | Transformers |
| Softmax | $0, 1), sum = 1 | Multiclass output |

### How to Choose an Activation Function?

A practical starting point\:

- Hidden layers in ordinary MLPs: ReLU.
- When dead ReLU neurons are a concern: Leaky ReLU.
- Transformer architectures: GELU is a common choice.
- Binary classification output: Sigmoid.
- Multiclass classification output: Softmax.
- Regression output: Linear.

---

# 2.4 Weight Initialization

## What is Weight Initialization?

Weight initialization is the process of assigning initial values to the weights of a neural network before training begins.

Recall the neuron equation\:

\[
z=\sum_{i=1}^{n}w_ix_i+b
$$

At the beginning of training, the network does not know the correct weights.

The weights are initialized using a particular strategy and then updated through backpropagation and an optimizer such as gradient descent.

Proper weight initialization helps the network learn efficiently and reduces problems such as vanishing and exploding activations or gradients.

## A. Why is Weight Initialization Important?

Consider a network with several neurons.

If all neurons in the same layer start with exactly the same weights, they will receive the same gradients and continue learning the same features.

This is known as the **symmetry problem**\.

Therefore, weights generally need to be initialized differently.

Poor initialization can also cause\:

- Extremely small activations.
- Extremely large activations.
- Vanishing gradients.
- Exploding gradients.
- Slow convergence.
- Unstable training.

## B. Zero Initialization

In zero initialization, all weights are initialized to zero.

$$
W=0
$$

Although this appears simple, it is not suitable for initializing all weights in hidden layers.

Why?

If every neuron has the same initial weights, they produce the same outputs and receive the same updates. They cannot learn different features effectively.

This is called the symmetry problem.

**Note\:** Biases are commonly initialized to zero because identical bias initialization does not create the same symmetry issue when weights are initialized differently.

## C. Random Initialization

In random initialization, weights are assigned small random values.

For example\:

$$
W\sim N(0,0.01^2)
$$

Here, weights are sampled from a normal distribution with mean zero and standard deviation 0.01.

This breaks symmetry between neurons.

However, choosing an arbitrary random scale can still cause vanishing or exploding activations in deep networks.

This leads to more advanced initialization methods.

## D. Xavier Initialization (Glorot Initialization)

Xavier initialization is designed to keep the variance of activations reasonably stable across layers.

It is commonly used with sigmoid and tanh activation functions.

The objective is to prevent signals from becoming extremely small or large as they pass through the network.

For a uniform distribution\:

$$
W\sim U\left(
-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}
\right)
$$

Where\:

- \(n_{in}$ \= Number of input neurons.
- $n_{out}$ \= Number of output neurons.

For a normal distribution\:

$$
W\sim N\left(0,\frac{2}{n_{in}+n_{out}}\right)
$$

Here, the second parameter is the variance.

**Example\:**

Suppose\:

$$
n_{in}=100,\quad n_{out}=50
$$

For Xavier uniform initialization\:

$$
Limit=\sqrt{\frac{6}{100+50}}
$$

$$
Limit=\sqrt{0.04}=0.2
$$

Therefore\:

$$
W\sim U(-0.2,0.2)
$$

**Advantages\:**

- Helps maintain activation variance.
- Reduces the likelihood of vanishing or exploding signals.
- Works well with tanh and sigmoid in appropriate settings.

**Disadvantages\:**

- Not generally the preferred initialization for ReLU networks.
- Does not eliminate all gradient-related problems.

## E. He Initialization (Kaiming Initialization)

He initialization is specifically designed for ReLU and ReLU-like activation functions.

ReLU sets negative activations to zero, which changes the variance of signals passing through the network.

He initialization accounts for this behavior.

For a normal distribution\:

$$
W\sim N\left(0,\frac{2}{n_{in}}\right)
$$

For a uniform distribution\:

$$
W\sim U\left(
-\sqrt{\frac{6}{n_{in}}},
\sqrt{\frac{6}{n_{in}}}
\right)
$$

**Example\:**

Suppose\:

$$
n_{in}=100
$$

Then\:

$$
Variance=\frac{2}{100}=0.02
$$

$$
Standard\ Deviation=\sqrt{0.02}\approx0.1414
$$

Therefore, weights are initialized with mean zero and standard deviation approximately 0.1414.

**Advantages\:**

- Suitable for ReLU networks.
- Helps maintain signal variance.
- Supports stable training in deep networks.

**Disadvantages\:**

- Not necessarily optimal for every activation function.
- Does not completely prevent vanishing or exploding gradients.

## F. Xavier vs He Initialization

| Xavier Initialization | He Initialization |
|--------|--------|
| Also called Glorot initialization | Also called Kaiming initialization |
| Uses fan-in and fan-out | Standard form uses fan-in |
| Designed for approximately linear or symmetric activations such as tanh | Designed for ReLU and related activations |
| Variance: $2/(n_{in}+n_{out})$ | Variance: $2/n_{in}$ |
| Common with tanh | Common with ReLU |

## G. Python Example

Using Keras\:

```python
from tensorflow.keras.layers import Dense

# Xavier initialization
layer1 = Dense(
    128,
    activation="tanh",
    kernel_initializer="glorot_uniform"
)

# He initialization
layer2 = Dense(
    128,
    activation="relu",
    kernel_initializer="he_normal"
)
```

Common initializers\:

- `glorot_uniform`
- `glorot_normal`
- `he_uniform`
- `he_normal`
- `random_normal`
- `zeros`

---

# 2.5 Batch Normalization

## What is Batch Normalization?

Batch Normalization (BN) is a technique used to normalize the activations of a neural network layer during training.

It helps make training more stable and often allows faster convergence.

During training, the distribution of activations can change as network parameters are updated.

Batch Normalization adjusts these activations using the mean and variance calculated from the current mini-batch.

## A. Why is Batch Normalization Needed?

Suppose a neural network has multiple hidden layers.

During training\:

1. The first layer updates its weights.
2. Its outputs change.
3. The second layer receives different input distributions.
4. The second layer must continuously adapt to changing inputs.

This can make optimization difficult.

Batch Normalization helps stabilize the distributions of intermediate activations.

**Note\:** It was originally motivated partly by the idea of reducing internal covariate shift. Its benefits are now understood to involve broader optimization and smoothing effects, rather than simply fixing that phenomenon.

## B. How Does Batch Normalization Work?

Suppose a mini-batch contains the following values\:

$$
X=[2,4,6,8]
$$

### Step 1: Calculate the Mean

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\mu_B=\frac{2+4+6+8}{4}=5
$$

### Step 2: Calculate the Variance

$$
\sigma_B^2=
\frac{1}{m}
\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

$$
\sigma_B^2=
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
$$

$$
\sigma_B^2=\frac{20}{4}=5
$$

### Step 3: Normalize the Values

$$
\hat{x}_i=
\frac{x_i-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

Where $\epsilon$ is a small constant used for numerical stability.

Ignoring the small epsilon for simplicity\:

$$
\hat{X}\approx[-1.342,-0.447,0.447,1.342]
$$

Now, the batch has approximately zero mean and unit variance.

### Step 4: Apply Learnable Parameters

Batch Normalization introduces two trainable parameters\:

- Gamma ($\gamma$\): Controls scaling.
- Beta ($\beta$\): Controls shifting.

The final output is\:

$$
y_i=\gamma\hat{x}_i+\beta
$$

Why are gamma and beta needed?

Because forcing every activation to have exactly zero mean and unit variance may restrict what the network can learn.

Gamma and beta allow the network to learn an appropriate scale and shift.

## C. Batch Normalization During Training and Testing

**During training\:**

- Calculates the mean of the current mini-batch.
- Calculates the variance of the current mini-batch.
- Normalizes activations.
- Updates the learnable gamma and beta parameters.
- Maintains running estimates of mean and variance.

**During inference/testing\:**

- Uses the accumulated running mean and variance.
- Does not calculate statistics from the current test batch in the usual inference mode.

This ensures more consistent behavior during prediction.

## D. Where is Batch Normalization Placed?

A common arrangement in an ordinary MLP is\:

```text
Input
  |
Dense Layer
  |
Batch Normalization
  |
ReLU Activation
  |
Next Dense Layer
  |
Batch Normalization
  |
ReLU Activation
  |
Output
```

Another arrangement places the activation before normalization in certain architectures. The correct placement depends on the network design.

## E. Python Example Using Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    ReLU
)

model = Sequential([
    Dense(128, input_shape=(20,)),

    BatchNormalization(),
    ReLU(),

    Dense(64),

    BatchNormalization(),
    ReLU(),

    Dense(1, activation="sigmoid")
])

model.summary()
```

## F. Advantages of Batch Normalization

1. Improves training stability.
2. Can accelerate convergence.
3. Helps control activation distributions.
4. May provide a mild regularization effect due to mini-batch statistics.
5. Can make training less sensitive to initialization and learning-rate choices.

## G. Disadvantages of Batch Normalization

1. Adds computational overhead.
2. Requires maintaining running statistics.
3. Very small batch sizes may produce unreliable statistics.
4. Differences between training and inference behavior can cause problems if handled incorrectly.
5. May be less suitable for certain architectures or highly variable batch distributions.

---

# 3. Comparison of All Five Techniques

| Technique | Main purpose | Main benefit |
|--------|--------|--------|
| Number of layers | Controls network depth | Learns hierarchical patterns |
| Number of neurons | Controls layer width | Controls learning capacity |
| Activation functions | Introduces non-linearity | Learns complex relationships |
| Weight initialization | Sets starting weights | Improves training stability |
| Batch normalization | Normalizes activations | Can improve convergence |

# 4. Complete Example: Optimized Neural Network

The following example combines all five architecture optimization techniques in a single neural network.

**Problem\:** Binary classification using 20 input features.

```python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    ReLU
)

model = Sequential([

    # First hidden layer: 128 neurons
    Dense(
        128,
        kernel_initializer="he_normal",
        input_shape=(20,)
    ),
    BatchNormalization(),
    ReLU(),

    # Second hidden layer: 64 neurons
    Dense(
        64,
        kernel_initializer="he_normal"
    ),
    BatchNormalization(),
    ReLU(),

    # Third hidden layer: 32 neurons
    Dense(
        32,
        kernel_initializer="he_normal"
    ),
    BatchNormalization(),
    ReLU(),

    # Output layer
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)
```

### Explanation of the Architecture

| Configuration | Selected value | Reason |
|--------|--------|--------|
| Hidden layers | 3 | Provides multiple levels of learning |
| Neurons | 128, 64, 32 | Gradually reduces representation size |
| Activation | ReLU | Efficient nonlinear activation |
| Initialization | He Normal | Suitable for ReLU |
| Normalization | Batch Normalization | Helps stabilize training |
| Output | Sigmoid | Binary classification |

These values are an example starting architecture, not a universally optimal configuration.

---

# 5. Summary: Important Points for Exams

1. **Network depth** determines how many layers are available to learn hierarchical representations.
2. **Number of neurons** determines the computational capacity of each layer.
3. **Activation functions** introduce non-linearity and enable complex pattern learning.
4. **Weight initialization** provides appropriate starting weights and helps prevent unstable signal propagation.
5. **Batch normalization** normalizes layer activations and can improve training stability and convergence.

## Important Differences to Remember

| Comparison | Key difference |
|--------|--------|
| Shallow vs Deep | Fewer hidden layers vs multiple hidden layers |
| ReLU vs Leaky ReLU | ReLU outputs zero for negative inputs; Leaky ReLU allows a small negative output |
| Sigmoid vs Softmax | Sigmoid supports independent binary outputs; Softmax produces a probability distribution over mutually exclusive classes |
| Xavier vs He | Xavier is commonly used with tanh; He is commonly used with ReLU |
| Batch Normalization vs Weight Initialization | Batch Normalization operates on activations during training; weight initialization sets starting weights |

---

# 6. Test Your Understanding

**Architecture optimization quiz** (/5 answered)

**Next topic in your roadmap\:** Optimization Techniques — Gradient Descent optimizers, learning rate and scheduling, adaptive optimizers (SGD, Momentum, AdaGrad, RMSprop, Adam), and how they improve neural network training.

![Deep Learning in PyTorch with CIFAR-10 dataset | by Sergio Alves | Medium](https://images.openai.com/static-rsc-4/xTAtSyPBd6KbOtFBE8CFMjkiFw-Td-OVeSyQcUXE4J80YhIuJ2dszwgaMPaO0Cb9V7vBFBqChM6PyNpLIfLN_NIQB8iTWtooeLmg9Pp2OTNBZQDei3xKEtZQayFmMgs8YwFDE9YzLenB9B2LSbPX2ZDoe2awJ3-PTHprDDg9gZONC3zf_vddmhsulIwA6ucK?purpose=fullsize)

![Project 4: Deep Learning with TensorFlow](https://images.openai.com/static-rsc-4/UEuTn08pW_8q6MDUq-nAVWmv7ufung3f6WuPzvEZHySbHwKvqGccjji7VSIC2sABcOwhlRW9BKkJGQWsyoj-bGqvB2pa-uq3Irl7SGpPWu9xC5i8PAnR6SIF2GsqU_9elLGiNOVu3GjVtkeQClt5BUSdMey_JXgPwkUigihieOna1PScZw1HuYtvNcwZQnL5?purpose=fullsize)

![The basic units: neurons — Introduction<br> to Machine Learning](https://images.openai.com/static-rsc-4/23-_al-EK2x9z6xU5fOHQt1H94gYwQRdG3NcnIq6NYxUjwo1TcBs0vAYjkkYaxEP72jWgqaiq2AnruYJdM_uKie2E-qObP7NqkeSdosCQCm5AiX-v0oURI1cyeKrMraRq9EOH0pNeEQ0oZxfyB2Jx5kwGwWNYjfnO8R60LvCDkxw6dWS0m9XJ0tIfze9J34E?purpose=fullsize)

![Deep Learning Training Fundamentals Explained Simply | by Dewasheesh Rana | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/tJB-hO8f_XPxaQVF8PJvBfZTOW9d7dM3ocxbLiYXLEVoyG2GI_ANfxQK6MWbse3sFBoLXn-ZnrT8JqTiOfy5eUe0C6VvnRhsiWQYNd8gUg7gqPibysBfMCmtpgTQj5wB0wPSYoddI6Ybz67eVI4Vb6RJamup_KStu5_-jQGFhXEzbETM7lcsJ7GVNtoaDIr4?purpose=fullsize)

