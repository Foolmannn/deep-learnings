# Batch Normalization in Deep Learning

## 1. Introduction

**Batch Normalization (BN)** is a technique used in Deep Learning to improve the speed, stability, and performance of neural network training.

It was introduced by Sergey Ioffe and Christian Szegedy in the research paper *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* $2015).

During neural network training, the weights of each layer are continuously updated through backpropagation. As a result, the distribution of inputs received by subsequent layers can change throughout training.

This can make training slower and less stable.

Batch Normalization addresses this problem by normalizing the activations of a layer using the mean and variance calculated from a mini-batch of training data.

In simple terms\:

**Batch Normalization adjusts the values entering a neural network layer so that they have a more stable distribution during training.**

### Why is Batch Normalization needed?

Consider a neural network with multiple hidden layers\:

**Input Layer**

**Hidden Layer 1**

**Hidden Layer 2**

**Hidden Layer 3**

**Output Layer**

During training\:

1. The first hidden layer updates its weights.
2. These updates change the values passed to the second layer.
3. The second layer receives a different distribution of inputs.
4. This process continues through the network.

Consequently, deeper layers must continuously adapt to changing input distributions.

Batch Normalization helps make the training process more stable by normalizing layer activations.

---

## 2. Internal Covariate Shift

Internal Covariate Shift refers to the change in the distribution of inputs to a neural network layer during training.

Let's understand this with an example.

Suppose a hidden layer initially receives these values\:

`[2, 4, 6, 8, 10]`

After some weight updates, the same layer might receive\:

`[20, 40, 60, 80, 100]`

The distribution has changed significantly.

This happens because the parameters of previous layers are updated during training.

### Problems caused by changing distributions

1. **Slower training\:** Layers need to continuously adapt to changing input distributions.
2. **Unstable gradients\:** Large variations in activations can affect gradient propagation.
3. **Learning difficulties\:** Some neurons may receive values that are not suitable for their activation functions.
4. **Careful initialization\:** Training may become more sensitive to the initial weights.
5. **Lower learning rates\:** Smaller learning rates may be required to maintain stable training.

Batch Normalization was proposed to reduce these difficulties.

*Note\:* Internal Covariate Shift was the original motivation for Batch Normalization. Later research showed that its benefits are not explained entirely by reducing this phenomenon; smoother optimization and more stable gradients are also important.

---

## 3. How Batch Normalization Works

Batch Normalization works by normalizing the activations of a layer using the mean and variance of a mini-batch.

It generally performs four main operations\:

1. Calculate the mini-batch mean.
2. Calculate the mini-batch variance.
3. Normalize the activations.
4. Scale and shift the normalized values.

Let's understand each step mathematically.

### Step 1: Calculate the Mini-Batch Mean

Suppose a mini-batch contains the following values\:

$$
X = [2,4,6,8]
$$

The mean is calculated as\:

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

Where\:

- \(\mu_B$ \= Mini-batch mean
- $m$ \= Number of samples in the batch
- $x_i$ \= Individual activation value

For our example\:

$$
\mu_B = \frac{2+4+6+8}{4}
$$

$$
\boxed{\mu_B=5}
$$

### Step 2: Calculate the Mini-Batch Variance

Variance measures how far the values are spread around the mean.

The formula is\:

$$
\sigma_B^2 =
\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

Using our example\:

$$
\sigma_B^2 =
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
$$

$$
=\frac{9+1+1+9}{4}
$$

$$
\boxed{\sigma_B^2=5}
$$

### Step 3: Normalize the Activations

Now, each activation is normalized using the calculated mean and variance.

The formula is\:

$$
\hat{x}_i =
\frac{x_i-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

Where\:

- $\hat{x}_i$ \= Normalized activation
- $\mu_B$ \= Mini-batch mean
- $\sigma_B^2$ \= Mini-batch variance
- $\epsilon$ \= Small constant added for numerical stability

For example, assuming $\epsilon$ is very small\:

$$
\hat{x}_1 = \frac{2-5}{\sqrt{5}}
$$

$$
\hat{x}_1 \approx -1.342
$$

Similarly, the normalized values are approximately\:

$$
[-1.342,-0.447,0.447,1.342]
$$

After normalization, the values have approximately\:

- Mean = 0
- Variance = 1

**Before and after normalization**

**Original values** Mean = 5

(Batch Normalization)

**Normalized values** Mean ≈ 0

### Step 4: Scale and Shift

An important feature of Batch Normalization is that it does not simply force every layer to have a fixed output distribution.

After normalization, it introduces two learnable parameters\:

- **Gamma ($\gamma$\)\:** Controls scaling.
- **Beta ($\beta$\)\:** Controls shifting.

The final equation is\:

$$
\boxed{y_i=\gamma\hat{x}_i+\beta}
$$

Where\:

- $\hat{x}_i$ \= Normalized activation
- $\gamma$ \= Learnable scaling parameter
- $\beta$ \= Learnable shifting parameter
- $y_i$ \= Final Batch Normalization output

Initially, gamma is generally set to 1 and beta to 0.

During training, the network learns appropriate values for these parameters.

**Why are gamma and beta necessary?**

Normalization forces the data into a standardized distribution. However, a neural network might benefit from a different distribution for learning.

Gamma and beta allow the network to adjust the normalized values to a suitable scale and position.

Therefore, Batch Normalization provides normalization while maintaining the flexibility of the network.

---

## 4. Batch Normalization During Training and Testing

Batch Normalization behaves differently during training and inference (testing).

This is one of the most important concepts to understand.

### A. During Training

During training, Batch Normalization uses the mean and variance of the current mini-batch.

For every mini-batch\:

1. Calculate the batch mean.
2. Calculate the batch variance.
3. Normalize the activations.
4. Apply gamma and beta.
5. Update the learnable parameters through backpropagation.
6. Update the running mean and running variance.

For example, suppose the batch size is 32.

The network calculates the mean and variance using those 32 samples and normalizes their activations.

As training progresses, the running statistics are updated using the statistics from successive batches.

### B. During Testing (Inference)

During testing, we generally do not calculate the mean and variance from the current test batch.

Instead, Batch Normalization uses the **running mean and running variance** accumulated during training.

Why?

Suppose we deploy a trained model that receives only one input at a time.

Calculating the mean and variance from a single sample would not provide reliable statistics.

Therefore, the model uses previously accumulated statistics.

The inference equation becomes\:

$$
y = \gamma\left(
\frac{x-\mu_{\text{running}}}
{\sqrt{\sigma_{\text{running}}^2+\epsilon}}
\right)+\beta
$$

Here\:

- $\mu_{\text{running}}$ \= Running mean.
- $\sigma_{\text{running}}^2$ \= Running variance.
- $\gamma$ \= Learned scaling parameter.
- $\beta$ \= Learned shifting parameter.

### Difference between training and testing

| Training | Testing |
|--------|--------|
| Uses current batch mean | Uses running mean |
| Uses current batch variance | Uses running variance |
| Updates running statistics | Does not update running statistics |
| Uses training mode | Uses evaluation mode |
| Batch statistics introduce some noise | Uses fixed statistics |

---

## 5. Placement of Batch Normalization in a Neural Network

Batch Normalization is generally placed between a linear operation and an activation function.

For example\:

**Input**

**Dense / Linear Layer**

Weighted sum

**Batch Normalization**

Normalize, scale and shift

**Activation Function (ReLU)**

**Next Layer**

This is commonly written as\:

$$
\boxed{\text{Dense}\rightarrow
\text{BatchNorm}\rightarrow\text{ReLU}}
$$

Why this order?

- Dense calculates the weighted sum.
- Batch Normalization normalizes the resulting activations.
- ReLU introduces non-linearity.

This is a common convention, not a universal rule. Some architectures use different arrangements.

### Batch Normalization in CNNs

In Convolutional Neural Networks (CNNs), Batch Normalization is generally placed after the convolution layer and before the activation function.

$$
\text{Conv2D}\rightarrow\text{BatchNorm}
\rightarrow\text{ReLU}
$$

For convolutional layers, statistics are calculated separately for each feature channel, across the batch and spatial dimensions.

For example, if a convolutional layer produces 64 channels, Batch Normalization generally maintains\:

- 64 mean values.
- 64 variance values.
- 64 gamma parameters.
- 64 beta parameters.

This helps normalize feature maps produced by convolutional layers.

---

## 6. Advantages of Batch Normalization

Batch Normalization provides several benefits during neural network training.

### 1. Faster Training

Batch Normalization often allows neural networks to converge faster by making optimization more stable.

It can allow the use of higher learning rates, reducing the number of training iterations required.

### 2. More Stable Gradient Flow

It helps control the scale of activations flowing through the network.

This can make optimization more stable and reduce some difficulties associated with vanishing or exploding gradients.

However, it does not completely eliminate these problems.

### 3. Less Sensitivity to Weight Initialization

Without normalization, poor weight initialization can make training difficult.

Batch Normalization reduces sensitivity to initialization, although appropriate initialization is still important.

### 4. Regularization Effect

Since batch statistics vary slightly between mini-batches, Batch Normalization introduces some noise during training.

This can provide a mild regularization effect and sometimes reduce overfitting.

However, it is not a replacement for Dropout or other regularization techniques.

### 5. Better Optimization

Batch Normalization can make the optimization landscape smoother, allowing gradient-based optimization methods to work more effectively.

### 6. Improved Training of Deep Networks

It can make training deeper neural networks easier by controlling activation scales across layers.

These benefits were among the motivations and findings of the original Batch Normalization research.  [Google Research](https://research.google/pubs/batch-normalization-accelerating-deep-network-training-by-reducing-internal-covariate-shift/?utm_source=chatgpt.com) [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v37/ioffe15.html?utm_source=chatgpt.com) [IEEE Technology Navigator](https://technav.ieee.org/topic/batch-normalization/?utm_source=chatgpt.com)

---

## 7. Disadvantages and Limitations of Batch Normalization

Although Batch Normalization is useful, it also has some limitations.

### 1. Dependence on Batch Size

Batch Normalization calculates statistics from mini-batches.

When the batch size is very small, the calculated mean and variance may be unreliable.

For example\:

- Batch size 64: Generally provides more stable statistics.
- Batch size 32: Often works well.
- Batch size 2: Statistics may be noisy.

The ideal batch size depends on the dataset and architecture.

### 2. Additional Computational Cost

Batch Normalization introduces additional calculations\:

- Mean calculation.
- Variance calculation.
- Normalization.
- Scaling and shifting.

Although the overhead is usually modest, it increases computational requirements.

### 3. Different Training and Testing Behavior

Batch Normalization uses batch statistics during training but running statistics during inference.

If these running statistics are inaccurate, model performance during inference may decrease.

### 4. Not Always Suitable for Small-Batch Training

In tasks involving limited GPU memory, the batch size may be too small for reliable Batch Normalization.

In such cases, alternatives such as Group Normalization or Layer Normalization may be more appropriate.

### 5. Additional Parameters

Batch Normalization introduces learnable gamma and beta parameters for each normalized feature or channel.

This slightly increases the number of model parameters.

---

## 8. Batch Normalization vs Layer Normalization

Both Batch Normalization and Layer Normalization normalize neural network activations, but they calculate statistics differently.

| Basis | Batch Normalization | Layer Normalization |
|--------|--------|--------|
| Normalization | Across batch dimension | Across features of each sample |
| Batch size dependency | Yes | No |
| Training and testing | Different behavior | Same normalization behavior |
| Running statistics | Required in standard BN | Not required |
| Common applications | CNNs and MLPs | Transformers and RNNs |
| Small batch suitability | Can be problematic | Generally suitable |

### Example

Suppose we have a batch containing 4 samples, each with 3 features\:

$$
X=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9\\
10&11&12
\end{bmatrix}
$$

**Batch Normalization\:** Calculates statistics separately for each feature across the four samples.

**Layer Normalization\:** Calculates statistics separately for each sample across its three features.

Remember: Batch Normalization looks across samples; Layer Normalization looks across features within a sample.

---

## 9. Implementation of Batch Normalization Using TensorFlow/Keras

Keras provides a built-in `BatchNormalization()` layer.

### Example 1: Using Batch Normalization in a Simple Neural Network

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU

# Create neural network
model = Sequential([
    Dense(64, input_shape=(10,)),
    BatchNormalization(),
    ReLU(),

    Dense(32),
    BatchNormalization(),
    ReLU(),

    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()
```

**Explanation\:**

- `Dense(64)`\: Creates a fully connected layer with 64 neurons.
- `BatchNormalization()`\: Normalizes the activations of the previous Dense layer.
- `ReLU()`\: Applies the ReLU activation function.
- `Dense(32)`\: Creates another hidden layer.
- `Dense(1)`\: Produces the final binary classification output.

### Example 2: Training the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

During training, Keras automatically\:

1. Calculates batch mean and variance.
2. Normalizes activations.
3. Updates gamma and beta through backpropagation.
4. Updates running statistics.

### Example 3: Making Predictions

```python
predictions = model.predict(X_test)
```

Keras automatically switches Batch Normalization to inference behavior when using `model.predict()`\.

You do not need to manually calculate the running mean and variance.

---

## 10. Implementation of Batch Normalization Using PyTorch

PyTorch also provides built-in Batch Normalization layers.

The commonly used layers are\:

| Layer | Purpose |
|--------|--------|
| `nn.BatchNorm1d()` | Fully connected layers and 1D features |
| `nn.BatchNorm2d()` | Convolutional 2D feature maps |
| `nn.BatchNorm3d()` | 3D convolutional feature maps |

### Example 1: Batch Normalization in a Neural Network

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = NeuralNetwork()

print(model)
```

**Explanation\:**

- `nn.Linear(10, 64)`\: Takes 10 input features and produces 64 outputs.
- `nn.BatchNorm1d(64)`\: Normalizes the 64 features.
- `nn.ReLU()`\: Applies the ReLU activation function.
- `nn.Linear(64, 32)`\: Creates another hidden layer.
- `nn.BatchNorm1d(32)`\: Normalizes 32 features.

### Example 2: Training the Model

```python
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training mode
model.train()

for epoch in range(50):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Here\:

- `model.train()` activates training behavior for Batch Normalization.
- `loss.backward()` calculates gradients.
- `optimizer.step()` updates model parameters.

### Example 3: Testing the Model

```python
model.eval()

with torch.no_grad():

    predictions = model(X_test)
```

Here\:

- `model.eval()` switches Batch Normalization to inference mode.
- `torch.no_grad()` disables gradient calculation.
- Running mean and variance are used for normalization.

**Important\:** In PyTorch, forgetting `model.eval()` before inference is a common mistake because Batch Normalization may continue using current batch statistics.

---

## 11. Important Parameters of Batch Normalization

Both TensorFlow/Keras and PyTorch provide parameters to control Batch Normalization.

| Parameter | Meaning |
|--------|--------|
| `momentum` | Controls how running statistics are updated |
| `epsilon (eps)` | Small value to prevent division by zero |
| `gamma` | Learnable scaling parameter |
| `beta` | Learnable shifting parameter |
| `affine` | Enables or disables learnable scale and shift in PyTorch |
| `track_running_stats` | Controls whether PyTorch tracks running statistics |

### Running Mean and Variance Update

The running mean is updated approximately as follows\:

$$
\mu_{\text{new}} =
(1-m)\mu_{\text{old}}+m\mu_B
$$

Similarly, running variance is updated using\:

$$
\sigma_{\text{new}}^2 =
(1-m)\sigma_{\text{old}}^2+m\sigma_B^2
$$

Here, $m$ represents the moving-average update factor.

**Important framework difference\:**

- Keras commonly uses `momentum=0.99`\.
- PyTorch commonly uses `momentum=0.1`\.

Their momentum conventions are opposite in the update equations, so the same numerical value does not mean the same thing.

---

## 12. Practical Tips When Using Batch Normalization

1. Place Batch Normalization after Dense or Convolutional layers and generally before activation functions.
2. Avoid extremely small batch sizes when possible.
3. Use `model.eval()` in PyTorch during inference.
4. Use `model.train()` during PyTorch training.
5. Do not manually normalize data again using Batch Normalization statistics.
6. Remember that Batch Normalization does not eliminate the need for proper weight initialization.
7. If training with very small batches, consider Group Normalization or Layer Normalization.

## 13. Alternatives to Batch Normalization

When Batch Normalization is not suitable, other normalization techniques can be used.

| Technique | Main idea | Common application |
|--------|--------|--------|
| Batch Normalization | Normalizes across batch samples | CNNs, MLPs |
| Layer Normalization | Normalizes features within each sample | Transformers, RNNs |
| Instance Normalization | Normalizes each sample and channel independently | Image style transfer |
| Group Normalization | Divides channels into groups and normalizes each group | Computer vision with small batches |
| RMS Normalization | Normalizes using root mean square without subtracting the mean | Transformers and LLMs |

These techniques provide different approaches to controlling activation distributions.  [IEEE Technology Navigator](https://technav.ieee.org/topic/batch-normalization/?utm_source=chatgpt.com) [DOI](https://doi.org/10.1109%2FTPAMI.2023.3319005?utm_source=chatgpt.com) [Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=batchnorm2d&utm_source=chatgpt.com)

---

## 14. Summary of Batch Normalization

Batch Normalization is an important technique in Deep Learning that normalizes the activations of neural network layers using mini-batch statistics.

The overall process can be summarized as\:

**Key points to remember\:**

- Batch Normalization was introduced in 2015.
- It normalizes layer activations using mini-batch statistics.
- It uses gamma and beta as learnable parameters.
- It can accelerate training and improve optimization stability.
- It maintains running statistics for inference.
- It can have limitations when the batch size is very small.
- It is widely used in CNNs and fully connected neural networks.

### References for further study

1. Original Batch Normalization paper (ICML 2015)\.
2. Dive into Deep Learning – Batch Normalization\.
3. IEEE – Batch Normalization overview\.

---