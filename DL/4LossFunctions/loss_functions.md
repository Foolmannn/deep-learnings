# Loss Functions in Deep Learning – Detailed Study Notes

Loss functions are one of the most important concepts in Deep Learning. They help a neural network understand **how wrong its predictions are and how it should improve its parameters (weights and biases) during training.**

Since you have already studied Perceptrons, MLPs, and Forward Propagation, let's connect loss functions with the complete neural network training process.

# 1. What is a Loss Function?

A **loss function** is a mathematical function that measures the difference between the actual output (ground truth) and the predicted output of a neural network.

In simple terms\:

- The neural network makes a prediction.
- We compare that prediction with the actual answer.
- The loss function calculates the prediction error.
- Backpropagation uses this error to calculate gradients.
- The optimizer updates the weights and biases to reduce the loss.

For example, suppose we are building a neural network to predict house prices.

| Actual price | Predicted price | Error |
|--------|--------|--------|
| $200,000 | $190,000 | \$10,000 |
| $300,000 | $280,000 | \$20,000 |
| $400,000 | $450,000 | \$50,000 |

The loss function calculates a numerical value representing how poorly the model performed.

A smaller loss generally indicates better predictions.

## 1.1 Mathematical representation

Let\:

- $y$ \= Actual output
- $\hat{y}$ \= Predicted output
- $L$ \= Loss function

Then\:

$$
L = f(y,\hat{y})
$$

Our objective during training is\:

$$
\boxed{\min_{\theta} L(y,\hat{y})}
$$

Here, $\theta$ represents all trainable parameters of the neural network (weights and biases).

We want to find parameters that minimize the loss function.

---

# 2. Why Do We Need Loss Functions?

Consider a neural network that predicts whether an email is spam.

Input\:

```text
Congratulations! You won a free iPhone.
```

Actual output\:

```python
y = 1  # Spam
```

Suppose the neural network predicts\:

```python
y_pred = 0.20
```

If we interpret the prediction as a probability of spam, the model is saying there is only a 20% chance that the email is spam.

But the actual answer is spam.

Therefore, the prediction is incorrect.

We need a mathematical mechanism to quantify how incorrect the prediction is.

That is the job of the loss function.

Without a loss function, the neural network would not have a clearly defined training objective.

## 2.1 Relationship between loss function, backpropagation, and optimizer

These three concepts work together but have different responsibilities.

| Component | Responsibility |
|--------|--------|
| Forward propagation | Generates predictions |
| Loss function | Measures prediction errors |
| Backpropagation | Calculates gradients |
| Optimizer | Updates weights and biases |

The overall process\:

**Input Data**

Features and actual labels

**Forward Propagation**

Calculate predicted output

**Loss Function**

Calculate prediction error

**Backpropagation**

Calculate gradients of weights and biases

**Optimizer**

Update parameters using gradients

Repeat until the model learns

An important distinction\:

**The loss function does not directly update weights.** It provides the objective whose gradients are used by backpropagation and the optimizer.

---

# 3. Types of Loss Functions

Loss functions are selected depending on the type of machine learning problem.

There are three major categories\:

**1\. Regression Loss Functions**

Used when predicting continuous numerical values.

Examples: MSE, MAE, Huber Loss.

---

**2\. Classification Loss Functions**

Used when predicting classes or class probabilities.

Examples: Binary Cross-Entropy, Categorical Cross-Entropy.

---

**3\. Specialized Loss Functions**

Used for specific learning problems.

Examples: Contrastive Loss, Triplet Loss, Dice Loss, KL Divergence.

Let's understand each one mathematically and practically.

# 4. Regression Loss Functions

Regression problems predict continuous numerical values.

Examples\:

- House price prediction
- Temperature prediction
- Stock price prediction
- Sales forecasting
- Salary prediction

Suppose we have the following predictions\:

| Actual | Predicted |
|--------|--------|
| 100 | 110 |
| 200 | 180 |
| 300 | 330 |

We need a loss function that measures the difference between actual and predicted values.

## 4.1 Mean Squared Error (MSE)

Mean Squared Error is one of the most commonly used regression loss functions.

It calculates the average of squared differences between actual and predicted values.

### Mathematical formula

$$
\boxed{
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
$$

Where\:

- $n$ \= Number of samples
- $y_i$ \= Actual value
- $\hat{y}_i$ \= Predicted value

### Example

Consider\:

| Actual | Predicted | Error | Squared error |
|--------|--------|--------|--------|
| 100 | 110 | \-10 | 100 |
| 200 | 180 | 20 | 400 |
| 300 | 330 | \-30 | 900 |

Therefore\:

$$
MSE = \frac{100+400+900}{3}
$$

$$
\boxed{MSE = 466.67}
$$

The loss value is 466.67.

### Why do we square the errors?

There are three major reasons\:

**1\. Prevent cancellation of errors**

Suppose errors are\:

$$
+10,-10
$$

Their sum is zero, even though both predictions are incorrect.

Squaring gives\:

$$
10^2+(-10)^2=200
$$

**2\. Penalize larger errors**

Compare\:

$$
10^2=100
$$

$$
50^2=2500
$$

An error of 50 receives a much larger penalty than an error of 10.

**3\. Mathematical convenience**

MSE is differentiable and has a convenient gradient, which makes it useful for gradient-based optimization.

### Derivative of MSE

For one prediction\:

$$
L=(y-\hat y)^2
$$

Differentiating with respect to prediction\:

$$
\frac{\partial L}{\partial \hat y}
=2(\hat y-y)
$$

For $n$ samples\:

$$
\boxed{
\frac{\partial MSE}{\partial \hat y_i}
=\frac{2}{n}(\hat y_i-y_i)
}
$$

This gradient indicates how the prediction should change to reduce the loss.

### Python implementation

```python
import numpy as np

y_true = np.array([100, 200, 300])
y_pred = np.array([110, 180, 330])

mse = np.mean((y_true - y_pred) ** 2)

print("MSE:", mse)
```

Output\:

```text
MSE: 466.6666666666667
```

Using Scikit-learn\:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)

print(mse)
```

### Advantages

- Simple to understand and implement.
- Differentiable everywhere.
- Penalizes large prediction errors.
- Works well when large errors should be strongly penalized.

### Disadvantages

- Highly sensitive to outliers.
- Squaring errors gives greater importance to large deviations.
- The loss is expressed in squared units of the target variable.

(Best suited for) Regression tasks where large errors should receive strong penalties.

---

## 4.2 Mean Absolute Error (MAE)

Mean Absolute Error calculates the average absolute difference between actual and predicted values.

Unlike MSE, it does not square the errors.

### Mathematical formula

$$
\boxed{
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
}
$$

### Example

Using the previous dataset\:

| Actual | Predicted | Absolute error |
|--------|--------|--------|
| 100 | 110 | 10 |
| 200 | 180 | 20 |
| 300 | 330 | 30 |

Therefore\:

$$
MAE=\frac{10+20+30}{3}
$$

$$
\boxed{MAE=20}
$$

### Derivative of MAE

For one prediction\:

$$
L=|y-\hat y|
$$

Its derivative with respect to prediction is\:

$$
\frac{\partial L}{\partial \hat y}
=
\begin{cases}
1 & \hat y>y\\
-1 & \hat y<y
\end{cases}
$$

At $\hat y=y$, the ordinary derivative is undefined, but an optimizer can use a subgradient.

Notice something important\:

Unlike MSE, the gradient magnitude of MAE does not increase with the error magnitude.

This is one reason MAE is less sensitive to outliers.

### Python implementation

```python
import numpy as np

y_true = np.array([100, 200, 300])
y_pred = np.array([110, 180, 330])

mae = np.mean(np.abs(y_true - y_pred))

print("MAE:", mae)
```

Output\:

```text
MAE: 20.0
```

### Advantages

- Less sensitive to outliers than MSE.
- Easy to interpret.
- Expressed in the same units as the target variable.

### Disadvantages

- Not differentiable at zero error.
- Constant gradient magnitude can make optimization less smooth.
- Does not penalize large errors as strongly as MSE.

(Best suited for) Regression problems where you want to reduce average absolute prediction error and limit outlier influence.

---

## 4.3 Huber Loss

Huber Loss combines the advantages of MSE and MAE.

- For small errors, it behaves like MSE.
- For large errors, it behaves like MAE.

This makes it particularly useful when the dataset contains outliers.

### Mathematical formula

$$
L_\delta(y,\hat y)=
\begin{cases}
\frac{1}{2}(y-\hat y)^2
& |y-\hat y|\leq\delta\\
\delta\left(|y-\hat y|-\frac12\delta\right)
& |y-\hat y|>\delta
\end{cases}
$$

Here, $\delta$ is a threshold that determines when the loss switches from quadratic to linear behavior.

### Understanding the formula

Suppose\:

$$
\delta=1
$$

If error is 0.5\:

$$
L=\frac12(0.5)^2=0.125
$$

If error is 3\:

$$
L=1(3-0.5)=2.5
$$

The large error is penalized linearly instead of quadratically.

### Python implementation

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):

    error = np.abs(y_true - y_pred)

    loss = np.where(
        error <= delta,
        0.5 * error**2,
        delta * (error - 0.5 * delta)
    )

    return np.mean(loss)
```

### Advantages

- More robust to outliers than MSE.
- Smooth around zero.
- Combines quadratic and linear penalties.

### Disadvantages

- Requires selecting a suitable delta.
- May not be as straightforward to interpret as MAE.

---

## 4.4 Mean Squared Logarithmic Error (MSLE)

MSLE calculates the squared difference between logarithmic actual and predicted values.

### Formula

$$
\boxed{
MSLE=\frac1n\sum_{i=1}^{n}
\left[
\log(1+y_i)-\log(1+\hat y_i)
\right]^2
}
$$

Why use logarithms?

Sometimes relative errors matter more than absolute errors.

For example\:

- Predicting 100 instead of 200.
- Predicting 1,000 instead of 1,100.

Both have different absolute errors, but the relative difference is more meaningful in certain applications.

MSLE emphasizes relative differences and reduces the impact of very large numerical values.

It generally requires non-negative target and prediction values.

### Python

```python
from sklearn.metrics import mean_squared_log_error

loss = mean_squared_log_error(y_true, y_pred)
```

### Applications

- Sales forecasting.
- Population prediction.
- Demand forecasting.
- Growth-related prediction problems.

---

## 4.5 Comparison of Regression Loss Functions

| Loss | Error calculation | Outlier sensitivity |
|--------|--------|--------|
| MSE | Squared error | High |
| MAE | Absolute error | Lower |
| Huber | Squared then linear | Moderate |
| MSLE | Squared log difference | Depends on relative errors |

![](regression_loss.jpg)

# 5. Classification Loss Functions

Classification is different from regression.

In classification, the neural network predicts a class or the probability of belonging to a particular class.

Examples\:

- Spam vs Not Spam
- Cat vs Dog
- Disease vs No Disease
- Positive vs Negative reviews
- Multiple-digit classification (0–9)

The most commonly used classification loss functions are Cross-Entropy-based losses.

## 5.1 Binary Cross-Entropy (BCE)

Binary Cross-Entropy is used when we have exactly two classes.

Examples\:

- Spam / Not Spam
- Yes / No
- Positive / Negative
- Fraud / Not Fraud

The target labels are generally represented as\:

$$
y\in\{0,1\}
$$

The model outputs a probability\:

$$
\hat y\in[0,1]
$$

For example\:

```python
y_true = 1
y_pred = 0.9
```

This means the actual class is 1, and the model predicts a 90% probability of class 1.

### Mathematical formula

$$
\boxed{
BCE=-\left[
y\log(\hat y)+(1-y)\log(1-\hat y)
\right]
}
$$

For multiple samples\:

$$
BCE=-\frac1n\sum_{i=1}^{n}
\left[
y_i\log(\hat y_i)
+(1-y_i)\log(1-\hat y_i)
\right]
$$

Let's understand how this works.

### Case 1: Actual output is 1

If\:

$$
y=1
$$

Then\:

$$
BCE=-\log(\hat y)
$$

Suppose the prediction is\:

$$
\hat y=0.9
$$

Then\:

$$
BCE=-\log(0.9)
$$

$$
\boxed{BCE=0.1053}
$$

The loss is small because the prediction is close to the actual answer.

But suppose\:

$$
\hat y=0.1
$$

Then\:

$$
BCE=-\log(0.1)
$$

$$
\boxed{BCE=2.3026}
$$

The loss is significantly higher because the model made a highly incorrect prediction.

### Case 2: Actual output is 0

If\:

$$
y=0
$$

Then\:

$$
BCE=-\log(1-\hat y)
$$

Suppose\:

$$
\hat y=0.1
$$

Then\:

$$
BCE=-\log(0.9)=0.1053
$$

The prediction is correct because the probability of class 1 is low.

### Understanding BCE visually

![](bce_loss.jpg)


The graph demonstrates an important property\:

- Correct and confident predictions receive low loss.
- Incorrect and confident predictions receive high loss.
- Uncertain predictions receive intermediate loss.

### Why do we use logarithms?

Logarithms heavily penalize confident incorrect predictions.

For example, when the actual class is 1\:

| Predicted probability | BCE |
|--------|--------|
| 0.99 | 0.0101 |
| 0.90 | 0.1054 |
| 0.70 | 0.3567 |
| 0.50 | 0.6931 |
| 0.10 | 2.3026 |
| 0.01 | 4.6052 |

This is very useful because we want a neural network to learn not just the correct class but also meaningful probabilities.

### Python implementation

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss

y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])

print(binary_cross_entropy(y_true, y_pred))
```

### Using PyTorch

```python
import torch
import torch.nn as nn

criterion = nn.BCELoss()

y_pred = torch.tensor(
    [0.9, 0.2, 0.8, 0.7]
)

y_true = torch.tensor(
    [1.0, 0.0, 1.0, 1.0]
)

loss = criterion(y_pred, y_true)

print(loss)
```

**Important\:** `nn.BCELoss()` expects probabilities, not raw logits.

In practice, PyTorch commonly uses\:

```python
criterion = nn.BCEWithLogitsLoss()
```

This combines the sigmoid operation and BCE calculation into a numerically stable implementation.

With `BCEWithLogitsLoss`, provide the raw output from the final linear layer, without applying sigmoid first.

### Advantages

- Suitable for binary classification.
- Penalizes confident incorrect predictions.
- Works well with sigmoid output probabilities.
- Provides useful gradients for neural network training.

---

## 5.2 Categorical Cross-Entropy (CCE)

Categorical Cross-Entropy is used for multi-class classification when each sample belongs to exactly one class.

For example, imagine classifying an image into\:

- Cat
- Dog
- Horse

The neural network produces probabilities for all three classes.

Suppose\:

```python
actual = [0, 1, 0]

predicted = [0.1, 0.8, 0.1]
```

The actual class is Dog.

The model predicts\:

- Cat: 10%
- Dog: 80%
- Horse: 10%

The prediction is correct and reasonably confident.

### Mathematical formula

For a single sample\:

$$
\boxed{
CCE=-\sum_{i=1}^{C}y_i\log(\hat y_i)
}
$$

Where\:

- $C$ \= Number of classes.
- $y_i$ \= Actual one-hot encoded label.
- $\hat y_i$ \= Predicted probability for class $i$\.

For our example\:

$$
CCE=-[
0\log(0.1)+1\log(0.8)+0\log(0.1)
]
$$

Therefore\:

$$
CCE=-\log(0.8)
$$

$$
\boxed{CCE=0.2231}
$$

Notice that only the probability corresponding to the actual class contributes to the loss.

### What happens when the prediction is incorrect?

Suppose\:

```python
actual = [0, 1, 0]

predicted = [0.8, 0.1, 0.1]
```

Then\:

$$
CCE=-\log(0.1)
$$

$$
\boxed{CCE=2.3026}
$$

The model receives a much higher loss.

### Python implementation

```python
import numpy as np

def categorical_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred),
        axis=1
    )

    return np.mean(loss)

y_true = np.array([
    [0, 1, 0],
    [1, 0, 0]
])

y_pred = np.array([
    [0.1, 0.8, 0.1],
    [0.7, 0.2, 0.1]
])

print(categorical_cross_entropy(y_true, y_pred))
```

### PyTorch implementation

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Raw logits, not softmax probabilities
y_pred = torch.tensor([
    [1.0, 3.0, 0.5],
    [2.5, 1.0, 0.2]
])

# Class indices
y_true = torch.tensor([1, 0])

loss = criterion(y_pred, y_true)

print(loss)
```

**Important\:** PyTorch's `CrossEntropyLoss` internally applies the equivalent of log-softmax and negative log-likelihood. Do not apply softmax before passing predictions to this loss.

---

## 5.3 Sparse Categorical Cross-Entropy

Sparse Categorical Cross-Entropy performs the same basic calculation as Categorical Cross-Entropy.

The difference is in the representation of actual labels.

Categorical Cross-Entropy generally uses one-hot encoded labels\:

```python
y_true = [0, 1, 0]
```

Sparse Categorical Cross-Entropy uses integer class labels\:

```python
y_true = 1
```

Suppose we have 10 classes.

One-hot encoding requires a vector of 10 elements for each sample, whereas sparse labels require only one integer.

### Keras implementation

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

For one-hot labels\:

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 5.4 Binary vs Categorical vs Sparse Categorical Cross-Entropy

This is particularly important when building neural networks.

| Property | Binary CE | Categorical CE | Sparse Categorical CE |
|--------|--------|--------|--------|
| Classes | 2 | Multiple | Multiple |
| Labels | 0 or 1 | One-hot | Integer |
| Output activation | Sigmoid | Softmax | Softmax |
| Example output | 0.8 | $$0.1, 0.8, 0.1$$ | $$0.1, 0.8, 0.1$$ |
| Typical problem | Spam detection | Image classification | Digit classification |

Remember\:

- Binary classification: BCE.
- Multi-class classification with one-hot labels: Categorical CE.
- Multi-class classification with integer labels: Sparse Categorical CE.

---

# 6. Why Softmax and Cross-Entropy Are Commonly Used Together

You have already studied the MLP and forward propagation. Let's connect the output layer to the loss function.

Suppose the network is classifying an image into three classes.

The final linear layer produces logits\:

$$
z=[2.0,1.0,0.1]
$$

These logits are not probabilities.

We apply Softmax\:

$$
P(y_i)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

The resulting probabilities are approximately\:

$$
\hat y=[0.659,0.242,0.099]
$$

Suppose the actual class is the first class\:

$$
y=[1,0,0]
$$

Categorical Cross-Entropy becomes\:

$$
L=-\log(0.659)
$$

$$
L\approx0.417
$$

This loss is then used in backpropagation.
![](bp_loss.jpg)
## 6.1 An important mathematical relationship

When Softmax and Categorical Cross-Entropy are combined, the gradient with respect to each logit becomes\:

$$
\boxed{
\frac{\partial L}{\partial z_i}=\hat y_i-y_i
}
$$

This is a very important result in deep learning.

It means that the error signal at the output logits is simply\:

$$
\text{Predicted probability} -
\text{Actual probability}
$$

For example\:

$$
\hat y=[0.659,0.242,0.099]
$$

$$
y=[1,0,0]
$$

Therefore\:

$$
\frac{\partial L}{\partial z}
=
[-0.341,0.242,0.099]
$$

These gradients are passed backward through the neural network.

This is how the loss function becomes connected to the learning process.

---

# 7. Other Important Loss Functions in Deep Learning

Beyond regression and classification, deep learning uses several specialized loss functions.

## 7.1 Hinge Loss

Hinge Loss is commonly associated with Support Vector Machines, but it can also be used in neural networks for classification.

Its objective is not simply to classify correctly. It tries to maintain a margin between classes.

For binary labels\:

$$
y\in\{-1,+1\}
$$

The formula is\:

$$
\boxed{
L=\max(0,1-y\hat y)
}
$$

Here, $\hat y$ represents a real-valued model score.

### Example

If\:

$$
y=1,\quad \hat y=2
$$

Then\:

$$
L=\max(0,1-2)=0
$$

The prediction satisfies the margin.

But if\:

$$
y=1,\quad \hat y=0.5
$$

Then\:

$$
L=\max(0,1-0.5)=0.5
$$

There is a penalty because the prediction does not satisfy the required margin.

**Applications\:**

- Support Vector Machines.
- Margin-based classification.
- Some neural network classification tasks.

## 7.2 Kullback-Leibler (KL) Divergence

KL Divergence measures the difference between two probability distributions.

It is useful when a neural network needs to learn a target probability distribution rather than only a class label.

Formula\:

$$
\boxed{
D_{KL}(P||Q)=
\sum_i P(i)\log\frac{P(i)}{Q(i)}
}
$$

Where\:

- $P$ \= True or reference distribution.
- $Q$ \= Predicted or approximating distribution.

Applications\:

- Variational Autoencoders (VAEs).
- Knowledge distillation.
- Probabilistic machine learning.
- Distribution matching.

Important: KL divergence is not symmetric\:

$$
D_{KL}(P||Q)\neq D_{KL}(Q||P)
$$

## 7.3 Focal Loss

Focal Loss is particularly useful for highly imbalanced classification datasets.

Imagine a dataset containing\:

- 9,900 normal transactions.
- 100 fraudulent transactions.

A model might achieve high accuracy by predicting almost everything as normal.

However, it would fail to detect fraud.

Focal Loss addresses this issue by reducing the contribution of easily classified examples and emphasizing difficult examples.

For binary classification\:

$$
\boxed{
FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)
}
$$

Where\:

- $p_t$ \= Probability assigned to the correct class.
- $\alpha_t$ \= Class balancing factor.
- $\gamma$ \= Focusing parameter.

When a sample is correctly classified with high confidence, $p_t$ is close to 1, so\:

$$
(1-p_t)^\gamma\approx0
$$

Its loss becomes smaller.

When a sample is difficult to classify, its contribution remains relatively larger.

Applications\:

- Object detection.
- Fraud detection.
- Rare-event classification.
- Highly imbalanced image classification.

## 7.4 Dice Loss

Dice Loss is widely used in image segmentation.

Suppose we are building a model that identifies tumors in medical images.

The model must classify each pixel as\:

- Tumor.
- Background.

Dice Loss measures the overlap between predicted and actual segmentation regions.

Dice coefficient\:

$$
Dice=
\frac{2|A\cap B|}{|A|+|B|}
$$

Where\:

- $A$ \= Predicted segmentation.
- $B$ \= Actual segmentation.

Dice Loss\:

$$
\boxed{L_{Dice}=1-Dice}
$$

A higher Dice coefficient indicates better overlap, while a lower Dice Loss indicates better segmentation.

Applications\:

- Medical image segmentation.
- Satellite image segmentation.
- Object segmentation.

## 7.5 Contrastive Loss and Triplet Loss

These losses are commonly used in metric learning.

Instead of directly predicting a class, the model learns a useful representation (embedding) of the input.

For example, in face recognition, we want\:

- Embeddings of the same person's face to be close.
- Embeddings of different people's faces to be far apart.

### Contrastive Loss

It compares pairs of examples.

A common formulation is\:

$$
L= yD^2+(1-y)\max(0,m-D)^2
$$

Where\:

- $D$ \= Distance between two embeddings.
- $y=1$ for a similar pair.
- $y=0$ for a dissimilar pair.
- $m$ \= Margin.

### Triplet Loss

Triplet Loss uses three examples\:

1. Anchor: Reference example.
2. Positive: Similar example.
3. Negative: Different example.

Its formula is\:

$$
\boxed{
L=\max(0,D(a,p)-D(a,n)+\alpha)
}
$$

Where\:

- $D(a,p)$ \= Distance between anchor and positive.
- $D(a,n)$ \= Distance between anchor and negative.
- $\alpha$ \= Margin.

The objective is to make the anchor closer to the positive than to the negative by at least the margin.

Applications\:

- Face recognition.
- Image retrieval.
- Similarity search.
- Recommendation systems.

---

# 8. Comparison of Important Deep Learning Loss Functions

Here is a consolidated reference table.

| Loss function | Problem type | Main purpose |
|--------|--------|--------|
| MSE | Regression | Penalize squared errors |
| MAE | Regression | Minimize absolute errors |
| Huber | Regression | Handle outliers |
| MSLE | Regression | Penalize relative/log-scale errors |
| Binary CE | Binary classification | Probability-based classification |
| Categorical CE | Multi-class classification | Multi-class probability prediction |
| Sparse Categorical CE | Multi-class classification | Same as CCE, integer labels |
| Hinge Loss | Classification | Enforce classification margin |
| KL Divergence | Distribution learning | Compare probability distributions |
| Focal Loss | Imbalanced classification | Focus on difficult samples |
| Dice Loss | Segmentation | Measure region overlap |
| Contrastive Loss | Metric learning | Learn pairwise similarity |
| Triplet Loss | Metric learning | Learn relative embedding distances |

---

# 9. How Does a Loss Function Actually Help a Neural Network Learn?

This is probably the most important section because understanding the formulas alone is not enough.

Let's consider a simple neural network\:

$$
x\rightarrow w\rightarrow \hat y
$$

For simplicity, assume\:

$$
\hat y=wx
$$

And our loss function is MSE for one sample\:

$$
L=(y-\hat y)^2
$$

Suppose\:

$$
x=2,\quad y=10,\quad w=2
$$

### Step 1: Forward propagation

$$
\hat y=wx
$$

$$
\hat y=2\times2=4
$$

### Step 2: Calculate loss

$$
L=(10-4)^2
$$

$$
\boxed{L=36}
$$

Our model is making a significant prediction error.

### Step 3: Calculate the gradient

We want\:

$$
\frac{\partial L}{\partial w}
$$

Using the chain rule\:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\hat y}
\times
\frac{\partial\hat y}{\partial w}
$$

First\:

$$
\frac{\partial L}{\partial\hat y}
=2(\hat y-y)
$$

$$
=2(4-10)=-12
$$

Second\:

$$
\frac{\partial\hat y}{\partial w}=x=2
$$

Therefore\:

$$
\frac{\partial L}{\partial w}=-12\times2
$$

$$
\boxed{\frac{\partial L}{\partial w}=-24}
$$

### Step 4: Update the weight

Using Gradient Descent\:

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

Suppose learning rate\:

$$
\eta=0.1
$$

Then\:

$$
w_{new}=2-0.1(-24)
$$

$$
\boxed{w_{new}=4.4}
$$

### Step 5: Forward propagation again

$$
\hat y=4.4\times2=8.8
$$

New loss\:

$$
L=(10-8.8)^2
$$

$$
\boxed{L=1.44}
$$

Look at the improvement\:

**Before updating**

# 36

**After updating**

# 1.44

The loss decreased by 96% after one weight update in this example.

This is how a loss function guides neural network learning.

In actual MLPs, there are many weights and biases, and backpropagation calculates their gradients using the chain rule.

---

# 10. Loss Function vs Cost Function vs Error

These terms are related, but they are not always interchangeable.

### Error

Error is the difference between actual and predicted output.

$$
e=y-\hat y
$$

### Loss Function

Loss measures the error for an individual sample or prediction.

For example\:

$$
L=(y-\hat y)^2
$$

### Cost Function

Cost generally represents the aggregate loss across a dataset or batch.

For example, MSE\:

$$
J(\theta)=\frac1n\sum_{i=1}^{n}L_i
$$

In practice, many deep learning libraries use the word *loss* for both per-sample and averaged batch objectives.

---

# 11. How to Select the Correct Loss Function?

This is a very important practical consideration when building neural networks.

| Your problem | Recommended loss |
|--------|--------|
| House price prediction | MSE or MAE |
| Temperature prediction | MSE or Huber |
| Regression with outliers | Huber or MAE |
| Binary spam classification | Binary Cross-Entropy |
| Multi-class image classification | Categorical Cross-Entropy |
| Multi-class integer labels | Sparse Categorical Cross-Entropy |
| Imbalanced classification | Focal Loss or weighted Cross-Entropy |
| Medical image segmentation | Dice Loss or combined Dice + CE |
| Face recognition | Contrastive or Triplet Loss |
| Variational Autoencoder | Reconstruction loss + KL Divergence |

**One important point\:** The loss function is not selected based only on the final activation function. You should also consider label representation, the learning objective, data characteristics, and numerical stability.

---

# 12. Practical Implementation: Loss Functions in an MLP Using PyTorch

Let's implement a small neural network for binary classification.

We will use\:

- Input features.
- One hidden layer.
- ReLU activation.
- One output neuron.
- BCEWithLogitsLoss.
- Adam optimizer.

```python
import torch
import torch.nn as nn

# Create a simple MLP
class BinaryClassifier(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)

        )

    def forward(self, x):

        return self.network(x)

# Sample data
X = torch.randn(100, 5)

y = torch.randint(
    0, 2, (100, 1)
).float()

# Initialize model
model = BinaryClassifier(5)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
for epoch in range(100):

    # Forward propagation
    y_pred = model(X)

    # Calculate loss
    loss = criterion(y_pred, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}, "
            f"Loss: {loss.item():.4f}"
        )
```

Notice the training sequence\:

```python
y_pred = model(X)

loss = criterion(y_pred, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

This is the standard training pattern you will repeatedly encounter in PyTorch.

Also notice that we did not apply sigmoid inside the model. `BCEWithLogitsLoss` handles that internally in a numerically stable manner.

---

# 13. Important Concepts You Should Understand Next

Before moving forward, make sure you understand these relationships\:

1. **Loss function\:** Measures how incorrect predictions are.
2. **Gradient\:** Indicates how the loss changes when a parameter changes.
3. **Backpropagation\:** Calculates gradients for all trainable parameters.
4. **Gradient Descent\:** Updates parameters in the direction that reduces loss.
5. **Learning Rate\:** Controls the size of each parameter update.
6. **Batch Loss\:** Usually averages or otherwise aggregates losses across a batch.
7. **Regularization\:** Adds a penalty to the training objective to discourage undesirable model complexity.

One additional distinction: regularization terms such as L1 and L2 are often added to the data loss to form the total training objective\:

$$
L_{total}=L_{data}+\lambda L_{regularization}
$$

This is how neural networks can balance prediction accuracy with model complexity.

---
