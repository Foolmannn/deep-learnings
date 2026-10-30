# Perceptron Learning: How Does a Perceptron Actually Learn Using Loss Functions?

The important question is\:

**How does a perceptron know that its weights are wrong, and how does it decide how much to change them?**

This is the central idea behind learning in neural networks.

A perceptron learns by\:

1. Taking input features.
2. Calculating a weighted sum.
3. Producing a prediction.
4. Comparing its prediction with the actual output.
5. Calculating how wrong its prediction is using a loss function.
6. Updating weights and bias to reduce the loss.
7. Repeating this process over the training dataset.

However, there is an important distinction: the classical perceptron learning algorithm does not necessarily calculate a conventional probability-based loss such as binary cross-entropy. It uses a classification-error-driven update, which can also be derived from a particular loss function.

Let's understand both approaches mathematically and implement them from scratch.

---

# 1. First, understand what learning means

Consider a binary classification problem.

We want to predict whether a student passes or fails based on two features\:

- $x_1$\: Hours studied.
- $x_2$\: Hours slept.

Our dataset\:

| Hours studied | Hours slept | Actual output |
|--------|--------|--------|
| 1 | 2 | 0 |
| 2 | 3 | 0 |
| 5 | 6 | 1 |
| 7 | 8 | 1 |

Here\:

- 0 = Fail
- 1 = Pass

Initially, the perceptron does not know the correct relationship between the inputs and outputs.

It starts with some weights and bias.

For example\:

$$
w_1=0.5,\quad w_2=0.2,\quad b=-1
$$

When an input arrives, it calculates\:

$$
z=w_1x_1+w_2x_2+b
$$

Then applies the step activation function\:

$$
\hat y =
\begin{cases}
1 & z\geq 0\\
0 & z<0
\end{cases}
$$

Suppose the input is\:

$$
X=[2,3]
$$

The weighted sum is\:

$$
z=(0.5)(2)+(0.2)(3)-1
$$

$$
z=1+0.6-1=0.6
$$

Therefore\:

$$
\hat y=1
$$

But the actual output is\:

$$
y=0
$$

The model has made an incorrect prediction.

**Learning means modifying the parameters $w_1,w_2,b$ so that the model makes better predictions on future inputs.**

# 2. What is a loss function?

A loss function is a mathematical function that measures how incorrect a model's prediction is.

It takes the actual output and predicted output and returns a numerical value representing the prediction error.

Generally\:

$$
L(y,\hat y)
$$

Where\:

- $y$ \= Actual output.
- $\hat y$ \= Predicted output.
- $L$ \= Loss.

A higher loss generally indicates a worse prediction.

For example, consider\:

| Actual output | Predicted output | Error |
|--------|--------|--------|
| 1 | 1 | 0 |
| 1 | 0 | 1 |
| 0 | 0 | 0 |
| 0 | 1 | 1 |

The simplest loss for binary classification is zero-one loss\:

$$
L(y,\hat y)=
\begin{cases}
0 & y=\hat y\\
1 & y\neq\hat y
\end{cases}
$$

However, this loss has an important problem: it is not differentiable in the way required for ordinary gradient descent.

To understand why, we need to understand the relationship between activation functions, loss functions, and optimization.

# 3. Why is the step activation function a problem for gradient descent?

This is one of the most important concepts in understanding perceptron learning.

Remember the perceptron activation function\:

$$
\hat y=f(z)
$$

Where\:

$$
f(z)=
\begin{cases}
1 & z\geq0\\
0 & z<0
\end{cases}
$$

Let's visualize it.

Notice something important\:

- When $z<0$, output is always 0.
- When $z\geq0$, output is always 1.
- At the threshold, the function jumps suddenly.

Its derivative is\:

$$
f'(z)=0
$$

Almost everywhere, except at the discontinuity where it is undefined.

Why does this matter?

Because gradient descent requires derivatives to determine how the model's parameters should change.

Remember the weighted sum\:

$$
z=w_1x_1+w_2x_2+b
$$

To update the weights using gradient descent, we need\:

$$
\frac{\partial L}{\partial w_1}
$$

Using the chain rule\:

$$
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial \hat y}
\times
\frac{\partial \hat y}{\partial z}
\times
\frac{\partial z}{\partial w_1}
$$

But\:

$$
\frac{\partial \hat y}{\partial z}=0
$$

Therefore, the gradient becomes zero almost everywhere.

This means ordinary gradient descent cannot effectively learn through a hard step activation.

**This is why the classical perceptron uses a special update rule instead of ordinary gradient descent through the step function.**

Let's explore how that update rule is connected to a loss function.

# 4. The Perceptron Loss Function

The classical perceptron learning algorithm is commonly formulated using a perceptron loss function.

For this formulation, we represent the target labels as\:

$$
y\in\{-1,+1\}
$$

Instead of 0 and 1.

Our prediction is based on the sign of the weighted sum\:

$$
\hat y=\operatorname{sign}(w^Tx+b)
$$

Now consider\:

$$
z=w^Tx+b
$$

For a correctly classified sample\:

- If $y=+1$, then $z>0$\.
- If $y=-1$, then $z<0$\.

Both conditions can be represented as\:

$$
yz>0
$$

For an incorrectly classified sample\:

$$
yz<0
$$

The perceptron loss for one sample is\:

$$
\boxed{L(w,b)=\max(0,-y(w^Tx+b))}
$$

Let's understand this equation.

**Case 1: Correct classification**

Suppose\:

$$
y=+1,\quad z=3
$$

Then\:

$$
L=\max(0,-(1)(3))
$$

$$
L=\max(0,-3)=0
$$

There is no loss.

**Case 2: Incorrect classification**

Suppose\:

$$
y=+1,\quad z=-2
$$

Then\:

$$
L=\max(0,-(1)(-2))
$$

$$
L=\max(0,2)=2
$$

The loss is 2.

**Case 3: Negative sample correctly classified**

Suppose\:

$$
y=-1,\quad z=-4
$$

Then\:

$$
L=\max(0,-(-1)(-4))
$$

$$
L=\max(0,-4)=0
$$

Again, no loss.

**Perceptron loss behavior**

| Actual label | Weighted sum | Loss |
|--------|--------|--------|
| \+1 | \+3 | 0 |
| \+1 | \-2 | 2 |
| \-1 | \-4 | 0 |
| \-1 | \+3 | 3 |

The loss is zero for correctly classified samples and positive for incorrectly classified samples.

Notice that this loss does not measure the distance between the predicted binary label and the actual label. It measures how far the sample is on the wrong side of the decision boundary, in terms of its signed score.

# 5. How does the perceptron loss update weights?

Now we arrive at the actual learning mechanism.

Our loss function is\:

$$
L=\max(0,-y(w^Tx+b))
$$

For a misclassified sample, the loss is\:

$$
L=-y(w^Tx+b)
$$

We use gradient descent\:

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

And\:

$$
b_{new}=b_{old}-\eta\frac{\partial L}{\partial b}
$$

Let's calculate the derivatives.

First\:

$$
L=-y(w^Tx+b)
$$

Differentiating with respect to $w$\:

$$
\frac{\partial L}{\partial w}=-yx
$$

Differentiating with respect to $b$\:

$$
\frac{\partial L}{\partial b}=-y
$$

Substituting into gradient descent\:

$$
w_{new}=w_{old}-\eta(-yx)
$$

Therefore\:

$$
\boxed{w_{new}=w_{old}+\eta yx}
$$

Similarly\:

$$
b_{new}=b_{old}+\eta y
$$

These are the perceptron update rules when labels are encoded as $-1,+1$\.

Notice something interesting\:

**We have mathematically derived the perceptron trick from a loss function and gradient descent\!**

This is the connection between the perceptron trick and loss-based learning.

# 6. Numerical Example: Learning Through the Perceptron Loss

Let's understand the entire learning process with actual numbers.

Suppose we have\:

$$
X=[2,3]
$$

Actual class\:

$$
y=-1
$$

Initial parameters\:

$$
w=[0.5,0.2]
$$

$$
b=-1
$$

Learning rate\:

$$
\eta=0.1
$$

### Step 1: Forward propagation

Calculate the weighted sum\:

$$
z=w^TX+b
$$

$$
z=(0.5)(2)+(0.2)(3)-1
$$

$$
z=0.6
$$

Since $z$ is positive\:

$$
\hat y=+1
$$

But the actual label is $-1$\.

Therefore, the prediction is incorrect.

### Step 2: Calculate the loss

$$
L=\max(0,-yz)
$$

Substitute\:

$$
L=\max(0,-(-1)(0.6))
$$

$$
\boxed{L=0.6}
$$

### Step 3: Calculate the gradients

For a misclassified sample\:

$$
\frac{\partial L}{\partial w}=-yx
$$

$$
\frac{\partial L}{\partial w}
=-(-1)[2,3]
$$

$$
=[2,3]
$$

For the bias\:

$$
\frac{\partial L}{\partial b}=-y
$$

$$
\frac{\partial L}{\partial b}=1
$$

### Step 4: Update weights

Using gradient descent\:

$$
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
$$

$$
w_{new}=[0.5,0.2]-0.1[2,3]
$$

$$
\boxed{w_{new}=[0.3,-0.1]}
$$

Update bias\:

$$
b_{new}=-1-(0.1)(1)
$$

$$
\boxed{b_{new}=-1.1}
$$

### Step 5: Make another prediction

Use the updated parameters\:

$$
z=(0.3)(2)+(-0.1)(3)-1.1
$$

$$
z=0.6-0.3-1.1
$$

$$
z=-0.8
$$

Since $z<0$\:

$$
\hat y=-1
$$

(Prediction corrected)

Calculate the new loss\:

$$
L=\max(0,-(-1)(-0.8))
$$

$$
L=\max(0,-0.8)
$$

$$
\boxed{L=0}
$$

We have successfully reduced the loss from 0.6 to 0.

This is how a perceptron learns through its loss function.

---

# 7. What happens when we have multiple training samples?

So far, we have used one training sample.

But in machine learning, we generally have hundreds, thousands, or millions of samples.

For example\:

$$
X=
\begin{bmatrix}
x_{11}&x_{12}\\
x_{21}&x_{22}\\
x_{31}&x_{32}\\
x_{41}&x_{42}
\end{bmatrix}
$$

And\:

$$
y=[-1,-1,+1,+1]
$$

We need to calculate the loss for the entire dataset.

The total perceptron loss can be expressed as\:

$$
\boxed{
J(w,b)=\sum_{i=1}^{n}
\max(0,-y_i(w^Tx_i+b))
}
$$

Alternatively, we can use the average\:

$$
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}
\max(0,-y_i(w^Tx_i+b))
$$

Where\:

- $n$ \= Number of training samples.
- $x_i$ \= Individual input sample.
- $y_i$ \= Actual label.
- $w$ \= Weight vector.
- $b$ \= Bias.

The objective of training is\:

$$
\boxed{\min_{w,b}J(w,b)}
$$

In simple terms, we want to find the weights and bias that minimize the total loss across the training dataset.

## Three ways of updating parameters

There are different approaches to calculating and applying updates.

| Approach | How it works |
|--------|--------|
| Stochastic Gradient Descent (SGD) | Updates weights after each sample |
| Batch Gradient Descent | Calculates gradients using the entire dataset before updating |
| Mini-batch Gradient Descent | Calculates gradients using a small batch of samples |

The classical perceptron algorithm generally updates after each individual sample, making it an online or sample-wise learning algorithm.

# 8. Implementing Perceptron Learning Using a Loss Function

Now let's implement what we have learned.

We will use\:

- NumPy for mathematical calculations.
- A perceptron loss function.
- Gradient descent for parameter updates.
- The $-1,+1$ label representation.

### Step 1: Prepare the dataset

```python
import numpy as np

X = np.array([
    [2, 3],
    [1, 2],
    [3, 1],
    [6, 5],
    [7, 8],
    [8, 6]
])

# Labels must be -1 and +1
y = np.array([-1, -1, -1, 1, 1, 1])
```

### Step 2: Implement the perceptron

```python
class Perceptron:

    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def predict(self, X):

        z = np.dot(X, self.weights) + self.bias

        return np.where(z >= 0, 1, -1)

    def calculate_loss(self, X, y):

        z = np.dot(X, self.weights) + self.bias

        # Perceptron loss
        losses = np.maximum(0, -y * z)

        return np.mean(losses)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.loss_history = []

        for epoch in range(self.epochs):

            for i in range(n_samples):

                # Forward propagation
                z = np.dot(X[i], self.weights) + self.bias

                # Prediction
                y_pred = 1 if z >= 0 else -1

                # Update only if misclassified
                if y[i] * z <= 0:

                    # Calculate gradients
                    dw = -y[i] * X[i]
                    db = -y[i]

                    # Gradient descent
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db

            # Calculate loss after the epoch
            loss = self.calculate_loss(X, y)

            self.loss_history.append(loss)

            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

            # Stop if all samples are correctly classified
            if np.all(y * (np.dot(X, self.weights) + self.bias) > 0):
                break
```

### Step 3: Train the model

```python
model = Perceptron(
    lr=0.01,
    epochs=100
)

model.fit(X, y)
```

### Step 4: Make predictions

```python
predictions = model.predict(X)

print("Actual:", y)
print("Predicted:", predictions)
```

### Step 5: Visualize the loss

```python
import matplotlib.pyplot as plt

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("Perceptron Loss")
plt.title("Perceptron Learning")

plt.grid()
plt.show()
```

The loss curve shows how the model's parameters change during training.

**Note\:** The perceptron loss is not guaranteed to decrease after every individual sample update. A later update can increase the loss on an earlier sample. For linearly separable data, the algorithm eventually finds a separating solution under standard assumptions.

Also, notice that the training code uses the condition\:

```python
if y[i] * z <= 0:
```

This handles samples on the decision boundary as well as misclassified samples. It is a common convention for the classical perceptron algorithm.

---

# 9. But What About Binary Cross-Entropy Loss?

Now comes an important distinction.

You might already have encountered binary cross-entropy (BCE) while studying logistic regression.

Why don't we use BCE directly with the ordinary perceptron?

Let's understand.

The classical perceptron uses\:

- Step activation.
- Hard binary predictions.
- Perceptron loss.
- Error-based parameter updates.

But modern neural network classification commonly uses\:

- Sigmoid activation.
- Probability predictions.
- Binary cross-entropy loss.
- Gradient descent and backpropagation.

These are different learning approaches.

## 9.1 Sigmoid activation

Instead of the step activation, we can use the sigmoid function\:

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

The sigmoid function converts any real-valued input into a value between 0 and 1.

For example\:

| Weighted sum $z$ | Sigmoid output |
|--------|--------|
| \-3 | 0.047 |
| \-2 | 0.119 |
| \-1 | 0.269 |
| 0 | 0.500 |
| 1 | 0.731 |
| 2 | 0.881 |
| 3 | 0.953 |

Unlike the step function, sigmoid is differentiable.

This allows us to calculate gradients and update weights using ordinary gradient descent.

## 9.2 Binary cross-entropy loss

For binary classification, the binary cross-entropy loss is\:

$$
\boxed{
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
}
$$

Where\:

- $y$ \= Actual label (0 or 1).
- $\hat y$ \= Predicted probability.

Let's understand it with examples.

**Example 1: Correct positive prediction**

$$
y=1,\quad \hat y=0.9
$$

$$
L=-[1\log(0.9)+0\log(0.1)]
$$

$$
L=-\log(0.9)
$$

$$
L\approx0.105
$$

Very small loss.

**Example 2: Incorrect positive prediction**

$$
y=1,\quad \hat y=0.1
$$

$$
L=-\log(0.1)
$$

$$
L\approx2.303
$$

Very large loss.

**Example 3: Correct negative prediction**

$$
y=0,\quad \hat y=0.1
$$

$$
L=-\log(1-0.1)
$$

$$
L\approx0.105
$$

Again, very small loss.

Compare the predictions\:

| Actual | Predicted probability | BCE loss |
|--------|--------|--------|
| 1 | 0.90 | 0.105 |
| 1 | 0.10 | 2.303 |
| 0 | 0.10 | 0.105 |
| 0 | 0.90 | 2.303 |

BCE penalizes confident incorrect predictions much more heavily than uncertain predictions.

## 9.3 How does BCE update weights?

Let's derive the gradient.

Our forward propagation\:

$$
z=w^Tx+b
$$

$$
\hat y=\sigma(z)
$$

Our loss\:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

Using the chain rule, we get\:

$$
\boxed{\frac{\partial L}{\partial z}=\hat y-y}
$$

Since\:

$$
\frac{\partial z}{\partial w}=x
$$

We get\:

$$
\boxed{\frac{\partial L}{\partial w}=(\hat y-y)x}
$$

For the bias\:

$$
\boxed{\frac{\partial L}{\partial b}=\hat y-y}
$$

Finally, applying gradient descent\:

$$
\boxed{w=w-\eta(\hat y-y)x}
$$

$$
\boxed{b=b-\eta(\hat y-y)}
$$

This is the weight update rule for a single sigmoid neuron trained using binary cross-entropy.

Notice how this differs from the classical perceptron update.

The sigmoid-based model receives a continuous error signal, while the classical perceptron uses a hard classification error.

# 10. Perceptron Loss vs Binary Cross-Entropy Loss

This comparison is extremely important because it helps you understand the transition from a classical perceptron to modern neural networks.

| Feature | Perceptron Loss | Binary Cross-Entropy |
|--------|--------|--------|
| Common activation | Step | Sigmoid |
| Labels | Usually -1, +1 | 0, 1 |
| Prediction | Hard class | Probability |
| Loss | $\max(0,-yz)$ | $-[y\log p+(1-y)\log(1-p)]$ |
| Differentiability | Piecewise linear, not differentiable at the boundary | Differentiable with sigmoid |
| Gradient | Only misclassified samples receive updates | Correct and incorrect predictions can receive updates |
| Optimization | Perceptron update | Gradient descent |
| Usage | Classical perceptron | Logistic regression and binary neural network output layers |

One subtle but important point\:

The perceptron loss can have zero loss for a correctly classified sample even if it is extremely close to the decision boundary. It does not explicitly encourage a larger classification margin.

BCE, on the other hand, provides a continuous penalty based on predicted probability.

# 11. Implementing a Sigmoid Neuron With Binary Cross-Entropy

Let's implement a sigmoid-based neuron to see how modern gradient-based learning differs from the classical perceptron.

```python
import numpy as np

class SigmoidNeuron:

    def __init__(self, lr=0.1, epochs=1000):

        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        self.loss_history = []

        for epoch in range(self.epochs):

            # Forward propagation
            z = np.dot(X, self.weights) + self.bias

            y_pred = self.sigmoid(z)

            # Binary cross-entropy
            epsilon = 1e-15

            predictions = np.clip(
                y_pred,
                epsilon,
                1 - epsilon
            )

            loss = -np.mean(
                y * np.log(predictions)
                + (1-y) * np.log(1-predictions)
            )

            self.loss_history.append(loss)

            # Calculate gradients
            dw = np.dot(X.T, (y_pred-y)) / n_samples

            db = np.mean(y_pred-y)

            # Update parameters
            self.weights -= self.lr * dw

            self.bias -= self.lr * db

    def predict_proba(self, X):

        z = np.dot(X, self.weights) + self.bias

        return self.sigmoid(z)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)
```

Train it\:

```python
X = np.array([
    [2, 3],
    [1, 2],
    [3, 1],
    [6, 5],
    [7, 8],
    [8, 6]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = SigmoidNeuron(
    lr=0.1,
    epochs=1000
)

model.fit(X, y)

print(model.predict(X))

print(model.predict_proba(X))
```

Visualize its loss\:

```python
import matplotlib.pyplot as plt

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")

plt.title("Learning Using Binary Cross Entropy")

plt.grid()
plt.show()
```

Notice the difference\:

- In the classical perceptron, the prediction is a hard class.
- In the sigmoid neuron, the prediction is a probability.
- BCE measures the quality of that probability.
- Gradient descent adjusts the weights to reduce BCE.

Technically, this sigmoid neuron is a logistic-regression-style single neuron rather than a classical step-activation perceptron.

---

# 12. How Does This Connect to Deep Learning?

Now let's connect everything to the neural networks you will study next.

A neural network contains multiple neurons organized into layers.

Each neuron performs a calculation\:

$$
z=w^Tx+b
$$

Then applies an activation function\:

$$
a=f(z)
$$

For example, a simple neural network might have\:

During training, the network performs two major processes.

**1\. Forward propagation**

Information travels from input to output.

$$
X\rightarrow Z\rightarrow A\rightarrow\hat y
$$

The network generates a prediction.

**2\. Backpropagation**

The network calculates how much each parameter contributed to the loss.

It calculates gradients using the chain rule and propagates those gradients backward through the network.

Then the optimizer updates weights\:

$$
\boxed{W_{new}=W_{old}-\eta\frac{\partial L}{\partial W}}
$$

$$
\boxed{b_{new}=b_{old}-\eta\frac{\partial L}{\partial b}}
$$

This process repeats over multiple epochs.

The important difference is that a single-layer sigmoid neuron can calculate its gradients directly, whereas a multi-layer neural network requires backpropagation to calculate gradients for all its layers.

# 13. Complete Comparison: How Different Models Learn

You have already studied linear regression, logistic regression, and now perceptrons. Let's connect them.

| Model | Activation / output | Loss function | Learning method |
|--------|--------|--------|--------|
| Linear Regression | Linear | MSE | Gradient descent / normal equation |
| Logistic Regression | Sigmoid | Binary Cross-Entropy | Gradient descent |
| Classical Perceptron | Step | Perceptron loss | Perceptron update rule |
| Sigmoid Neuron | Sigmoid | Binary Cross-Entropy | Gradient descent |
| Multi-Layer Perceptron | ReLU, sigmoid, etc. | Depends on task | Backpropagation + optimizer |

This table is worth keeping in your notes.

# 14. Final Understanding: How Does a Perceptron Actually Learn?

Let's summarize the entire learning process.

## Classical perceptron learning pipeline

**1\. Input features**

$X=[x_1,x_2,\ldots,x_n]$

**2\. Weighted sum**

$z=w^TX+b$

**3\. Step activation**

$\hat y=\operatorname{sign}(z)$

**4\. Calculate prediction loss**

$L=\max(0,-yz)$

**5\. Update weights and bias**

$w=w+\eta yX$

$b=b+\eta y$

**6\. Repeat for training samples and epochs**

**Continue until convergence or stopping condition**

For a sigmoid neuron trained with BCE, steps 3–5 are different: it produces probabilities, calculates BCE, and uses gradients to update parameters.

## 15. Important points for revision

1. A perceptron learns by adjusting weights and bias based on prediction errors.
2. A loss function measures how incorrect a prediction is.
3. Classical perceptron learning uses a perceptron-specific loss and update rule.
4. The hard step activation has zero derivative almost everywhere, so ordinary gradient descent cannot train through it.
5. The perceptron update can be derived using the gradient of perceptron loss for misclassified samples.
6. Sigmoid activation and BCE allow continuous gradient-based learning.
7. Gradient descent updates parameters in the direction that reduces the loss.
8. Backpropagation extends this gradient calculation to multiple layers.

---

## What I recommend studying next

You now have the foundation for understanding how neural networks learn.

The next logical topic is **Multi-Layer Perceptron (MLP) and Forward Propagation**\.

I recommend studying it in this order\:

1. Single neuron vs multiple neurons.
2. Input layer, hidden layer, and output layer.
3. How weights and biases work between layers.
4. Forward propagation with numerical examples.
5. Why hidden layers are needed.
6. How MLP solves XOR.
7. Implementing an MLP from scratch using NumPy.

Once you understand forward propagation, backpropagation will make much more sense because you will already understand how the network generates its predictions.

![Activation Functions in Deep Learning](https://images.openai.com/static-rsc-4/Ca2iQ7anc1VEP5v_DX7sG9UX9SUvd2mCzh1AILSRQA6a8acYenWiz5HiRYRmcZXYQW9HIaQAfmPUIr5YAirZcdpzAhzZwcYZU_AjHLSNmKaGE2miOC2OUaA2jYh5j40bcgq0XBH3GGIO3kgAxOCxFBHeFlKa3wF35uKtJKBIchQqEFAiA40lxRABkEuOghk0?purpose=fullsize)

![Perceptron : A Deep Dive into Foundational Neural Network Algorithm | by Madhav Reddy | Medium](https://images.openai.com/static-rsc-4/8qDJb5HEuuaslZBxGt6HOUI1W7-4XIy8XmB7zrAldOkMqkuYfq3XE7RkTicxOlpFR93YGMCQ-PCkVQRtWI7rKTrgFBOWdQqZ55xuqbIzOQnCAS93nX9jRIKuJHrwykBdbeNNjQMDpDINPRuU1sfiKs0gVKO0KGdD_BcV2kZTU_ZB_qou2djm3kaa8Hl8E6tq?purpose=fullsize)

![Logistic Regression, the Sigmoid Function, and Decision Boundaries (Made Intuitive) | by Ayush Saxena | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/qg4-Rm9vgZinvUYiKwTl7PgDJFReKNU0XcyMN4wYUiDjlP49F47ihCylXVptpdwD7JO2KvW8SpzxshXtoP9ZFbzkl__4QNt2Ai6WJIlokicvOa2Z0khHtiDrQu-S8620zcPeRywR3TjqOHAmyIJF4TMi4gWmmE_Tw0XpgHDPr20ItKK0amVzzB46_Qi5u83R?purpose=fullsize)

![AiDL0: What-Why-How Neural Networks — Part A | by Arun Singh | Dec, 2025 | Medium](https://images.openai.com/static-rsc-4/YuLZPQsnJnWDn06OlQbOTE2tLhbUqj6qhSBVSxwyhPdw13hAwzidz4UvDPZ0rDIrun5N98ZPzAg95XKyHQiMifmmPibumVga0vDO_Rrtmokl-ErNTgg19I9B1DPPNsooTkxAOLkD4hJyvkqs1O0Yg9NQnRZMUmoHC37oEQvsUzFCRW7FhXsHaYAkoOz7eK7W?purpose=fullsize)

---
Powered by [AI Exporter](https://saveai.net)