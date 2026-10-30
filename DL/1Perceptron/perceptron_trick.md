# The Perceptron Trick in Deep Learning – Detailed Study Notes

The **Perceptron Trick** is an important concept in the foundation of neural networks and deep learning. It explains how a perceptron learns by adjusting its weights and bias whenever it makes an incorrect prediction.

Before understanding the perceptron learning algorithm mathematically, the perceptron trick helps us develop an intuitive understanding of how a model improves its decision boundary.

Let's study it step by step.

# 1. What is the Perceptron Trick?

The perceptron trick is a simple learning technique used to train a perceptron for binary classification.

Its main idea is\:

**Whenever the perceptron makes an incorrect prediction, adjust its weights and bias so that the decision boundary moves in a direction that makes the incorrect prediction more likely to become correct.**

In other words\:

- If a positive sample is incorrectly classified as negative, adjust the weights to increase its predicted score.
- If a negative sample is incorrectly classified as positive, adjust the weights to decrease its predicted score.
- If the prediction is correct, no adjustment is required.

The perceptron repeatedly performs these adjustments until it correctly classifies the training samples or reaches a predefined number of iterations.

# 2. Understanding the Perceptron Before the Trick

A perceptron is the simplest form of an artificial neural network.

It takes multiple input features, multiplies them by corresponding weights, adds a bias, and produces an output.

Consider a dataset containing two features\:

- $x_1$\: Hours studied
- $x_2$\: Hours slept

We want to predict whether a student passes or fails.

The perceptron calculates\:

$$
z=w_1x_1+w_2x_2+b
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $x_1,x_2$ | Input features |
| $w_1,w_2$ | Weights |
| $b$ | Bias |
| $z$ | Weighted sum |

The weighted sum is passed through a step activation function.

$$
\hat y =
\begin{cases}
1 & z\geq 0\\
0 & z<0
\end{cases}
$$

Here\:

- 1 represents the positive class.
- 0 represents the negative class.

### Example

Suppose our perceptron has\:

$$
w_1=2,\quad w_2=1,\quad b=-5
$$

And the input is\:

$$
X=(3,2)
$$

Calculate\:

$$
z=(2)(3)+(1)(2)-5
$$

$$
z=6+2-5=3
$$

Since $z\geq0$\:

$$
\hat y=1
$$

The perceptron predicts the positive class.

But how does it learn these weights in the first place?

This is where the perceptron trick comes in.

# 3. Understanding the Decision Boundary

Before implementing the perceptron trick, you should understand the decision boundary.

The decision boundary is the line or hyperplane that separates different classes.

For two input features\:

$$
w_1x_1+w_2x_2+b=0
$$

This equation represents a straight line.

Rearranging\:

$$
x_2=-\frac{w_1}{w_2}x_1-\frac{b}{w_2}
$$

This is the familiar straight-line equation\:

$$
y=mx+c
$$

Where\:

- Slope: $-w_1/w_2$
- Intercept: $-b/w_2$

The perceptron classifies points based on which side of this line they fall.


![Machine Learning ](p0.jpg)

The objective of training is to find suitable weights and bias so that the decision boundary separates the two classes correctly.

The perceptron trick helps us move this boundary whenever a sample is incorrectly classified.

# 4. The Main Idea Behind the Perceptron Trick

Let's understand the actual trick using a simple example.

Suppose we have a perceptron with\:

$$
w_1=2,\quad w_2=1,\quad b=-5
$$

Consider a positive training sample\:

$$
X=(1,1),\quad y=1
$$

The perceptron calculates\:

$$
z=(2)(1)+(1)(1)-5
$$

$$
z=-2
$$

Since $z<0$\:

$$
\hat y=0
$$

But the actual output is\:

$$
y=1
$$

Therefore, the perceptron incorrectly classified the positive sample.

We need to adjust the weights so that this sample produces a higher weighted sum.

The perceptron trick does exactly that.

## Case 1: Positive sample incorrectly classified as negative

Suppose\:

$$
y=1,\quad \hat y=0
$$

We want to increase the weighted sum.

Therefore, we update the weights by adding the input features\:

$$
w_i^{new}=w_i^{old}+\eta x_i
$$

And update the bias\:

$$
b^{new}=b^{old}+\eta
$$

Where\:

- $\eta$ \= Learning rate
- $x_i$ \= Input feature
- $w_i$ \= Corresponding weight

Let's assume the learning rate is\:

$$
\eta=1
$$

Our original parameters are\:

$$
w_1=2,\quad w_2=1,\quad b=-5
$$

Input\:

$$
X=(1,1)
$$

Updated weights\:

$$
w_1^{new}=2+(1)(1)=3
$$

$$
w_2^{new}=1+(1)(1)=2
$$

Updated bias\:

$$
b^{new}=-5+1=-4
$$

Our new parameters are\:

$$
w_1=3,\quad w_2=2,\quad b=-4
$$

Let's check the prediction again\:

$$
z=(3)(1)+(2)(1)-4
$$

$$
z=1
$$

Since $z\geq0$\:

$$
\hat y=1
$$

(Correct prediction)

The perceptron successfully corrected its prediction by adjusting its weights and bias.

## Case 2: Negative sample incorrectly classified as positive

Now consider the opposite situation.

Suppose\:

$$
y=0,\quad \hat y=1
$$

The perceptron predicted positive, but the actual class is negative.

We need to decrease the weighted sum.

Therefore, we subtract the input features from the weights\:

$$
w_i^{new}=w_i^{old}-\eta x_i
$$

And update the bias\:

$$
b^{new}=b^{old}-\eta
$$

### Example

Suppose\:

$$
w_1=2,\quad w_2=1,\quad b=-1
$$

Input\:

$$
X=(2,1)
$$

Actual output\:

$$
y=0
$$

Calculate the weighted sum\:

$$
z=(2)(2)+(1)(1)-1
$$

$$
z=4
$$

Since $z\geq0$\:

$$
\hat y=1
$$

The prediction is incorrect.

Assume\:

$$
\eta=1
$$

Update the weights\:

$$
w_1^{new}=2-2=0
$$

$$
w_2^{new}=1-1=0
$$

Update the bias\:

$$
b^{new}=-1-1=-2
$$

Our new parameters are\:

$$
w_1=0,\quad w_2=0,\quad b=-2
$$

Calculate again\:

$$
z=(0)(2)+(0)(1)-2
$$

$$
z=-2
$$

Therefore\:

$$
\hat y=0
$$

(Correct prediction)

The perceptron has successfully corrected the classification again.

# 5. General Mathematical Formula of the Perceptron Trick

We can combine both cases into one mathematical equation.

The weight update rule is\:

$$
\boxed{w_i^{new}=w_i^{old}+\eta(y-\hat y)x_i}
$$

Similarly, the bias update rule is\:

$$
\boxed{b^{new}=b^{old}+\eta(y-\hat y)}
$$

Where\:

| Parameter | Description |
|--------|--------|
| $w_i$ | Current weight |
| $\eta$ | Learning rate |
| $y$ | Actual output |
| $\hat y$ | Predicted output |
| $x_i$ | Input feature |
| $b$ | Bias |

Let's verify the formula for both cases.

**Case 1: Positive sample classified as negative**

$$
y=1,\quad \hat y=0
$$

$$
y-\hat y=1
$$

Therefore\:

$$
w_i^{new}=w_i^{old}+\eta x_i
$$

We add the input to the weights.

**Case 2: Negative sample classified as positive**

$$
y=0,\quad \hat y=1
$$

$$
y-\hat y=-1
$$

Therefore\:

$$
w_i^{new}=w_i^{old}-\eta x_i
$$

We subtract the input from the weights.

**Case 3: Correct prediction**

If\:

$$
y=\hat y
$$

Then\:

$$
y-\hat y=0
$$

Therefore\:

$$
w_i^{new}=w_i^{old}
$$

$$
b^{new}=b^{old}
$$

No update is required.

# 6. Geometrical Understanding of the Perceptron Trick

This is one of the most important parts to understand.

The perceptron trick does not directly move individual data points. Instead, it changes the weights and bias, which changes the decision boundary.

Consider two classes\:

- Positive class: Blue points
- Negative class: Red points

Initially, the decision boundary might incorrectly classify several points.

![The Perceptron: A Foundational Analysis of the First Learning Machine - perfecXion.ai](p1.jpg)


Now, suppose a positive point is incorrectly classified as negative.

The algorithm increases the weighted sum for that point by updating the weights.

As a result, the decision boundary shifts or rotates in a direction that makes the point more likely to be classified as positive.

![Lecture 3: The Perceptron](p2.jpg)



Similarly, when a negative point is incorrectly classified as positive, the algorithm adjusts the parameters in the opposite direction.

After multiple updates, the decision boundary may correctly separate all training samples.
![Day #14: The Dawn of Deep Learning — The Perceptron: Simple Origins, Profound Future | by Monica Ashok | Medium](p3.jpg)


**Important\:** The boundary does not necessarily move parallel to itself. Depending on the input features and the weight updates, it can shift, rotate, or both.

# 7. Perceptron Trick vs Perceptron Learning Algorithm

These two terms are closely related.

The perceptron trick refers to the basic idea of adjusting weights and bias based on incorrect predictions.

The perceptron learning algorithm applies this idea systematically to the entire training dataset.

The algorithm follows these steps\:

1. Initialize weights and bias, usually with zeros or small values.
2. Select a training sample.
3. Calculate the weighted sum.
4. Apply the step activation function.
5. Compare the predicted output with the actual output.
6. Update weights and bias if the prediction is incorrect.
7. Repeat the process for multiple epochs.
8. Stop when all samples are correctly classified or the maximum number of epochs is reached.

### Mathematical representation

For a dataset containing $n$ samples and $m$ features\:

$$
X\in \mathbb{R}^{n\times m}
$$

$$
y\in\{0,1\}^{n}
$$

For each sample\:

$$
z=w^Tx+b
$$

$$
\hat y=
\begin{cases}
1 & z\geq0\\
0 & z<0
\end{cases}
$$

Update\:

$$
w=w+\eta(y-\hat y)x
$$

$$
b=b+\eta(y-\hat y)
$$

# 8. Python Implementation From Scratch

Now let's implement the perceptron trick using NumPy.

We will create a simple binary classification dataset.

### Step 1: Import libraries

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Create a dataset

```python
X = np.array([
    [2, 3],
    [1, 1],
    [3, 2],
    [6, 5],
    [7, 8],
    [8, 6]
])

y = np.array([0, 0, 0, 1, 1, 1])
```

Here\:

- First three samples belong to class 0.
- Last three samples belong to class 1.

### Step 3: Implement the perceptron

```python
class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            errors = 0

            for i in range(n_samples):

                # Calculate weighted sum
                z = np.dot(X[i], self.weights) + self.bias

                # Make prediction
                y_pred = self.activation(z)

                # Calculate error
                error = y[i] - y_pred

                # Update weights and bias
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

                if error != 0:
                    errors += 1

            print(f"Epoch {epoch+1}, Errors: {errors}")

            # Stop if all samples are correctly classified
            if errors == 0:
                break

    def predict(self, X):

        z = np.dot(X, self.weights) + self.bias

        return self.activation(z)
```

### Step 4: Train the model

```python
model = Perceptron(
    learning_rate=0.1,
    epochs=100
)

model.fit(X, y)
```

### Step 5: Make predictions

```python
predictions = model.predict(X)

print("Actual:", y)
print("Predicted:", predictions)
```

Expected result\:

```text
Actual:    [0 0 0 1 1 1]
Predicted: [0 0 0 1 1 1]
```

The perceptron learns the decision boundary by repeatedly applying the perceptron trick.

# 9. Visualizing the Decision Boundary Using Python

Let's visualize how the model separates the two classes.

```python
plt.figure(figsize=(8, 6))

# Plot class 0
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='red',
    label='Class 0'
)

# Plot class 1
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='blue',
    label='Class 1'
)

# Decision boundary
w1, w2 = model.weights

x_values = np.linspace(0, 10, 100)

if w2 != 0:
    y_values = -(w1 * x_values + model.bias) / w2

    plt.plot(
        x_values,
        y_values,
        label='Decision Boundary'
    )

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.title('Perceptron Decision Boundary')

plt.legend()
plt.grid()
plt.show()
```

This plot helps you understand how the learned weights and bias define the decision boundary.

# 10. What Happens When the Dataset Is Not Linearly Separable?

This is an important limitation of the perceptron.

The perceptron can successfully learn a decision boundary when the dataset is linearly separable.

But what happens when the dataset cannot be separated using a straight line?

Consider the XOR dataset.

| $x_1$ | $x_2$ | Output |
|--------|--------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |


![Multilayer Perceptrons: Architecture, Forward Pass & PyTorch - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](p4.jpg)

A single straight line cannot separate these two classes.

Therefore, a single-layer perceptron cannot learn the XOR function.

Even if we repeatedly apply the perceptron trick, the model cannot find a perfect decision boundary.

This leads us to an important development in deep learning\:

**Multi-Layer Perceptrons (MLPs).**

By adding hidden layers and nonlinear activation functions, neural networks can learn nonlinear decision boundaries.

# 11. Limitations of the Perceptron Trick

| Limitation | Explanation |
|--------|--------|
| Linear classification | A single perceptron can only learn linear decision boundaries. |
| XOR problem | Cannot solve XOR using a single-layer perceptron. |
| No probability output | Step activation produces only binary predictions. |
| Sensitive to data order | Different sample orders can produce different learned weights. |
| No optimal margin guarantee | It finds a separating boundary but does not necessarily maximize the margin. |
| Convergence limitations | It is not guaranteed to converge for non-linearly separable datasets. |

One important theoretical result is the **Perceptron Convergence Theorem**\.

It states that if a dataset is linearly separable, the perceptron learning algorithm will converge to a solution after a finite number of updates, assuming an appropriate fixed positive learning rate.

# 12. Perceptron Trick vs Gradient Descent

You might wonder whether the perceptron trick is the same as gradient descent.

They are related in that both update model parameters to improve predictions, but their update mechanisms are different.

| Perceptron Trick | Gradient Descent |
|--------|--------|
| Uses classification errors | Uses gradients of a loss function |
| Uses a step activation | Commonly uses differentiable activations |
| Updates when prediction is incorrect | Usually updates based on calculated gradients |
| Simple update rule | Uses calculus and the chain rule |
| Primarily used in a basic perceptron | Used to train many ML and DL models |

The perceptron update can also be understood through a perceptron loss function.

For labels represented as $-1$ and $+1$, define\:

$$
L(w,b)=\max(0,-y(w^Tx+b))
$$

For a misclassified sample, this loss is positive. Its gradient-based update produces the perceptron update rule.

This connects the perceptron trick to the broader idea of optimization in machine learning.

---

# 13. Important Concepts to Remember

Here is a quick revision summary.

| Concept | Key point |
|--------|--------|
| Perceptron | Simplest artificial neural network for binary classification |
| Weighted sum | $z=w^Tx+b$ |
| Activation | Step function |
| Decision boundary | $w^Tx+b=0$ |
| Positive misclassification | Increase weights using input features |
| Negative misclassification | Decrease weights using input features |
| Weight update | $w=w+\eta(y-\hat y)x$ |
| Bias update | $b=b+\eta(y-\hat y)$ |
| Learning rate | Controls update magnitude |
| Convergence | Guaranteed for linearly separable data under standard assumptions |
| XOR | Cannot be solved by a single-layer perceptron |


The most important next topic is **Multi-Layer Perceptron (MLP)** because it extends the single perceptron into a network capable of learning nonlinear patterns.


---
