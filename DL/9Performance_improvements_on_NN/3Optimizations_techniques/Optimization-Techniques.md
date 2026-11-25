#  Optimization Techniques in Deep Learning (Detailed Notes)

## 1. Introduction to Optimization Techniques

Optimization techniques are methods used in Deep Learning to improve the process of training neural networks by finding the optimal values of weights and biases that minimize the loss function.

During neural network training, the model makes predictions using its current weights and biases. These predictions are compared with the actual target values to calculate the loss (error). The optimization algorithm then adjusts the weights and biases to reduce this error.

The main purpose of optimization is to find a set of parameters that produces the minimum possible loss and enables the neural network to make accurate predictions.

### Why Are Optimization Techniques Needed?

Training a deep neural network involves millions of parameters, and selecting appropriate values for these parameters is a challenging task. Optimization techniques make this process more efficient.

They are needed for the following reasons\:

1. **Minimizing Loss\:** They help reduce the difference between predicted and actual outputs.
2. **Faster Training\:** They accelerate the process of finding suitable weights and biases.
3. **Improving Convergence\:** They help the model reach a minimum of the loss function more efficiently.
4. **Training Stability\:** They reduce unnecessary fluctuations in weight updates.
5. **Handling Complex Networks\:** They make it easier to train deep neural networks containing many layers and parameters.
6. **Improving Model Performance\:** Proper optimization helps the network learn meaningful patterns from training data.

### Basic Working Process of Optimization

Optimization works together with forward propagation and backpropagation.
![](../../images/nn_training_loop.jpg)

The process can be explained through the following steps\:

1. **Weight Initialization\:** The network initializes weights and biases with suitable initial values.
2. **Forward Propagation\:** Input data passes through the network, and the model generates predictions.
3. **Loss Calculation\:** The predicted outputs are compared with actual outputs using a loss function.
4. **Backpropagation\:** The gradients of the loss function with respect to the weights and biases are calculated.
5. **Optimization\:** The optimizer uses the calculated gradients to update the weights and biases.
6. **Iteration\:** The process repeats over multiple batches and epochs until the model learns an appropriate set of parameters.

The overall objective is\:

$$
\boxed{\min_{W,b} L(W,b)}
$$

Where\:

- $W$ \= Weights of the neural network.
- $b$ \= Biases of the neural network.
- $L$ \= Loss function.

The major optimization techniques used in Deep Learning are explained below.

---

## 2. Gradient Descent Optimization

### Definition

Gradient Descent is one of the most fundamental optimization algorithms in Deep Learning. It is used to minimize the loss function by repeatedly adjusting the weights and biases in the opposite direction of the calculated gradients.

The gradient represents the direction in which the loss function increases most rapidly. Therefore, moving in the opposite direction of the gradient helps reduce the loss.

In simple terms, Gradient Descent helps a neural network determine how to change its weights so that its predictions become more accurate.

### Understanding Gradient Descent

Consider a neural network that initially makes incorrect predictions. Its loss function has a relatively high value.

The purpose of Gradient Descent is to find the values of weights and biases that produce the lowest possible loss.

Imagine standing on a hill and trying to reach its lowest point. You observe the slope around your current position and take a step downhill. You repeat this process until you reach a low point.

Gradient Descent follows a similar principle\:

- The current position represents the current weights.
- The height represents the loss value.
- The slope represents the gradient.
- The downhill direction represents the direction of weight updates.
- The lowest point represents a minimum of the loss function.
![](../../images/gd_optimization.jpg)

### Mathematical Working of Gradient Descent

Suppose the loss function is represented by\:

$$
L(W)
$$

We want to find the weight $W$ that minimizes this loss.

The gradient of the loss function with respect to the weight is\:

$$
\frac{\partial L}{\partial W}
$$

The gradient tells us how the loss changes when the weight changes.

The weight update formula is\:

$$
\boxed{W_{new}=W_{old}-\eta
\frac{\partial L}{\partial W}}
$$

Where\:

- $W_{new}$ \= Updated weight.
- $W_{old}$ \= Previous weight.
- $\eta$ \= Learning rate.
- $\frac{\partial L}{\partial W}$ \= Gradient of the loss function.

The negative sign indicates that weights are updated in the opposite direction of the gradient.

The bias is updated similarly\:

$$
b_{new}=b_{old}-\eta
\frac{\partial L}{\partial b}
$$

### Step-by-Step Example

Suppose\:

- Initial weight = 5
- Learning rate = 0.1
- Gradient = 2

Using the Gradient Descent formula\:

$$
W_{new}=W_{old}-\eta \times Gradient
$$

Substituting the values\:

$$
W_{new}=5-(0.1\times2)
$$

$$
\boxed{W_{new}=4.8}
$$

Therefore, the weight changes from 5 to 4.8.

This update is performed repeatedly using newly calculated gradients until the loss becomes sufficiently small.

### Types of Gradient Descent

There are three major types of Gradient Descent based on how much training data is used to calculate the gradient.

#### A. Batch Gradient Descent

**Definition\:**

Batch Gradient Descent calculates the gradient using the entire training dataset before updating the weights and biases.

For example, if a dataset contains 10,000 training samples, the algorithm processes all 10,000 samples before performing one weight update.

**Working Process\:**

1. Initialize the network weights.
2. Pass the entire training dataset through the network.
3. Calculate the loss for all training samples.
4. Calculate the average gradient using the entire dataset.
5. Update the weights and biases.
6. Repeat the process for the next epoch.

**Advantages\:**

- Produces relatively stable gradient updates.
- Uses the entire dataset to calculate each update.
- Provides a more accurate estimate of the overall training gradient.
- Suitable for smaller datasets.

**Disadvantages\:**

- Computationally expensive for large datasets.
- Requires more memory.
- Weight updates happen less frequently.
- Training can be slow when the dataset is very large.

#### B. Stochastic Gradient Descent (SGD)

**Definition\:**

Stochastic Gradient Descent updates the weights and biases using the gradient calculated from only one training sample at a time.

Unlike Batch Gradient Descent, it does not wait for the entire dataset to be processed before updating the parameters.

For example, if the dataset contains 10,000 samples, SGD performs an individual weight update after processing each sample.

**Working Process\:**

1. Initialize the network weights.
2. Select one training sample.
3. Perform forward propagation.
4. Calculate the loss for that sample.
5. Perform backpropagation to calculate its gradient.
6. Update the weights and biases immediately.
7. Repeat the process for the remaining samples.

**Advantages\:**

- Requires less memory per update.
- Performs frequent weight updates.
- Can start learning quickly.
- Noise in updates can sometimes help escape saddle points or shallow local minima.

**Disadvantages\:**

- Weight updates are noisy.
- Loss may fluctuate significantly.
- May take longer to settle near a minimum.
- Requires careful learning rate selection.

#### C. Mini-Batch Gradient Descent

**Definition\:**

Mini-Batch Gradient Descent divides the entire training dataset into smaller groups called mini-batches and calculates gradients using one mini-batch at a time.

It combines the advantages of Batch Gradient Descent and Stochastic Gradient Descent.

For example, if a dataset contains 10,000 samples and the batch size is 100, the dataset is divided into 100 mini-batches.

The network updates its weights after processing each mini-batch.

**Working Process\:**

1. Initialize the network weights.
2. Divide the training dataset into smaller batches.
3. Select one mini-batch.
4. Perform forward propagation.
5. Calculate the average loss for the batch.
6. Perform backpropagation.
7. Update weights and biases.
8. Repeat for all remaining mini-batches.

**Advantages\:**

- Balances computational efficiency and update stability.
- Requires less memory than processing the entire dataset at once.
- Supports efficient parallel computation using GPUs.
- Provides more frequent updates than Batch Gradient Descent.

**Disadvantages\:**

- Batch size must be selected appropriately.
- Updates still contain some noise.
- Performance can depend on batch size.

### Comparison of Gradient Descent Types

| Feature | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|--------|--------|--------|
| Data used per update | Entire dataset | One sample | Small batch |
| Update frequency | Low | Very high | Moderate |
| Memory requirement | High | Low | Moderate |
| Stability | High | Low | Relatively high |
| Training speed | Can be slow | Fast updates | Efficient |
| Common usage | Small datasets | Online learning | Deep Learning |

![](../../images/gd_types.jpg)

**Important\:** Mini-Batch Gradient Descent is commonly used in modern Deep Learning because it offers a practical balance between computational efficiency and gradient stability.

---

## 3. Learning Rate Optimization

### Definition

Learning Rate Optimization is the process of selecting an appropriate learning rate to control how much the weights and biases of a neural network change during each optimization step.

The learning rate is one of the most important hyperparameters in neural network training because it directly affects the speed and stability of learning.

A learning rate that is too high may cause the model to overshoot the minimum, while a learning rate that is too low may make training extremely slow.

### Understanding Learning Rate

During Gradient Descent, the learning rate determines the size of each step taken toward minimizing the loss function.

The weight update formula is\:

$$
W_{new}=W_{old}-\eta\nabla L(W)
$$

Here, $\eta$ represents the learning rate.

For example, consider a neural network with a learning rate of 0.01.

The optimizer uses 0.01 as the scaling factor for the calculated gradient when updating the weight.

A larger learning rate produces larger updates for the same gradient, while a smaller learning rate produces smaller updates.

### Effects of Different Learning Rates

#### A. Learning Rate Too High

When the learning rate is excessively high, the optimizer takes very large steps during weight updates.

As a result\:

- It may jump over the minimum point.
- The loss function may fluctuate significantly.
- The optimizer may fail to converge.
- Training may become unstable.
- The loss may increase instead of decreasing.

In extreme situations, the loss can become extremely large or even result in numerical errors such as NaN.

#### B. Learning Rate Too Low

When the learning rate is excessively low, the optimizer takes very small steps toward minimizing the loss.

As a result\:

- Training becomes very slow.
- The model requires many iterations to converge.
- Computational resources are wasted.
- The model may spend a long time in flat regions of the loss function.

#### C. Appropriate Learning Rate

An appropriate learning rate allows the optimizer to make sufficiently large updates without overshooting the minimum.

Benefits include\:

- Faster convergence.
- More stable training.
- Efficient weight updates.
- Better utilization of computational resources.
![](../../images/gd_learning_rate.jpg)

### Methods of Selecting Learning Rate

**1\. Manual Tuning**

In this method, different learning rates are tested manually to identify which value gives the best training performance.

For example\:

- 0.1
- 0.01
- 0.001
- 0.0001

The learning rate that provides stable training and good convergence is selected.

**2\. Grid Search**

Grid Search tests a predefined set of learning rates systematically.

For example, the following values can be tested\:

$$
\{0.1,0.01,0.001,0.0001\}
$$

The performance of the model is evaluated for each value, and a suitable learning rate is selected.

**3\. Random Search**

Random Search selects learning rates randomly from a specified range.

It can explore a wider range of possible values without testing every predefined combination.

**4\. Learning Rate Finder**

A Learning Rate Finder gradually increases the learning rate during a short training run and observes how the loss changes.

The learning rate can then be selected from a region where the loss decreases effectively without becoming unstable.

**5\. Adaptive Optimizers**

Algorithms such as Adam, RMSprop, and AdaGrad automatically adjust effective learning rates for individual parameters using historical gradient information.

### Example Using Keras

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)
```

In this example, the learning rate is set to 0.01.

**Note\:** Learning rate is not the same as the number of epochs. Learning rate controls the size of parameter updates, whereas epochs determine how many times the model processes the entire training dataset.

---

## 4. Adaptive Optimizers

### Definition

Adaptive Optimizers are advanced optimization algorithms that automatically adjust the effective learning rate of individual parameters based on the gradients calculated during training.

In traditional Gradient Descent, a common learning rate is generally used for all parameters. However, different parameters may have gradients with different magnitudes and learning patterns.

Adaptive optimizers address this problem by using information from previous gradients to determine how much each parameter should be updated.

They make neural network training more efficient, especially when dealing with large datasets, complex architectures, and sparse gradients.

### Why Are Adaptive Optimizers Needed?

Consider a neural network containing thousands or millions of weights.

Some weights may receive large gradients, while others may receive very small gradients. Applying the same effective update size to every parameter may not always be efficient.

Adaptive optimizers solve this problem through the following approaches\:

1. They maintain information about previous gradients.
2. They calculate statistics from historical gradients.
3. They adjust the effective learning rate for individual parameters.
4. They help stabilize weight updates.
5. They can improve convergence speed.

There are three important adaptive optimization algorithms\:

### A. AdaGrad (Adaptive Gradient Algorithm)

**Definition\:**

AdaGrad is an adaptive optimization algorithm that automatically adjusts the learning rate for each parameter based on the accumulated squared gradients from previous iterations.

It gives smaller effective learning rates to parameters that have frequently received large gradients and relatively larger learning rates to parameters with smaller accumulated gradients.

It is particularly useful for sparse data, where some parameters are updated less frequently than others.

**Working Principle\:**

AdaGrad maintains a separate accumulated squared-gradient value for each parameter.

The process works as follows\:

1. Calculate the gradient of the loss function with respect to each parameter.
2. Square the calculated gradient.
3. Add the squared gradient to the accumulated squared gradients from previous iterations.
4. Divide the learning rate by the square root of the accumulated value.
5. Update the parameter using the adjusted learning rate.
6. Repeat the process during training.

**Mathematical Formulas**

First, calculate the accumulated squared gradients\:

$$
G_t=G_{t-1}+g_t^2
$$

Then update the weights\:

$$
\boxed{
W_{t+1}=W_t-
\frac{\eta}{\sqrt{G_t}+\epsilon}g_t
}
$$

Where\:

- $W_t$ \= Current weight.
- $W_{t+1}$ \= Updated weight.
- $g_t$ \= Current gradient.
- $G_t$ \= Accumulated squared gradients.
- $\eta$ \= Initial learning rate.
- $\epsilon$ \= Small constant to prevent division by zero.

**Example\:**

Suppose a parameter receives relatively large gradients repeatedly.

As the squared gradients accumulate, $G_t$ increases. Consequently, the effective learning rate decreases.

On the other hand, a parameter that receives relatively small or infrequent gradients has a smaller accumulated value and can retain a relatively larger effective learning rate.

**Advantages\:**

1. Automatically adjusts the learning rate for individual parameters.
2. Works effectively with sparse data.
3. Requires less manual adjustment of individual learning rates.
4. Can perform well in problems involving infrequent parameter updates.

**Disadvantages\:**

1. Accumulated squared gradients continuously increase.
2. The effective learning rate may become extremely small.
3. Training may stop making significant progress.
4. It may perform poorly during long training processes.

**Applications\:**

- Natural Language Processing.
- Sparse feature learning.
- Problems involving infrequent parameter updates.
- Text classification and related machine learning tasks.

### B. RMSprop (Root Mean Square Propagation)

**Definition\:**

RMSprop is an adaptive optimization algorithm that improves upon AdaGrad by using an exponentially weighted moving average of squared gradients instead of accumulating all squared gradients.

One major limitation of AdaGrad is that its accumulated squared gradients continuously increase, causing the effective learning rate to become excessively small.

RMSprop addresses this problem by giving more importance to recent gradients and gradually reducing the influence of older gradients.

**Working Principle\:**

RMSprop maintains a moving average of squared gradients.

The working process is\:

1. Calculate the current gradient.
2. Square the gradient.
3. Calculate the exponentially weighted moving average of squared gradients.
4. Divide the learning rate by the square root of this moving average.
5. Update the weights using the adjusted learning rate.
6. Repeat the process for subsequent iterations.

**Mathematical Formulas**

First, calculate the moving average of squared gradients\:

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

Then update the weights\:

$$
\boxed{
W_{t+1}=W_t-
\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
}
$$

Where\:

- $v_t$ \= Moving average of squared gradients.
- $v_{t-1}$ \= Previous moving average.
- $g_t$ \= Current gradient.
- $\beta$ \= Decay rate.
- $\eta$ \= Learning rate.
- $\epsilon$ \= Small constant for numerical stability.

A commonly used decay rate is approximately 0.9, although it can be adjusted.

**Understanding the Formula\:**

The moving average gives more importance to recent gradients while gradually reducing the contribution of older gradients.

For example, if the recent gradients become large, the moving average increases, reducing the effective learning rate.

If the recent gradients become smaller, the moving average adjusts accordingly.

This allows RMSprop to adapt its updates throughout training without continuously shrinking the learning rate as AdaGrad does.

**Advantages\:**

1. Automatically adjusts learning rates for individual parameters.
2. Prevents the learning rate from continuously decreasing due to an ever-growing accumulation.
3. Handles changing gradient magnitudes effectively.
4. Often performs well in recurrent neural networks and other deep learning applications.
5. Can provide faster and more stable convergence than basic Gradient Descent.

**Disadvantages\:**

1. Requires tuning of hyperparameters.
2. Adds memory requirements for storing moving averages.
3. Does not always provide the best generalization performance.
4. Its performance depends on the learning rate and decay rate.

**Applications\:**

- Recurrent Neural Networks (RNNs).
- Time-series prediction.
- Sequence modeling.
- Deep neural network training.

### C. Adam (Adaptive Moment Estimation)

(Widely used optimizer)

**Definition\:**

Adam is an advanced adaptive optimization algorithm that combines the advantages of Momentum and RMSprop.

It uses two moving averages of gradients to determine how the weights should be updated\:

1. First Moment: Exponentially weighted moving average of gradients.
2. Second Moment: Exponentially weighted moving average of squared gradients.

The first moment helps determine the direction of updates, while the second moment helps adjust the effective learning rate.

Adam is widely used because it generally provides fast and efficient training with relatively little manual tuning.

**Working Principle\:**

Adam maintains two separate moving averages.

**Step 1: Calculate the Gradient**

First, calculate the gradient of the loss function with respect to the weights.

$$
g_t=\nabla_W L(W_t)
$$

**Step 2: Calculate the First Moment**

The first moment estimates the average gradient.

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

This helps smooth the gradients and provides information about the general direction of weight updates.

**Step 3: Calculate the Second Moment**

The second moment estimates the average of squared gradients.

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

This helps determine the effective learning rate for each parameter.

**Step 4: Bias Correction**

At the beginning of training, both moving averages are initialized to zero. Therefore, their initial values may be biased toward zero.

Adam corrects this bias using\:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Where $t$ represents the current iteration.

**Step 5: Update Weights**

Finally, the weights are updated using\:

$$
\boxed{
W_{t+1}=W_t-
\frac{\eta\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}
$$

Where\:

- $m_t$ \= First moment estimate.
- $v_t$ \= Second moment estimate.
- $\hat m_t$ \= Bias-corrected first moment.
- $\hat v_t$ \= Bias-corrected second moment.
- $\beta_1$ \= First moment decay rate.
- $\beta_2$ \= Second moment decay rate.
- $\eta$ \= Learning rate.
- $\epsilon$ \= Small constant for numerical stability.

Common default values are\:

$$
\beta_1=0.9
$$

$$
\beta_2=0.999
$$

$$
\eta=0.001
$$

**Why Does Adam Combine Momentum and RMSprop?**

- Momentum uses previous gradients to smooth updates and maintain a consistent direction.
- RMSprop uses squared gradients to adapt the effective learning rate.
- Adam combines both mechanisms to achieve efficient parameter updates.

**Advantages\:**

1. Combines Momentum and adaptive learning rates.
2. Usually converges quickly.
3. Adjusts effective learning rates for individual parameters.
4. Works well with large and complex neural networks.
5. Requires relatively little manual learning rate tuning.
6. Handles noisy and sparse gradients effectively.

**Disadvantages\:**

1. Requires additional memory to store two moving averages for every parameter.
2. May sometimes generalize worse than SGD with Momentum.
3. Performance still depends on hyperparameter selection.
4. Fast convergence does not always guarantee the best final model performance.

**Applications\:**

- Image classification.
- Natural Language Processing.
- Deep Neural Networks.
- Transformer architectures.
- Computer Vision.
- Speech recognition.

### Comparison of Adaptive Optimizers

| Feature | AdaGrad | RMSprop | Adam |
|--------|--------|--------|--------|
| Main mechanism | Accumulated squared gradients | Moving average of squared gradients | First and second moments |
| Adaptive learning rate | Yes | Yes | Yes |
| Uses Momentum-like averaging | No | No | Yes |
| Learning rate behavior | Continuously decreases | Adapts to recent gradients | Adapts using two moments |
| Major strength | Sparse data | Changing gradients | Fast and flexible training |
| Major limitation | Learning rate may become too small | Hyperparameter sensitivity | Additional memory usage |

### Python Implementation Using Keras

```python
import tensorflow as tf

# AdaGrad
adagrad = tf.keras.optimizers.Adagrad(
    learning_rate=0.01
)

# RMSprop
rmsprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.001
)

# Adam
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=adam,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Here, Adam is selected as the optimizer. The other optimizers can be used by replacing `adam` with `adagrad` or `rmsprop`\.

---

## 5. Learning Rate Scheduling

### Definition

Learning Rate Scheduling is an optimization technique in which the learning rate is automatically adjusted during the training process according to a predefined strategy or the performance of the model.

In ordinary Gradient Descent, a fixed learning rate is used throughout training. However, using the same learning rate during all stages of training may not be efficient.

A larger learning rate can help the model learn quickly during the initial training stages, while a smaller learning rate can help it make more precise adjustments during later stages.

Learning Rate Scheduling solves this problem by gradually or conditionally changing the learning rate during training.

### Why Is Learning Rate Scheduling Needed?

Learning Rate Scheduling is important for the following reasons\:

1. **Faster Initial Learning\:** A relatively high learning rate helps the model make larger updates during early training.
2. **Fine-Tuning\:** A smaller learning rate allows the model to make more precise updates when approaching a minimum.
3. **Improved Convergence\:** It helps the optimizer approach a minimum more smoothly.
4. **Reduced Oscillations\:** Smaller learning rates can reduce unnecessary fluctuations near a minimum.
5. **Training Stability\:** Appropriate adjustments can prevent unstable updates.
6. **Better Performance\:** It can improve the final training and validation performance when properly configured.

### Types of Learning Rate Scheduling

#### A. Step Decay Scheduling

**Definition\:**

Step Decay is a learning rate scheduling technique in which the learning rate is reduced by a fixed factor after a specified number of epochs.

For example, suppose the initial learning rate is 0.1 and it is reduced by half every 10 epochs.

| Epoch | Learning Rate |
|--------|--------|
| 1–10 | 0.1 |
| 11–20 | 0.05 |
| 21–30 | 0.025 |
| 31–40 | 0.0125 |

**Mathematical Formula\:**

$$
\eta_t=\eta_0\gamma^{\lfloor t/s\rfloor}
$$

Where\:

- $\eta_t$ \= Learning rate at time $t$\.
- $\eta_0$ \= Initial learning rate.
- $\gamma$ \= Decay factor.
- $s$ \= Number of epochs between reductions.

**Advantages\:**

- Simple to understand and implement.
- Provides predictable learning rate changes.
- Helps improve fine-tuning during later training stages.

**Disadvantages\:**

- Learning rate changes suddenly rather than smoothly.
- Requires selecting the reduction interval and decay factor.

#### B. Exponential Decay Scheduling

**Definition\:**

Exponential Decay is a scheduling technique in which the learning rate decreases exponentially as the training process progresses.

Unlike Step Decay, which reduces the learning rate at specific intervals, Exponential Decay gradually reduces the learning rate over time.

**Mathematical Formula\:**

$$
\boxed{\eta_t=\eta_0e^{-kt}}
$$

Where\:

- $\eta_t$ \= Learning rate at time $t$\.
- $\eta_0$ \= Initial learning rate.
- $e$ \= Mathematical constant.
- $k$ \= Decay rate.
- $t$ \= Training step or time.

**Working Principle\:**

1. Start with an initial learning rate.
2. Calculate the new learning rate according to the decay formula.
3. Reduce the learning rate as training progresses.
4. Use the updated learning rate for subsequent weight updates.
5. Continue until training is completed.

**Advantages\:**

- Provides smooth learning rate reduction.
- Helps achieve more precise updates during later training.
- Easy to implement mathematically.

**Disadvantages\:**

- Requires choosing an appropriate decay rate.
- An excessively high decay rate can slow learning too early.
- The learning rate may become very small during long training.

#### C. Cosine Annealing

**Definition\:**

Cosine Annealing is a learning rate scheduling technique in which the learning rate decreases according to a cosine function over a specified training period.

It provides a smooth and gradual reduction in the learning rate rather than making sudden changes.

The learning rate starts at a maximum value and gradually decreases toward a minimum value following a cosine-shaped curve.

**Mathematical Formula\:**

$$
\eta_t=\eta_{min}+
\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi t}{T}\right)
$$

Where\:

- $\eta_t$ \= Learning rate at time $t$\.
- $\eta_{max}$ \= Maximum learning rate.
- $\eta_{min}$ \= Minimum learning rate.
- $t$ \= Current training step.
- $T$ \= Total number of steps in the schedule.

**Working Principle\:**

1. Start with a maximum learning rate.
2. Gradually decrease it following a cosine curve.
3. Make smaller updates as training progresses.
4. Reach the minimum learning rate at the end of the schedule.

**Advantages\:**

- Provides smooth learning rate adjustments.
- Helps avoid sudden learning rate changes.
- Can improve convergence and fine-tuning.
- Commonly used in modern deep learning training.

**Disadvantages\:**

- Requires selecting the schedule duration.
- May require additional configuration when used with restarts.

#### D. Reduce Learning Rate on Plateau

**Definition\:**

Reduce Learning Rate on Plateau is a scheduling technique in which the learning rate is automatically reduced when the model's performance stops improving for a specified number of epochs.

A plateau occurs when the monitored metric, such as validation loss, stops showing meaningful improvement.

Instead of reducing the learning rate after a fixed number of epochs, this technique makes adjustments based on actual model performance.

**Working Principle\:**

1. Start training with an initial learning rate.
2. Monitor a selected metric, usually validation loss.
3. Check whether the metric continues improving.
4. If there is no improvement for a specified number of epochs, reduce the learning rate.
5. Continue training using the smaller learning rate.

**Example\:**

Suppose\:

- Initial learning rate = 0.001
- Reduction factor = 0.5
- Patience = 5 epochs

If validation loss does not improve for five consecutive epochs, the learning rate becomes\:

$$
0.001\times0.5=0.0005
$$

If another plateau occurs, it may become\:

$$
0.0005\times0.5=0.00025
$$

**Advantages\:**

- Adjusts learning rates based on actual training performance.
- Does not require a fixed reduction schedule.
- Helps improve training when progress slows down.
- Useful when the best training duration is not known in advance.

**Disadvantages\:**

- Requires monitoring a suitable metric.
- May reduce the learning rate too frequently if patience is too small.
- Requires appropriate configuration of patience and reduction factor.

### Comparison of Learning Rate Scheduling Methods

| Method | How Learning Rate Changes |
|--------|--------|
| Step Decay | Decreases after fixed intervals |
| Exponential Decay | Decreases exponentially over time |
| Cosine Annealing | Decreases following a cosine curve |
| Reduce on Plateau | Decreases when performance stops improving |
![](../../images/lr_schedules.jpg)

### Python Implementation Using Keras

```python
import tensorflow as tf

scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[scheduler]
)
```

**Explanation of Parameters\:**

- `monitor="val_loss"`\: Monitors validation loss.
- `factor=0.5`\: Reduces the learning rate to half.
- `patience=5`\: Waits for five epochs without improvement before reducing the learning rate.
- `min_lr=0.00001`\: Sets the minimum allowed learning rate.
- `epochs=50`\: Trains the model for a maximum of 50 epochs.

---

## 6. Momentum

### Definition

Momentum is an optimization technique used with Gradient Descent to accelerate the training process and reduce unnecessary fluctuations during weight updates.

It improves ordinary Gradient Descent by considering both the current gradient and information from previous gradients when updating the weights and biases.

In ordinary Gradient Descent, the weight update depends only on the current gradient. However, Momentum maintains a velocity term that remembers previous updates and uses that information to determine the next update.

It helps the model move faster in directions where gradients consistently point in the same direction and reduces oscillations when gradients frequently change direction.

### Understanding Momentum Through an Example

Imagine a ball rolling down a hill.

- Initially, the ball starts moving slowly.
- As it continues moving in the same direction, it gains speed.
- Its previous movement contributes to its current movement.
- It continues moving faster until resistance or a change in direction affects its motion.

Momentum in neural networks follows a similar principle.

Instead of calculating each weight update independently, it uses information from previous updates to build a consistent direction of movement.

### Why Is Momentum Needed?

Ordinary Gradient Descent has some limitations, particularly when optimizing complex loss functions.

For example\:

1. It may oscillate when the loss function has a narrow valley.
2. It may move slowly through flat regions.
3. It may require many iterations to reach a minimum.
4. Its updates depend only on the current gradient.

Momentum addresses these problems by maintaining a velocity term.

### Working Principle of Momentum

The working process can be explained through the following steps\:

1. **Initialize Velocity\:** Initially, the velocity is generally set to zero.
2. **Calculate Gradient\:** Calculate the gradient of the loss function using backpropagation.
3. **Calculate New Velocity\:** Combine the previous velocity with the current gradient.
4. **Update Weights\:** Use the calculated velocity to update the weights.
5. **Repeat\:** Continue the process for subsequent training iterations.

### Mathematical Formulas

Momentum uses two main equations.

**Step 1: Velocity Update**

$$
v_t=\beta v_{t-1}+\eta g_t
$$

**Step 2: Weight Update**

$$
\boxed{W_{t+1}=W_t-v_t}
$$

Where\:

- $v_t$ \= Current velocity.
- $v_{t-1}$ \= Previous velocity.
- $\beta$ \= Momentum coefficient.
- $\eta$ \= Learning rate.
- $g_t$ \= Current gradient.
- $W_t$ \= Current weight.
- $W_{t+1}$ \= Updated weight.

The momentum coefficient controls how much information from previous updates is retained.

A commonly used value is\:

$$
\beta=0.9
$$

This means that previous velocity contributes significantly to the current update.

### How Momentum Improves Gradient Descent

Momentum provides two important improvements.

**1\. Accelerating Learning in Consistent Directions**

Suppose the gradients consistently point toward the same direction.

In ordinary Gradient Descent, each update depends only on the current gradient.

With Momentum, previous updates contribute to the current velocity. Therefore, the velocity increases in magnitude when updates consistently point in the same direction.

As a result, the network can move faster toward a minimum.

**2\. Reducing Oscillations**

Consider a loss function with a narrow valley.

The gradients may point alternately in different directions across the steep sides of the valley.

Ordinary Gradient Descent may repeatedly move from one side to the other, causing unnecessary oscillations.

Momentum smooths these updates by retaining information from previous directions.

As a result, it reduces excessive side-to-side movement and helps the optimizer move toward the minimum more efficiently.
![](../../images/momentum.jpg)
### Advantages of Momentum

1. **Faster Convergence\:** It accelerates learning when gradients consistently point in the same direction.
2. **Reduced Oscillations\:** It reduces unnecessary fluctuations during weight updates.
3. **Better Optimization\:** It helps the optimizer move through narrow valleys more efficiently.
4. **Improved Training Stability\:** It smooths parameter updates by using historical gradient information.
5. **Efficient Learning\:** It can reduce the number of iterations required to reach a good solution.

### Disadvantages of Momentum

1. **Additional Hyperparameter\:** It requires selecting an appropriate momentum coefficient.
2. **Overshooting\:** Excessively high momentum may cause the optimizer to move beyond the minimum.
3. **Parameter Tuning\:** The momentum coefficient and learning rate must be selected appropriately.
4. **Additional Computation\:** It requires maintaining and updating velocity values.

### Types of Momentum-Based Optimization

Two commonly discussed approaches are\:

**A. Classical Momentum**

Classical Momentum uses the previous velocity and current gradient to calculate the new velocity.

$$
v_t=\beta v_{t-1}+\eta g_t
$$

$$
W_{t+1}=W_t-v_t
$$

It helps accelerate Gradient Descent and reduce oscillations.

**B. Nesterov Accelerated Gradient (NAG)**

Nesterov Accelerated Gradient is an improved variation of Momentum.

Instead of calculating the gradient only at the current position, it first considers an approximate future position based on the current momentum and calculates the gradient at that position.

This allows the optimizer to anticipate the direction of movement and adjust its updates accordingly.

Its basic formulation is\:

$$
g_t=\nabla L(W_t-\beta v_{t-1})
$$

$$
v_t=\beta v_{t-1}+\eta g_t
$$

$$
W_{t+1}=W_t-v_t
$$

**Advantages of NAG\:**

- Provides a more informed update direction.
- Can reduce overshooting.
- Often improves convergence compared with classical Momentum.

### Python Implementation Using Keras

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
)
```

Here\:

- `learning_rate=0.01` determines the step size.
- `momentum=0.9` determines the contribution of previous velocity.
- `SGD` uses Stochastic Gradient Descent with Momentum.

To enable Nesterov Momentum\:

```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

---

## 7. Overall Comparison of Optimization Techniques

| Technique | Main Purpose | Key Benefit |
|--------|--------|--------|
| Gradient Descent | Updates weights using gradients | Minimizes loss |
| Learning Rate Optimization | Selects an appropriate learning rate | Controls update size |
| Adaptive Optimizers | Adjusts effective learning rates for individual parameters | Efficient parameter updates |
| Learning Rate Scheduling | Changes learning rate during training | Improves convergence |
| Momentum | Uses previous update information | Accelerates learning and reduces oscillations |

## 8. Relationship Between Optimization Techniques

Although these techniques are discussed separately, they are closely related and can be used together.

For example, consider training a neural network using Adam with a learning rate scheduler.

- **Adam** determines how the weights should be updated using first and second moment estimates.
- **Learning Rate** controls the overall scale of those updates.
- **Learning Rate Scheduling** changes the learning rate during training.
- **Backpropagation** calculates the gradients used by Adam.

Similarly, SGD can be combined with Momentum and Learning Rate Scheduling.

This combination is frequently used when training deep neural networks.

## 9. Conclusion

Optimization techniques are essential components of Deep Learning because they determine how a neural network learns by adjusting its weights and biases to minimize the loss function.

Gradient Descent provides the fundamental mechanism for updating parameters, while learning rate optimization controls the size of these updates. Adaptive optimizers such as AdaGrad, RMSprop, and Adam improve the optimization process by adjusting effective learning rates based on historical gradients.

Learning Rate Scheduling changes the learning rate throughout training to improve convergence, whereas Momentum accelerates learning and reduces oscillations by considering previous updates.

By selecting appropriate optimization techniques, we can improve training speed, stability, convergence, and the overall performance of deep neural networks.

---

## 10. Important Points to Remember 

| Concept | Remember |
|--------|--------|
| Gradient Descent | Moves weights in the opposite direction of the gradient |
| Batch GD | Uses the entire dataset for one update |
| Stochastic GD | Uses one sample for one update |
| Mini-Batch GD | Uses a small group of samples for one update |
| Learning Rate | Determines the size of weight updates |
| AdaGrad | Accumulates squared gradients |
| RMSprop | Uses an exponentially weighted average of squared gradients |
| Adam | Combines first and second moment estimates |
| Learning Rate Scheduling | Changes learning rate during training |
| Momentum | Uses previous updates to accelerate learning |
| Nesterov Momentum | Calculates gradients at an approximate look-ahead position |

**Important distinction\:** Adam already incorporates Momentum-like behavior through its first moment estimate. Therefore, you do not normally need to add classical SGD Momentum separately when using Adam.

