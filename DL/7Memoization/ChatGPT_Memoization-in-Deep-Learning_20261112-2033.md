# Memoization in Deep Learning (DL) – Detailed Study Notes

First, an important clarification: **Memoization is not a learning algorithm like Gradient Descent, Backpropagation, or MLP.** It is a programming and computational optimization technique that can be used when implementing deep learning models.

Memoization helps avoid performing the same computation repeatedly by storing previously calculated results and reusing them when needed.

Let's understand it from the fundamentals and then explore its applications in Deep Learning using Python and PyTorch.

# 1. What is Memoization?

Memoization is an optimization technique in which the result of an expensive function call is stored so that when the same function is called again with the same input, the previously calculated result is returned instead of performing the computation again.

In simple words\:

**Calculate once, store the result, and reuse it whenever possible.**

Consider a simple example.

Suppose we have a function that calculates the square of a number.

```python
def square(n):
    print("Calculating...")
    return n * n

print(square(5))
print(square(5))
print(square(5))
```

Output\:

```text
Calculating...
25
Calculating...
25
Calculating...
25
```

The function calculates the same result three times.

We can optimize this using memoization.

```python
cache = {}

def square(n):

    if n in cache:
        print("Fetching from cache")
        return cache[n]

    print("Calculating...")
    result = n * n

    cache[n] = result

    return result

print(square(5))
print(square(5))
print(square(5))
```

Output\:

```text
Calculating...
25
Fetching from cache
25
Fetching from cache
25
```

Notice that the calculation happens only once.

The subsequent function calls reuse the stored result.

## 1.1 How Memoization Works

## Memoization workflow

**Function receives input**

Example: square(5)

**Check cache**

Is the result already stored?

(YES)

**Cache hit**

Return stored result

(NO)

**Cache miss**

Perform computation

**Store the result in cache**

Reuse it for future identical inputs

There are two important terms\:

- **Cache Hit\:** The required result is already available in memory.
- **Cache Miss\:** The required result is not available, so the computation must be performed.

# 2. Why is Memoization Important in Deep Learning?

Deep learning models perform enormous numbers of mathematical operations.

For example, consider a neural network\:
![](mem0.jpg)
During training, a neural network repeatedly performs\:

1. Forward propagation.
2. Loss calculation.
3. Backpropagation.
4. Weight updates.

Some computations may be expensive, and in particular workflows, certain results may be reused.

Memoization can help reduce unnecessary repeated computations.

However, there is an important distinction\:

**Standard neural network training does not generally memoize forward-pass activations across different training batches or weight updates.**

Why?

Because the weights change during training. A result calculated using old weights may no longer be valid.

Memoization is useful when the function is deterministic and its inputs, including all relevant dependencies, have not changed.

# 3. Memoization vs Caching

These terms are closely related, but they are not exactly identical.

| Memoization | Caching |
|--------|--------|
| A specific optimization technique | A broader storage technique |
| Stores function results | Stores different kinds of data |
| Usually keyed by function inputs | Can use many storage strategies |
| Avoids repeated function execution | Reduces repeated computation or data access |

For example\:

- Memoization: Store the output of a mathematical function.
- Caching: Store frequently accessed datasets, images, model responses, or intermediate results.

Memoization is essentially a form of caching.

# 4. Applications of Memoization in Deep Learning

Let's explore where memoization and related caching techniques are useful in DL.

## 4.1 Reusing Expensive Mathematical Computations

Suppose a deep learning model repeatedly calculates an expensive mathematical function.

For example\:

$$
f(x)=x^2+2x+5
$$

If the same input occurs multiple times, we can store its result.

Python implementation\:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(x):

    print("Computing...")

    return x**2 + 2*x + 5

print(expensive_function(10))
print(expensive_function(10))
print(expensive_function(20))
```

Output\:

```text
Computing...
125
125
Computing...
445
```

Python's `lru_cache` automatically manages the cache.

Here\:

- `maxsize=128` means up to 128 recent results can be stored.
- Repeated inputs return cached results.
- When the cache is full, older entries can be removed according to the Least Recently Used (LRU) policy.

**Application in DL\:**

This can help when preprocessing functions or deterministic computations repeatedly receive identical inputs.

However, the function must account for all dependencies. For example, a function depending on model weights cannot safely cache only its input data if those weights change.

## 4.2 Memoization in Data Preprocessing

Data preprocessing is an important part of deep learning.

Consider an NLP project.

Suppose we have the following sentences\:

```python
sentences = [
    "Deep learning is interesting",
    "Machine learning is powerful",
    "Deep learning is interesting",
    "Deep learning is interesting"
]
```

We want to tokenize each sentence.

Without memoization, the same sentence gets tokenized repeatedly.

```python
def tokenize(sentence):

    print("Tokenizing...")

    return sentence.lower().split()

for sentence in sentences:
    print(tokenize(sentence))
```

The sentence `"Deep learning is interesting"` gets processed three times.

We can avoid that\:

```python
cache = {}

def tokenize(sentence):

    if sentence in cache:
        return cache[sentence]

    print("Tokenizing...")

    tokens = sentence.lower().split()

    cache[sentence] = tokens

    return tokens

for sentence in sentences:
    print(tokenize(sentence))
```

Now each unique sentence is tokenized only once.

**Applications\:**

- Text tokenization.
- Text normalization.
- Lemmatization.
- Feature extraction.
- Image resizing.
- Audio preprocessing.

This is particularly useful when processing datasets containing duplicate inputs.

## 4.3 Memoization in Feature Extraction

Feature extraction can be computationally expensive.

For example, consider extracting features from images using a pretrained CNN.
![](mem1.jpg)

Suppose we use a pretrained CNN to extract image features.

```python
features = {}

def extract_features(image_id, image):

    if image_id in features:
        return features[image_id]

    feature_vector = cnn_model(image)

    features[image_id] = feature_vector

    return feature_vector
```

If the same image is processed repeatedly with the same model and preprocessing configuration, its feature vector can be reused.

This can save computational time.

However, if the CNN's weights are being updated, the old feature vector becomes stale and must be recalculated.

## 4.4 Memoization in Recursive Neural Network Computations

Memoization is particularly useful in recursive computations.

Consider Fibonacci numbers\:

$$
F(n)=F(n-1)+F(n-2)
$$

With the initial conditions\:

$$
F(0)=0,\quad F(1)=1
$$

A normal recursive implementation\:

```python
def fibonacci(n):

    if n <= 1:
        return n

    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(6))
```

The same values are calculated multiple times.

For example, `fibonacci(3)` may be calculated repeatedly.

Using memoization\:

```python
def fibonacci(n, cache={}):

    if n in cache:
        return cache[n]

    if n <= 1:
        return n

    result = fibonacci(n-1, cache) + fibonacci(n-2, cache)

    cache[n] = result

    return result

print(fibonacci(6))
```
![](mem2.jpg)

This reduces the time complexity from\:

$$
O(2^n)
$$

to\:

$$
O(n)
$$

**Connection to Deep Learning\:**

Recursive neural network architectures and tree-structured models can involve repeated computations over identical substructures. Memoization can help when those substructures and model parameters remain unchanged.

It is not a universal optimization for recurrent neural networks (RNNs), however.

# 5. Memoization in PyTorch

Now let's explore how caching can be used in an actual deep learning framework.

PyTorch is widely used for implementing neural networks.

First, install PyTorch if you haven't already\:

```bash
pip install torch
```

## 5.1 Simple Memoization with PyTorch

Let's create a function that performs a mathematical operation on a tensor.

```python
import torch

cache = {}

def calculate_square(x):

    key = tuple(x.tolist())

    if key in cache:
        print("Using cached result")
        return cache[key]

    print("Calculating result")

    result = x ** 2

    cache[key] = result

    return result

x = torch.tensor([2.0, 3.0])

print(calculate_square(x))
print(calculate_square(x))
```

Output\:

```text
Calculating result
tensor([4., 9.])

Using cached result
tensor([4., 9.])
```

The second call reuses the previous result.

Note: This is a simple demonstration. In real PyTorch applications, tensor caching requires careful handling of device, dtype, mutation, and autograd dependencies. A cached tensor may also retain computational graphs and consume GPU memory.

## 5.2 Caching Feature Extraction in PyTorch

Let's consider a more realistic example.

We have a pretrained neural network that extracts features from images.

```python
import torch
import torchvision.models as models

model = models.resnet18(weights="DEFAULT")

model.eval()

feature_cache = {}

def extract_features(image_id, image):

    if image_id in feature_cache:
        return feature_cache[image_id]

    with torch.no_grad():

        features = model(image)

    feature_cache[image_id] = features

    return features
```

Here\:

- `model.eval()` puts the model into evaluation mode.
- `torch.no_grad()` prevents gradient calculation.
- `feature_cache` stores extracted features.
- Repeated requests for the same image reuse its stored features.

This example assumes the same model weights and preprocessing are used for every cached entry.

# 6. Memoization vs Gradient Descent

This is an important distinction because both are used to improve deep learning workflows, but they solve different problems.

| Memoization | Gradient Descent |
|--------|--------|
| Computational optimization | Optimization algorithm |
| Avoids repeated calculations | Minimizes model loss |
| Stores previous results | Updates model weights |
| Improves computational efficiency | Improves model performance |
| Does not train a model by itself | Used to train a model |

For example\:

Gradient Descent updates the weights\:

$$
W_{new}=W_{old}-\alpha\frac{\partial L}{\partial W}
$$

Memoization stores previously calculated results.

They are not alternatives to each other and can be used in the same overall application.

# 7. Memoization vs Dynamic Programming

You may also encounter memoization when studying algorithms and dynamic programming.

These concepts are closely connected.

Dynamic Programming (DP) solves problems by breaking them into smaller overlapping subproblems and storing their results.

There are two major approaches\:

**1\. Top-Down Approach (Memoization)**

- Starts with the original problem.
- Recursively solves smaller problems.
- Stores calculated results.
- Reuses results when needed.

---

**2\. Bottom-Up Approach (Tabulation)**

- Starts with the smallest subproblems.
- Solves them iteratively.
- Stores results in a table.
- Builds toward the final solution.

Example\:

| Memoization | Tabulation |
|--------|--------|
| Top-down | Bottom-up |
| Usually recursive | Usually iterative |
| Calculates when required | Calculates in an organized order |
| Uses a cache | Uses a table or array |

Both techniques avoid unnecessary repeated calculations.

# 8. Advantages of Memoization

1. **Improves computational efficiency\:** Avoids repeating expensive calculations.
2. **Reduces execution time\:** Previously calculated results are immediately available.
3. **Useful for large datasets\:** Repeated preprocessing operations can be avoided.
4. **Supports recursive algorithms\:** Particularly useful for overlapping subproblems.
5. **Reduces redundant processing\:** Useful in NLP, computer vision, and feature extraction workflows.

# 9. Disadvantages and Limitations

Memoization also has some limitations.

### 1. Memory consumption

Every cached result requires memory.

For large tensors, storing results can consume significant RAM or GPU memory.

### 2. Stale results

If the input dependencies change, the cached result may become incorrect.

For example\:

- Model weights change.
- Preprocessing configuration changes.
- Dataset content changes.

The cache must be invalidated or updated.

### 3. Not every computation is reusable

If each input is unique, memoization may provide little or no benefit.

### 4. Computational overhead

Checking the cache and managing stored results also require computation.

For very inexpensive operations, memoization can actually make the program slower.

### 5. Autograd complications

In PyTorch, cached tensors may retain references to computational graphs. Reusing them incorrectly during training can cause errors or incorrect gradient calculations.

# 10. Practical Comparison: Without Memoization vs With Memoization

Let's write a small experiment to understand the performance difference.

```python
import time
from functools import lru_cache

def normal_function(n):

    time.sleep(1)

    return n * n

@lru_cache(maxsize=None)
def memoized_function(n):

    time.sleep(1)

    return n * n

# Without memoization
start = time.time()

for i in range(3):
    print(normal_function(5))

print("Normal time:", time.time() - start)

# With memoization
start = time.time()

for i in range(3):
    print(memoized_function(5))

print("Memoized time:", time.time() - start)
```

Expected approximate result\:

| Method | Execution time |
|--------|--------|
| Without memoization | 3 seconds |
| With memoization | 1 second |

The first execution takes one second, but the next two executions reuse the stored result.

The exact time will vary depending on your computer.

---

# 11. Important Concepts to Remember

| Term | Meaning |
|--------|--------|
| Memoization | Storing function results for reuse |
| Cache | Storage for previously calculated or retrieved data |
| Cache Hit | Requested result is available |
| Cache Miss | Requested result is unavailable |
| LRU | Least Recently Used cache replacement policy |
| Cache Invalidation | Removing or updating outdated cached results |
| Dynamic Programming | Solving overlapping subproblems efficiently |
| Tabulation | Bottom-up dynamic programming |

# 12. When Should You Use Memoization in Deep Learning?

| DL Task | Potential usefulness |
|--------|--------|
| NLP tokenization | High, when inputs repeat |
| Image preprocessing | High, when images repeat |
| Feature extraction | High, with fixed model weights |
| Recursive computations | High, with overlapping subproblems |
| Model inference | Useful for repeated identical requests |
| Gradient Descent | Generally not directly applicable to weight updates |
| Backpropagation | Usually handled through computational graphs and framework optimizations rather than manual memoization |

# 13. What Should You Learn Next?

Since you are studying Deep Learning, I recommend understanding these related computational concepts\:

1. **Caching in Deep Learning** – Different caching techniques and how they are used in neural networks.
2. **Computational Graphs** – How frameworks such as PyTorch represent mathematical operations.
3. **Automatic Differentiation (Autograd)** – How gradients are calculated automatically.
4. **Gradient Checkpointing** – A technique that trades additional computation for lower memory usage.
5. **Batching and Vectorization** – How to perform multiple operations efficiently.

(Recommended next topic)

**Computational Graphs in Deep Learning**

This is especially important because it will help you understand how forward propagation, backpropagation, intermediate tensors, and gradient calculations are connected internally.

**Final takeaway\:** Memoization is a technique for saving computational results and reusing them instead of recalculating them. In Deep Learning, it is particularly useful for repeated preprocessing, feature extraction, inference, and recursive computations. It improves computational efficiency but does not directly improve the learning capability or accuracy of a neural network.
