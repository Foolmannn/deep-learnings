Tensors are the **core data structure in PyTorch**. Almost everything in PyTorch (inputs, weights, outputs, gradients) is represented as tensors.

You can think of a tensor as:

* **0D tensor** → scalar (single number)
* **1D tensor** → vector
* **2D tensor** → matrix
* **3D+ tensor** → higher dimensional arrays

Similar to NumPy arrays, but with:

1. **GPU support**
2. **Automatic differentiation (autograd)**
3. **Optimized deep learning operations**

---

# 1. Creating tensors

Import PyTorch:

```python
import torch
```

### Scalar tensor (0D)

```python
x = torch.tensor(5)

print(x)
print(x.ndim)
```

Output:

```python
tensor(5)
0
```

---

### Vector (1D)

```python
v = torch.tensor([1,2,3])

print(v)
print(v.ndim)
```

Output:

```python
tensor([1,2,3])
1
```

Shape:

```python
print(v.shape)
```

Output:

```python
torch.Size([3])
```

Means:

```python
3 elements
```

---

### Matrix (2D)

```python
m = torch.tensor([
    [1,2],
    [3,4]
])

print(m.shape)
```

Output:

```python
torch.Size([2,2])
```

Means:

```
2 rows
2 columns
```

---

### 3D tensor

Example:

```python
t = torch.tensor([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ]
])

print(t.shape)
```

Output:

```python
torch.Size([2,2,2])
```

Interpret:

```text
2 matrices
each matrix:
    2 rows
    2 columns
```

---

# 2. Tensor attributes

Example:

```python
x = torch.tensor([[1,2],[3,4]])
```

### Shape

```python
x.shape
```

Output:

```python
torch.Size([2,2])
```

---

### Dimension

```python
x.ndim
```

Output:

```python
2
```

---

### Datatype

```python
x.dtype
```

Output:

```python
torch.int64
```

---

### Device

```python
x.device
```

Output:

```python
cpu
```

or:

```python
cuda:0
```

---

# 3. Specifying datatype

```python
x = torch.tensor([1,2,3],
                 dtype=torch.float32)

print(x.dtype)
```

Output:

```python
float32
```

Common types:

```python
torch.float32
torch.float64
torch.int32
torch.int64
torch.bool
```

---

# 4. Creating special tensors

### Zeros

```python
torch.zeros((2,3))
```

Output:

```python
[[0,0,0],
 [0,0,0]]
```

---

### Ones

```python
torch.ones((2,2))
```

---

### Random

```python
torch.rand((2,3))
```

Random numbers:

```python
0 to 1
```

---

### Random integers

```python
torch.randint(
    low=0,
    high=10,
    size=(2,3)
)
```

---

### Identity matrix

```python
torch.eye(3)
```

Output:

```python
1 0 0
0 1 0
0 0 1
```

---

# 5. Tensor indexing

Like Python lists:

```python
x = torch.tensor([
    [1,2],
    [3,4]
])

print(x[0])
```

Output:

```python
[1,2]
```

Access element:

```python
x[1,0]
```

Output:

```python
3
```

---

Slice:

```python
x[:,1]
```

Output:

```python
[2,4]
```

Meaning:

```text
all rows
column 1
```

---

# 6. Tensor operations

Addition:

```python
a = torch.tensor([1,2])
b = torch.tensor([3,4])

print(a+b)
```

Output:

```python
[4,6]
```

---

Subtraction:

```python
a-b
```

---

Multiplication element-wise:

```python
a*b
```

Output:

```python
[3,8]
```

---

Division:

```python
a/b
```

---

# 7. Matrix multiplication

Very important in ML.

Use:

```python
torch.matmul()
```

or:

```python
@
```

Example:

```python
a = torch.tensor([
    [1,2],
    [3,4]
])

b = torch.tensor([
    [5,6],
    [7,8]
])

c = a @ b

print(c)
```

Output:

```python
[[19,22],
 [43,50]]
```

---

# 8. Reshaping tensors

Suppose:

```python
x = torch.arange(12)
```

Output:

```python
[0..11]
```

Shape:

```python
[12]
```

Convert:

```python
x.reshape(3,4)
```

Output:

```python
3 rows
4 cols
```

---

Another:

```python
x.view(2,6)
```

---

# 9. Squeezing and unsqueezing

Remove dimension:

```python
x = torch.rand(1,5)

x.squeeze()
```

Shape:

```python
[5]
```

---

Add dimension:

```python
x.unsqueeze(0)
```

Useful for batch dimension:

```python
[5]
→
[1,5]
```

---

# 10. Tensor → NumPy

Convert:

```python
tensor.numpy()
```

---

NumPy → Tensor

```python
torch.from_numpy(arr)
```

---

# 11. GPU tensors

Move tensor:

```python
device = "cuda"

x = x.to(device)
```

or:

```python
x.cuda()
```

Back:

```python
x.cpu()
```

---

# 12. Gradients (most important for deep learning)

PyTorch tracks operations for backpropagation.

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x**2

y.backward()

print(x.grad)
```

Output:

```python
4
```

Because:

[
y=x^2
]

Derivative:

[
\frac{dy}{dx}=2x
]

At:

[
x=2
]

Gradient:

[
4
]

This is **automatic differentiation**, which trains neural networks.

---

# 13. Broadcasting

PyTorch automatically expands dimensions.

Example:

```python
a = torch.tensor([
    [1,2],
    [3,4]
])

b = torch.tensor([10,20])

print(a+b)
```

Output:

```python
[
[11,22],
[13,24]
]
```

`b` becomes:

```text
[10,20]
[10,20]
```

internally.

---

# 14. Why tensors matter in ML

Training loop:

```python
X = torch.tensor(data)
y = model(X)
loss = loss_fn(y,target)

loss.backward()

optimizer.step()
```

Everything:

* Inputs → tensors
* Weights → tensors
* Predictions → tensors
* Loss → tensor
* Gradients → tensors

Deep learning = tensor operations + gradients + optimization.

---

For your stage in ML, focus deeply on:

1. `shape`
2. indexing
3. matrix multiplication
4. reshape/view
5. broadcasting
6. `requires_grad=True`
7. GPU usage

Those account for most beginner confusion in PyTorch.
