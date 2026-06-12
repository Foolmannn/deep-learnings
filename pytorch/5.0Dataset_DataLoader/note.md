In PyTorch, **Dataset** and **DataLoader** are the two core components used to efficiently load and process data during training.

Think of them as:

* **Dataset** → Defines **what data exists** and how to access a single sample.
* **DataLoader** → Defines **how data is delivered** to the model in batches.

---

# 1. Why Do We Need Dataset and DataLoader?

Suppose you have 50,000 images.

Without Dataset/DataLoader:

```python
for image in images:
    process(image)
```

Problems:

* No batching
* No shuffling
* Loads everything manually
* Slow for large datasets
* Difficult to use multiprocessing

PyTorch solves this using:

```python
Dataset -> DataLoader -> Model
```

---

# 2. Dataset

A Dataset is a class that represents your data.

PyTorch provides:

```python
torch.utils.data.Dataset
```

To create a custom dataset, implement:

1. `__len__()`
2. `__getitem__()`

---

## Dataset Structure

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self):
        # load data information
        pass

    def __len__(self):
        # return total number of samples
        pass

    def __getitem__(self, idx):
        # return one sample
        pass
```

---

## Example 1: Simple Dataset

```python
from torch.utils.data import Dataset

class NumberDataset(Dataset):

    def __init__(self):
        self.data = [1,2,3,4,5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

Usage:

```python
dataset = NumberDataset()

print(len(dataset))
```

Output:

```python
5
```

Access sample:

```python
print(dataset[0])
```

Output:

```python
1
```

---

# Understanding **getitem**()

This method tells PyTorch:

> "When I ask for sample number i, what should be returned?"

Example:

```python
dataset[2]
```

internally calls:

```python
dataset.__getitem__(2)
```

returns:

```python
3
```

---

# Example 2: Features and Labels

Suppose:

| Feature | Label |
| ------- | ----- |
| 1       | 0     |
| 2       | 0     |
| 3       | 1     |
| 4       | 1     |

Dataset:

```python
class CustomDataset(Dataset):

    def __init__(self):
        self.X = [1,2,3,4]
        self.y = [0,0,1,1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Usage:

```python
dataset = CustomDataset()

print(dataset[0])
```

Output:

```python
(1, 0)
```

---

# Real Dataset Example (CSV)

Suppose:

```text
age,height,label
20,170,0
25,180,1
30,175,0
```

Dataset:

```python
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

# 3. DataLoader

Dataset gives one sample at a time.

DataLoader gives:

* Batches
* Shuffling
* Parallel loading
* Automatic iteration

```python
from torch.utils.data import DataLoader
```

---

## Creating DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

Parameters:

| Parameter   | Meaning               |
| ----------- | --------------------- |
| dataset     | Dataset object        |
| batch_size  | Samples per batch     |
| shuffle     | Randomize order       |
| num_workers | Parallel loading      |
| drop_last   | Drop incomplete batch |
| pin_memory  | Faster GPU transfer   |

---

# Example

Dataset:

```python
dataset = NumberDataset()
```

DataLoader:

```python
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)
```

Iterate:

```python
for batch in loader:
    print(batch)
```

Output:

```python
tensor([1,2])
tensor([3,4])
tensor([5])
```

---

# Batch Size

Dataset:

```python
[1,2,3,4,5,6]
```

If:

```python
batch_size = 2
```

Batches become:

```python
[1,2]
[3,4]
[5,6]
```

---

# Shuffle

Without shuffle:

```python
1 2 3 4 5 6
```

Every epoch:

```python
1 2 3 4 5 6
```

With:

```python
shuffle=True
```

Epoch 1:

```python
4 2 1 6 3 5
```

Epoch 2:

```python
3 5 2 1 6 4
```

This prevents the model from learning patterns based on data order.

---

# drop_last

Suppose:

```python
10 samples
batch_size = 3
```

Batches:

```python
[1,2,3]
[4,5,6]
[7,8,9]
[10]
```

Last batch size is 1.

If:

```python
drop_last=True
```

Output:

```python
[1,2,3]
[4,5,6]
[7,8,9]
```

Last incomplete batch is discarded.

Useful for Batch Normalization.

---

# num_workers

Controls parallel data loading.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)
```

### num_workers=0

Main process loads data.

```text
Training
  |
Load Data
```

---

### num_workers=4

Four worker processes load data simultaneously.

```text
Worker1
Worker2
Worker3
Worker4
    |
 DataLoader
    |
 Training
```

Much faster for large datasets.

---

# pin_memory

Used when training on GPU.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True
)
```

Allows faster transfer:

```python
CPU -> GPU
```

---

# Complete Flow

Dataset:

```python
class MyDataset(Dataset):

    def __init__(self):
        self.X = torch.tensor([[1],[2],[3],[4]])
        self.y = torch.tensor([0,0,1,1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

DataLoader:

```python
dataset = MyDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

Training:

```python
for inputs, labels in loader:

    print(inputs)
    print(labels)
```

Possible output:

```python
tensor([[3],[1]])
tensor([1,0])

tensor([[4],[2]])
tensor([1,0])
```

---

# Training Loop Example

```python
for epoch in range(10):

    for X_batch, y_batch in loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()
```

Workflow:

```text
Dataset
   |
   V
DataLoader
   |
   V
Batch
   |
   V
Model
   |
   V
Loss
   |
   V
Backpropagation
```

---

# Built-in Datasets

PyTorch already provides many datasets through [torchvision datasets documentation](https://pytorch.org/vision/stable/datasets.html?utm_source=chatgpt.com).

Example:

```python
from torchvision import datasets

mnist = datasets.MNIST(
    root='./data',
    train=True,
    download=True
)
```

Then:

```python
loader = DataLoader(
    mnist,
    batch_size=64,
    shuffle=True
)
```

---

# Dataset vs DataLoader

| Feature                | Dataset | DataLoader |
| ---------------------- | ------- | ---------- |
| Stores data            | ✅       | ❌          |
| Returns single sample  | ✅       | ❌          |
| Returns batches        | ❌       | ✅          |
| Shuffling              | ❌       | ✅          |
| Parallel loading       | ❌       | ✅          |
| Iteration support      | Limited | Full       |
| Used by model directly | Rarely  | Usually    |

---

# Interview Questions

### Why implement `__len__()`?

To tell PyTorch how many samples exist.

```python
len(dataset)
```

---

### Why implement `__getitem__()`?

To tell PyTorch how to retrieve one sample.

```python
dataset[i]
```

---

### Why use DataLoader instead of Dataset directly?

Dataset gives:

```python
sample = dataset[5]
```

DataLoader provides:

* batching
* shuffling
* multiprocessing
* easy iteration

---

### What happens internally?

```text
DataLoader
    |
calls __getitem__(0)
calls __getitem__(1)
calls __getitem__(2)
...
    |
combines samples
    |
creates batch
    |
returns tensors
```

This separation of concerns is one of PyTorch's most important design patterns: **Dataset defines access to individual data samples, while DataLoader efficiently organizes those samples into batches for training and inference.**
