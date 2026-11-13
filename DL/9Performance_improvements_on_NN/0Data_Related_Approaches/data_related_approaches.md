# Data-Related Approaches to Improve Neural Network Performance

Data-related approaches are techniques that improve the **quality, quantity, relevance, and distribution of training data** before or during neural network training.

A neural network learns patterns from the examples it receives. Even a well-designed architecture with an advanced optimizer may perform poorly if the training data is incomplete, noisy, inconsistent, insufficient, or biased.

For example, imagine training a neural network to classify cats and dogs. If the dataset contains incorrect labels, very few dog images, or images with inconsistent sizes and lighting, the network may learn incorrect or incomplete patterns.

There are four major data-related approaches\:

1. **Data Preprocessing** – Cleaning and preparing data.
2. **Data Augmentation** – Creating modified training examples to increase data diversity.
3. **Feature Selection** – Choosing the most useful input features.
4. **Handling Imbalanced Data** – Addressing unequal representation of classes.

**Overview of the four approaches**

**1\. Data Preprocessing**

Makes raw data clean, consistent, and suitable for model training.

---

**2\. Data Augmentation**

Creates additional varied training examples from existing data.

---

**3\. Feature Selection**

Removes irrelevant or redundant input features.

---

**4\. Handling Imbalanced Data**

Prevents a model from favoring classes with many more examples.

---
## Data Preprocessing in Deep Learning

## 1. Introduction to Data Preprocessing

Data preprocessing is the process of preparing raw data before feeding it into a Neural Network.

In real-world applications, data is rarely available in a clean and suitable format. It may contain missing values, duplicate records, inconsistent formats, noise, different numerical scales, and irrelevant information.

Neural networks learn patterns directly from the data provided to them. Therefore, poor-quality input data can negatively affect the model's training process and prediction performance.

**Data preprocessing** involves transforming raw data into a clean, consistent, and suitable format so that a neural network can learn effectively.

### Why is Data Preprocessing important?

Consider building a neural network to predict house prices.

Suppose our dataset contains\:

| House | Area (sq. ft.) | Bedrooms | Age | Price (NPR) |
|--------|--------|--------|--------|--------|
| A | 1200 | 3 | 10 | 8,000,000 |
| B | 1800 | 4 | 5 | 12,000,000 |
| C | 900 | 2 | 15 | 5,500,000 |
| D | Missing | 3 | 8 | 9,000,000 |

There are some problems\:

- House D has a missing area value.
- Area and price have very different numerical scales.
- The data may contain outliers.
- Different features may need different preprocessing techniques.

If we directly feed this data into a neural network, it may struggle to learn effectively.

Data preprocessing helps solve these problems.

### Main objectives

1. Improve data quality.
2. Handle missing and inconsistent values.
3. Convert categorical data into numerical representations.
4. Bring numerical features to suitable scales.
5. Reduce noise and irrelevant information.
6. Make training more stable and efficient.

---

# 2. Major Steps in Data Preprocessing

Data preprocessing involves several steps depending on the type and quality of the dataset.

## Data preprocessing workflow

**Raw Data**

Collected from different sources

**1\. Data Cleaning**

Missing values, duplicates, errors, noise

**2\. Data Transformation**

Encoding, scaling, normalization

**3\. Feature Processing**

Selection, extraction, engineering

**4\. Dataset Splitting**

Training, validation, and testing

**Preprocessed Data**

Ready for Neural Network training

Let's understand each step.

# 3. Data Cleaning

Data cleaning is the process of identifying and correcting errors, inconsistencies, and unwanted information in a dataset.

It ensures that the neural network receives reliable input data.

Common data-cleaning operations include\:

- Handling missing values.
- Removing duplicate records.
- Handling outliers.
- Correcting inconsistent values.
- Handling noisy data.

## 3.1 Handling Missing Values

Missing values occur when some information is unavailable in a dataset.

For example\:

| Age | Salary | Experience |
|--------|--------|--------|
| 25 | 30000 | 2 |
| 30 | 45000 | 5 |
| 28 | NaN | 3 |
| NaN | 50000 | 6 |

Here, `NaN` represents a missing value.

Missing data can create problems because neural network operations generally require numerical inputs.

### Techniques to handle missing values

**A. Removing missing records**

If only a few records contain missing values, we may remove those records.

```python
import pandas as pd

df = pd.read_csv("data.csv")

df = df.dropna()
```

Advantages\:

- Simple to implement.
- Does not introduce estimated values.

Disadvantages\:

- Reduces dataset size.
- May remove valuable information.

**B. Mean Imputation**

Replace missing numerical values with the mean of the corresponding feature.

Formula\:

$$
Mean=\frac{\sum_{i=1}^{n}x_i}{n}
$$

Example\:

$$
Age=[20,25,30,35,NaN]
$$

Mean\:

$$
\frac{20+25+30+35}{4}=27.5
$$

The missing value is replaced with 27.5.

```python
df["Age"] = df["Age"].fillna(df["Age"].mean())
```

**C. Median Imputation**

Replace missing values with the median.

```python
df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)
```

Median is often preferred when the data contains outliers or is highly skewed.

**D. Mode Imputation**

For categorical features, missing values can be replaced with the most frequently occurring category.

```python
df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)
```

### Comparison

| Method | Suitable for |
|--------|--------|
| Removing records | Very few missing values |
| Mean | Numerical data with relatively symmetric distributions |
| Median | Numerical data with outliers or skewness |
| Mode | Categorical data |

**Important\:** Calculate imputation values using the training dataset only. Apply those same values to validation and test data to prevent data leakage.

---

## 3.2 Handling Duplicate Records

Duplicate records are repeated observations in a dataset.

Example\:

| Name | Age | Salary |
|--------|--------|--------|
| Ram | 25 | 30000 |
| Sita | 30 | 45000 |
| Ram | 25 | 30000 |

The first and third records are duplicates.

Duplicates may cause certain observations to receive more importance than intended during training.

Python implementation\:

```python
# Check duplicate records
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
```

**Benefits\:**

- Prevents unnecessary repetition.
- Reduces dataset size.
- Helps maintain data quality.

Note that repeated observations are not always duplicates. For example, two customers may legitimately have identical ages and salaries.

---

## 3.3 Handling Outliers

Outliers are observations that differ significantly from the majority of the data.

Example\:

Consider the following salary dataset\:

$$
[25000,30000,35000,40000,45000,500000]
$$

Here, 500000 is much larger than the other values.

It may be an error or a genuine high salary.

### Why are outliers problematic in Neural Networks?

Outliers can\:

1. Distort feature scaling.
2. Influence gradient calculations.
3. Affect model training.
4. Make the model focus excessively on unusual observations.

### Methods to detect outliers

**A. Interquartile Range (IQR)**

IQR is the difference between the third quartile and the first quartile.

$$
IQR=Q_3-Q_1
$$

Lower bound\:

$$
LB=Q_1-1.5(IQR)
$$

Upper bound\:

$$
UB=Q_3+1.5(IQR)
$$

Values outside these boundaries are potential outliers.

**B. Z-score**

Z-score measures how far an observation is from the mean in terms of standard deviations.

$$
Z=\frac{x-\mu}{\sigma}
$$

Where\:

- $x$ \= Observation.
- $\mu$ \= Mean.
- $\sigma$ \= Standard deviation.

A common rule is to investigate values where\:

$$
|Z|>3
$$

### Python example

```python
import numpy as np
from scipy import stats

z_scores = np.abs(stats.zscore(df["Salary"]))

outliers = df[z_scores > 3]

print(outliers)
```

### How to handle outliers?

- Remove them if they are confirmed data errors.
- Apply capping or clipping.
- Use transformations such as logarithms for highly skewed positive features.
- Keep them if they represent valid and important real-world observations.

Do not automatically remove all outliers. A rare but genuine observation may contain important information.

---

## 3.4 Handling Noisy Data

Noise refers to unwanted or random variations in data that do not represent meaningful patterns.

For example, temperature measurements may contain sensor errors.

Actual readings\:

$$
[25,26,25,27,80,26]
$$

The reading 80 may be caused by a faulty sensor.

Noise can make it difficult for neural networks to identify meaningful patterns.

### Techniques to reduce noise

1. Smoothing: Reduce sudden variations.
2. Filtering: Remove unwanted signals.
3. Outlier detection: Identify unusual observations.
4. Denoising: Recover useful information from corrupted data.

For image datasets, common denoising techniques include Gaussian and median filtering.

---
![](../../images/denoising0.jpg)
![](../../images/denoising1.jpg)
![](../../images/denoising2.jpg)
# 4. Data Transformation

After cleaning, data often needs to be transformed into a format that neural networks can process effectively.

Important transformation techniques include encoding categorical variables, normalization, and standardization.

## 4.1 Encoding Categorical Data

Neural networks operate on numerical values, so categorical data generally needs to be converted into numerical representations.

Consider\:

| Person | Gender | City |
|--------|--------|--------|
| Ram | Male | Kathmandu |
| Sita | Female | Pokhara |
| Hari | Male | Lalitpur |

Gender and City are categorical features.

We cannot directly feed ordinary text categories into a standard neural network layer.

### A. Label Encoding

Label encoding assigns a numerical integer to each category.

Example\:

| Gender | Encoded |
|--------|--------|
| Male | 0 |
| Female | 1 |

Python\:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(
    df["Gender"]
)
```

**Problem\:** Integer encoding can unintentionally imply an order between categories.

For example\:

| City | Encoded |
|--------|--------|
| Kathmandu | 0 |
| Pokhara | 1 |
| Lalitpur | 2 |

A neural network may interpret the numerical differences as meaningful, even though cities have no natural ranking.

Therefore, label encoding should be used carefully for nominal categories.

### B. One-Hot Encoding

One-hot encoding creates separate binary columns for each category.

Example\:

Original data\:

| City |
|--------|
| Kathmandu |
| Pokhara |
| Lalitpur |
| Kathmandu |

After one-hot encoding\:

| Kathmandu | Pokhara | Lalitpur |
|--------|--------|--------|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |

Python\:

```python
df = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)
```

**Advantages\:**

- Does not introduce artificial numerical ordering.
- Easy to implement.
- Suitable for nominal categorical variables.

**Disadvantages\:**

- Creates many additional columns when there are many categories.
- Can increase memory requirements.
- High-cardinality categorical features may require alternative encoding strategies.

### C. Ordinal Encoding

Ordinal encoding is useful when categories have a meaningful order.

Example\:

| Education | Encoded |
|--------|--------|
| High School | 1 |
| Bachelor's | 2 |
| Master's | 3 |
| PhD | 4 |

Python\:

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[[
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]]
)

df[["Education"]] = encoder.fit_transform(
    df[["Education"]]
)
```

### Comparison of encoding techniques

| Technique | Best suited for |
|--------|--------|
| Label Encoding | Target labels and suitable integer-coded categories |
| One-Hot Encoding | Nominal categories |
| Ordinal Encoding | Categories with meaningful order |

---

# 5. Feature Scaling

Feature scaling is one of the most important preprocessing steps in neural networks.

It transforms numerical features into comparable numerical ranges.

### Why is feature scaling necessary?

Consider a dataset with two features\:

| Person | Age | Annual Salary |
|--------|--------|--------|
| A | 20 | 300000 |
| B | 30 | 500000 |
| C | 40 | 700000 |

Age ranges from 20 to 40, while salary ranges from 300000 to 700000.

The difference in numerical scales is very large.

This can cause problems during optimization, particularly when using gradient-based algorithms.

Feature scaling helps prevent large-scale features from dominating optimization simply because of their numerical magnitude.

Two important techniques are\:

1. Normalization (Min-Max Scaling).
2. Standardization (Z-score Scaling).

## 5.1 Min-Max Normalization

Min-Max normalization transforms feature values into a specified range, commonly between 0 and 1.

**Mathematical formula\:**

$$
X_{norm}=
\frac{X-X_{min}}{X_{max}-X_{min}}
$$

Where\:

- $X$ \= Original value.
- $X_{min}$ \= Minimum feature value.
- $X_{max}$ \= Maximum feature value.

### Example

Suppose we have\:

$$
Age=[20,30,40,50,60]
$$

For the value 40\:

$$
X_{norm}=\frac{40-20}{60-20}
$$

$$
X_{norm}=\frac{20}{40}=0.5
$$

The normalized dataset becomes\:

| Original | Normalized |
|--------|--------|
| 20 | 0.00 |
| 30 | 0.25 |
| 40 | 0.50 |
| 50 | 0.75 |
| 60 | 1.00 |

**Visual representation**

Original range 20 – 60

Normalized range 0 – 1

### Python implementation

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

### Advantages

- Places features within a common range.
- Preserves the relative ordering of values.
- Useful for neural networks sensitive to input scale.
- Can work well with bounded activation functions.

### Disadvantages

- Sensitive to outliers.
- New observations outside the original range may produce values below 0 or above 1.
- Does not make data normally distributed.

### Applications

- Image pixel scaling.
- Neural networks using sigmoid activation.
- Datasets with known minimum and maximum ranges.

For example, image pixel values are commonly scaled from 0–255 to 0–1\:

$$
X_{scaled}=\frac{X}{255}
$$

## 5.2 Standardization (Z-score Normalization)

Standardization transforms data so that it has approximately\:

- Mean = 0
- Standard deviation = 1

**Mathematical formula\:**

$$
Z=\frac{X-\mu}{\sigma}
$$

Where\:

- $X$ \= Original value.
- $\mu$ \= Mean.
- $\sigma$ \= Standard deviation.

### Example

Suppose\:

$$
X=[10,20,30,40,50]
$$

Mean\:

$$
\mu=30
$$

Standard deviation (population)\:

$$
\sigma=\sqrt{
\frac{(10-30)^2+(20-30)^2+
(30-30)^2+(40-30)^2+(50-30)^2}{5}
}
$$

$$
\sigma=\sqrt{200}\approx14.14
$$

For the value 40\:

$$
Z=\frac{40-30}{14.14}
$$

$$
Z\approx0.707
$$

The complete standardized dataset is approximately\:

| Original | Standardized |
|--------|--------|
| 10 | \-1.414 |
| 20 | \-0.707 |
| 30 | 0 |
| 40 | 0.707 |
| 50 | 1.414 |

### Python implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### Advantages

- Centers data around zero.
- Makes feature scales comparable.
- Often useful for gradient-based optimization.
- Less sensitive to extreme values than Min-Max scaling in the sense that it does not force extreme values to define a fixed range.

### Disadvantages

- Does not remove outliers.
- Does not constrain values to a fixed range.
- Can still be strongly influenced by extreme values because mean and standard deviation are outlier-sensitive.

### Applications

- Deep neural networks.
- Regression problems.
- Datasets with differently scaled numerical features.
- Networks using ReLU and similar activations.

## 5.3 Normalization vs Standardization

This is an important distinction.

| Feature | Min-Max Normalization | Standardization |
|--------|--------|--------|
| Formula | $(X-X_{min})/(X_{max}-X_{min})$ | $(X-\mu)/\sigma$ |
| Output range | Usually 0 to 1 | No fixed range |
| Mean | Not necessarily 0 | 0 |
| Standard deviation | Not necessarily 1 | 1 |
| Outlier sensitivity | High | Can also be high |
| Common use | Image pixels, bounded features | General numerical features |

**Which one should we use in Neural Networks?**

There is no universally best choice.

- Use Min-Max scaling when a fixed range is desirable.
- Use Standardization as a common starting point for general numerical features.
- Use other robust scaling methods when extreme outliers are present.
- Choose based on the data distribution and network architecture.

---

# 6. Feature Engineering and Feature Selection

These two techniques are related but serve different purposes.

## 6.1 Feature Engineering

Feature engineering is the process of creating new meaningful features from existing data.

For example, suppose we have\:

| Date of Birth | Current Date |
|--------|--------|
| 2000-01-01 | 2026-01-01 |

Instead of giving the raw dates to a simple neural network, we can create an age feature.

Other examples\:

- Extracting month, day, and year from dates.
- Calculating total price from quantity and unit price.
- Extracting image edges or other useful descriptors.
- Creating interaction features between numerical variables.

Python example\:

```python
df["Total_Price"] = (
    df["Quantity"] * df["Unit_Price"]
)
```

Feature engineering can help the network receive more informative representations of the data.

## 6.2 Feature Selection

Feature selection is the process of identifying relevant features and removing unnecessary or irrelevant ones.

Suppose we are predicting house prices using\:

- House area.
- Number of bedrooms.
- Location.
- Building age.
- Random customer ID.

Customer ID might not contain meaningful information about house prices.

Keeping irrelevant features may increase model complexity and encourage overfitting.

Common feature selection methods\:

1. Correlation analysis.
2. Statistical tests.
3. Mutual information.
4. Recursive Feature Elimination (RFE).
5. Domain knowledge.

Python example\:

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

**Important\:** Feature selection must be performed using training data only, or within a proper cross-validation pipeline.

---

# 7. Dataset Splitting

After preprocessing, we divide the dataset into separate subsets.

This is important because we need to evaluate whether our model can generalize to unseen data.

The three main subsets are\:

### A. Training Dataset

Used to train the neural network.

The network learns weights and biases from this dataset.

### B. Validation Dataset

Used to evaluate the model during training.

It helps with\:

- Hyperparameter tuning.
- Selecting the best model.
- Early stopping.
- Comparing different architectures.

### C. Testing Dataset

Used for final evaluation after the model and its settings have been selected.

It estimates how well the model performs on unseen data.

### Common splitting ratios

| Dataset | Training | Validation | Testing |
|--------|--------|--------|--------|
| Dataset 1 | 70% | 15% | 15% |
| Dataset 2 | 80% | 10% | 10% |
| Dataset 3 | 60% | 20% | 20% |

These are common examples, not strict rules.

### Python implementation

```python
from sklearn.model_selection import train_test_split

# First separate the test set
X_train_val, X_test, y_train_val, y_test = (
    train_test_split(
        X, y,
        test_size=0.15,
        random_state=42
    )
)

# Split remaining data into train and validation
X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=42
    )
)
```

This gives approximately\:

- 70% training.
- 15% validation.
- 15% testing.

**Dataset distribution**

**70%**

**15%**

**15%**

Training

Validation

Testing

### Important: Data Leakage

Data leakage occurs when information from validation or testing data unintentionally influences the training process.

For example, suppose we standardize the entire dataset before splitting it.

Then the mean and standard deviation calculated from the test data have already influenced the training transformation.

This can make the final evaluation overly optimistic.

**Correct procedure\:**

1. Split the dataset.
2. Fit preprocessing transformations using training data only.
3. Transform training data.
4. Apply the same transformations to validation and testing data.
5. Train and evaluate the model.

---

# 8. Complete Data Preprocessing Pipeline in Python

Let's combine the important preprocessing steps into one practical example.

Suppose we have a dataset for predicting whether a customer will purchase a product.

Features include\:

- Age (numerical).
- Salary (numerical).
- City (categorical).
- Purchased (target).

### Step 1: Import libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

### Step 2: Load the dataset

```python
df = pd.read_csv("customers.csv")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]
```

### Step 3: Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps preserve the target class proportions in both subsets.

### Step 4: Define numerical and categorical features

```python
numerical_features = ["Age", "Salary"]

categorical_features = ["City"]
```

### Step 5: Create preprocessing pipelines

```python
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

### Step 6: Combine preprocessing

```python
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

### Step 7: Fit and transform

```python
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)
```

Notice the difference\:

- `fit_transform()` learns preprocessing parameters from training data and transforms it.
- `transform()` applies those learned parameters to test data.

This is an important practice for avoiding data leakage.

The resulting dataset is now suitable for feeding into a neural network.

---

# 9. Important Considerations When Preprocessing Data for Neural Networks

Some additional considerations are particularly important in Deep Learning.

| Data type | Common preprocessing |
|--------|--------|
| Tabular data | Imputation, encoding, scaling |
| Image data | Resizing, pixel scaling, augmentation |
| Text data | Tokenization, vocabulary mapping, padding |
| Audio data | Resampling, noise reduction, feature extraction |
| Time-series data | Missing-value handling, scaling, windowing |

A few important points\:

1. Neural networks generally benefit from appropriately scaled numerical inputs.
2. Different data types require different preprocessing techniques.
3. Avoid applying transformations that destroy meaningful information.
4. Do not use test data to learn preprocessing parameters.
5. For time-series forecasting, split data chronologically rather than randomly.
6. For image classification, preserve the correct label when applying augmentation.

---

# 10. Summary: Data Preprocessing Techniques

| Category | Technique | Purpose |
|--------|--------|--------|
| Data Cleaning | Missing value handling | Handle incomplete records |
| Data Cleaning | Duplicate removal | Remove unintended repeated records |
| Data Cleaning | Outlier handling | Manage unusual observations |
| Data Cleaning | Noise reduction | Remove unwanted variations |
| Transformation | Label Encoding | Convert categories into integers |
| Transformation | One-Hot Encoding | Convert nominal categories into binary columns |
| Transformation | Normalization | Scale values into a specified range |
| Transformation | Standardization | Center and scale numerical features |
| Feature Processing | Feature Engineering | Create meaningful new features |
| Feature Processing | Feature Selection | Remove irrelevant features |
| Dataset Preparation | Dataset Splitting | Separate training, validation, and testing data |

## 11. Key Concepts to Remember for Exams

- **Data preprocessing\:** Preparing raw data for machine learning and deep learning models.
- **Data cleaning\:** Identifying and correcting incomplete, duplicate, incorrect, or noisy data.
- **Encoding\:** Converting categorical variables into numerical representations.
- **Normalization\:** Scaling values to a specified range, commonly 0–1.
- **Standardization\:** Transforming data to have mean 0 and standard deviation 1.
- **Feature engineering\:** Creating meaningful features from existing data.
- **Feature selection\:** Selecting relevant features and removing unnecessary ones.
- **Data leakage\:** Unintentionally allowing information from validation or test data to influence model training.

## 2. Data Augmentation

### 2.1 What is Data Augmentation?

Data augmentation is a technique used to artificially increase the size and diversity of a training dataset by creating modified versions of existing samples.

Instead of collecting thousands of new examples, we apply transformations to existing examples to produce additional training samples.

**Example: Image Classification**

Suppose we have only 1,000 images of cats.

We can generate modified versions by\:

- Rotating images.
- Flipping images.
- Cropping images.
- Changing brightness.
- Adjusting contrast.
- Zooming in or out.
![](../../images/data_augumentation0.jpg)
![](../../images/data_augumentation1.jpg)
![](../../images/data_augumentation2.jpg)
The network sees different versions of the same underlying object and learns to recognize it despite changes in orientation, position, and lighting.

### 2.2 Why Do We Need Data Augmentation?

Suppose a neural network is trained using only front-facing images of cats.

When it receives a side-facing cat image, it may fail because it has not learned that the same cat can appear in different orientations.

Data augmentation exposes the model to more variations.

It helps to\:

1. Increase training data diversity.
2. Reduce overfitting.
3. Improve generalization.
4. Make the model robust to variations in real-world inputs.
5. Reduce the need for manually collecting new data.

### 2.3 Types of Data Augmentation

#### A. Image Data Augmentation
| Technique | Explanation |
|--------|--------|
| Rotation | Rotates an image by a certain angle |
| Flipping | Mirrors an image horizontally or vertically |
| Cropping | Selects a smaller region of an image |
| Zooming | Enlarges or reduces image content |
| Translation | Moves the image horizontally or vertically |
| Brightness adjustment | Changes image brightness |
| Contrast adjustment | Changes contrast |
| Noise injection | Adds small random variations to pixels |
![](../../images/image_data_augumentation0.jpg)
![](../../images/image_data_augumentation1.jpg)
![](../../images/image_data_augumentation2.jpg)
Note: Transformations must preserve the correct label. For example, flipping a handwritten digit 6 may turn it into something resembling 9, so not every transformation is suitable for every dataset.

#### B. Text Data Augmentation

Text augmentation creates alternative versions of existing sentences while attempting to preserve their meaning.

Original\:

> The movie was very interesting.

Possible augmented examples\:

- The film was very interesting.
- The movie was extremely interesting.
- The movie was really enjoyable.

Techniques include\:

- Synonym replacement.
- Random word deletion.
- Random word insertion.
- Random word swapping.
- Back translation (translating into another language and back).
- Paraphrasing using language models.

Text augmentation must preserve meaning and labels. Replacing a sentiment-bearing word such as "good" with "bad" would change the label.

#### C. Audio Data Augmentation

Audio augmentation generates variations in sound recordings.

Common methods\:

- Adding background noise.
- Changing pitch.
- Changing playback speed.
- Time shifting.
- Time stretching.
- Changing volume.

For example, a speech-recognition model can be trained with noisy and clean versions of the same speech to become more robust in noisy environments.

#### D. Time-Series Data Augmentation

Used in sensor readings, stock data, weather data, and activity recognition.

Common techniques\:

- Adding small random noise.
- Time shifting.
- Scaling.
- Window slicing.
- Time warping.
- Segment permutation (only when the task allows it).

For time-series forecasting, augmentation must respect temporal order and avoid using future information to construct past training samples.

### 2.4 Data Augmentation Using Python

Example using TensorFlow/Keras\:

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2)
])

# Example input batch
augmented_images = data_augmentation(images)
```

These transformations are typically applied during training. At inference time, the model receives the original image.

### 2.5 Advanced Data Augmentation Techniques

**Mixup**

Mixup combines two training examples and their labels.

$$
x'=\lambda x_i+(1-\lambda)x_j
$$

$$
y'=\lambda y_i+(1-\lambda)y_j
$$

Where $\lambda$ is a value between 0 and 1.

For example, if $\lambda=0.7$, the new input contains 70% of one image and 30% of another, with labels mixed in the same proportion.

**CutMix**

CutMix cuts a region from one image and places it into another image. The label is adjusted according to the proportion of each image remaining.
![](../../images/cutmix.jpg)

### 2.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|--------|--------|
| Reduces overfitting | Poor transformations can damage labels |
| Improves generalization | Adds training computation |
| Increases data diversity | Requires domain knowledge |
| Helps when data is limited | May introduce unrealistic samples |

---

## 3. Feature Selection

### 3.1 What is Feature Selection?

Feature selection is the process of identifying and selecting the most relevant input features from a dataset while removing irrelevant, redundant, or less useful features.

A *feature* is an individual input variable used by a machine learning model to make predictions.

For example, consider a neural network that predicts house prices.

| Feature | Example value |
|--------|--------|
| House area | 1500 sq. ft. |
| Number of bedrooms | 3 |
| Number of bathrooms | 2 |
| Location | Kathmandu |
| House color | Blue |
| House price | Rs. 15,000,000 |

Some features, such as house area and location, may be useful for predicting price. House color may have little predictive value.

Feature selection attempts to identify the useful features and remove unnecessary ones.
![](../../images/featureselection.jpg)

### 3.2 Why is Feature Selection Important?

Imagine a dataset containing 100 input features, but only 15 are useful for predicting the output.

If we feed all 100 features into the neural network, it may\:

- Waste computational resources processing irrelevant information.
- Learn patterns from noise.
- Become more prone to overfitting.
- Require more training time.
- Have difficulty identifying important relationships.

By selecting useful features, we can simplify the input and potentially improve model performance.

### 3.3 Types of Feature Selection Methods

There are three major approaches\:

#### A. Filter Methods

Filter methods select features using statistical measurements, without training a particular machine learning model.

They evaluate the relationship between each feature and the target variable.

Common techniques\:

**1\. Correlation**

Correlation measures the strength and direction of the relationship between two numerical variables.

$$
r=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

The correlation coefficient ranges from -1 to +1.

- \+1: Perfect positive correlation.
- \-1: Perfect negative correlation.
- 0: No linear correlation.

For example, house area may have a strong positive correlation with house price.

However, correlation does not necessarily imply causation, and a low linear correlation does not mean a feature is useless.

**2\. Chi-Square Test**

Used to examine the relationship between categorical features and categorical target variables.

**3\. ANOVA (Analysis of Variance)**

Can be used to measure whether numerical features differ significantly across target classes.

**4\. Mutual Information**

Measures how much information one variable provides about another.

It can detect both linear and nonlinear relationships.

#### B. Wrapper Methods

Wrapper methods evaluate different combinations of features by repeatedly training and evaluating a model.

They use model performance to determine which features are useful.

Common techniques\:

- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)

**Forward Selection\:**

1. Start with no selected features.
2. Add one feature at a time.
3. Train and evaluate the model.
4. Keep the feature that improves performance the most.
5. Repeat until adding more features no longer provides sufficient improvement.

**Backward Elimination\:**

1. Start with all features.
2. Remove one feature at a time.
3. Evaluate model performance.
4. Permanently remove features whose removal does not hurt performance.
5. Continue until an appropriate feature subset remains.

**Recursive Feature Elimination (RFE)\:**

1. Train a model using all available features.
2. Rank features by importance.
3. Remove the least important feature or group of features.
4. Retrain the model.
5. Repeat until the desired number of features remains.
![](../../images/recursivefeatureelimination.jpg)

#### C. Embedded Methods

Embedded methods perform feature selection as part of the model training process.

Common examples\:

- L1 Regularization (Lasso)
- Decision tree feature importance
- Random Forest feature importance

**L1 Regularization and Feature Selection**

Recall the L1 regularization technique\:

$$
L_{total}=L_{original}+\lambda\sum_{i=1}^{n}|w_i|
$$

L1 regularization encourages some weights to become exactly zero.

If a feature's corresponding weight becomes zero, that feature contributes nothing through that connection.

This is one reason L1 regularization can be used for feature selection.

Note: Neural networks can have multiple connections and nonlinear interactions, so a zero or small individual weight is not always equivalent to proving that an entire input feature is irrelevant.

### 3.4 Python Example: Feature Selection Using Correlation

```python
import pandas as pd

df = pd.read_csv("house_prices.csv")

# Calculate correlation with target
correlation = df.corr(numeric_only=True)

# Display correlation with house price
print(
    correlation["price"]
    .sort_values(ascending=False)
)
```

We can also visualize the correlations\:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

### 3.5 Python Example: Feature Selection Using Mutual Information

```python
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=5
)

X_selected = selector.fit_transform(X_train, y_train)

selected_features = X.columns[selector.get_support()]

print(selected_features)
```

This example selects the five features with the highest mutual-information scores for a classification problem.

For regression, use `mutual_info_regression` instead.

**Important\:** Feature selection must be performed using training data only. If it uses the entire dataset, information from the test set can leak into the training process.

### 3.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|--------|--------|
| Reduces input dimensionality | Useful features may accidentally be removed |
| Can reduce overfitting | Some methods are computationally expensive |
| Reduces training time | Results depend on the selection method |
| Removes irrelevant information | May miss interactions between features |
| Can improve interpretability | Selected features may vary across datasets |

### 3.7 Feature Selection vs Feature Extraction

These two terms are related but different.

| Feature Selection | Feature Extraction |
|--------|--------|
| Selects existing features | Creates new features |
| Original features are retained | Original features are transformed |
| Removes irrelevant features | Combines or transforms information |
| Example: Select 10 out of 50 features | Example: PCA transforms 50 features into 10 components |

For example, Principal Component Analysis (PCA) is a feature extraction technique, not a feature selection technique.

---

## 4. Handling Imbalanced Data

### 4.1 What is Imbalanced Data?

A dataset is imbalanced when the number of observations belonging to different classes is significantly unequal.

It is particularly important in classification problems.

Consider a dataset for detecting fraudulent transactions.

| Class | Number of samples | Percentage |
|--------|--------|--------|
| Normal transactions | 9,900 | 99% |
| Fraudulent transactions | 100 | 1% |
| **Total** | **10,000** | **100%** |

Here, normal transactions are the majority class, while fraudulent transactions are the minority class.
![](../../images/imbalanceddata.jpg)

### 4.2 Why is Imbalanced Data a Problem?

Suppose a neural network is trained to identify fraudulent transactions.

It predicts every transaction as normal.

Its accuracy becomes\:

$$
Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}
$$

$$
Accuracy=\frac{9900}{10000}=99\%
$$

The accuracy looks excellent, but the model has failed to identify even one fraudulent transaction.

This demonstrates why accuracy alone can be misleading when working with imbalanced datasets.

The model may become biased toward the majority class because it has many more examples of that class.

### 4.3 Techniques for Handling Imbalanced Data

There are several techniques to address this problem.

#### A. Random Oversampling

Oversampling increases the number of minority-class samples.

It works by randomly selecting existing minority-class examples and duplicating them.

Example\:

| Class | Before | After |
|--------|--------|--------|
| Normal | 900 | 900 |
| Fraud | 100 | 900 |
![](../../images/under&oversampling.jpg)

Python\:

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(
    X_train,
    y_train
)
```

**Advantages\:**

- Simple to implement.
- Retains all original minority samples.
- Increases the representation of minority classes.

**Disadvantages\:**

- Duplicated samples may cause overfitting.
- Increases training dataset size.
- Does not introduce genuinely new information.

#### B. Random Undersampling

Undersampling reduces the number of majority-class samples.

It randomly removes some observations from the majority class.

Example\:

| Class | Before | After |
|--------|--------|--------|
| Normal | 900 | 100 |
| Fraud | 100 | 100 |
![](../../images/undersampling.jpg)

Python\:

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(
    X_train,
    y_train
)
```

**Advantages\:**

- Reduces dataset size.
- Reduces training time.
- Balances class representation.

**Disadvantages\:**

- Removes potentially useful information.
- May reduce the model's ability to learn the majority class.
- Random removal may discard important examples.

#### C. SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is a popular technique that generates synthetic minority-class examples instead of simply duplicating existing examples.

It works by finding neighboring minority-class samples and generating new samples between them.

The basic interpolation formula is\:

$$
X_{new}=X_i+\lambda(X_{neighbor}-X_i)
$$

Where\:

- $X_i$ \= An existing minority-class sample.
- $X_{neighbor}$ \= One of its minority-class neighbors.
- $\lambda$ \= Random number between 0 and 1.

**Example\:**

Suppose two minority-class samples are\:

$$
X_1=(2,4)
$$

$$
X_2=(6,8)
$$

If $\lambda=0.5$\:

$$
X_{new}=(2,4)+0.5((6,8)-(2,4))
$$

$$
X_{new}=(4,6)
$$

A new synthetic minority sample is generated at the midpoint.

![](../../images/smote.jpg)

Python\:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

**Advantages\:**

- Generates new synthetic examples.
- Avoids simply duplicating all minority examples.
- Can improve minority-class learning.

**Disadvantages\:**

- Synthetic examples may not represent realistic data.
- Can generate overlapping or noisy samples.
- May perform poorly with high-dimensional or complex data.

#### D. Class Weighting

Instead of changing the dataset, we can change how much importance the neural network gives to each class during training.

The model assigns higher penalties to mistakes involving the minority class.

For example\:

- Normal transaction: weight = 1
- Fraudulent transaction: weight = 10

The loss function becomes\:

$$
L=-\sum_{i=1}^{N}w_{y_i}\log(p_{i,y_i})
$$

Where\:

- $w_{y_i}$ \= Weight assigned to the true class.
- $p_{i,y_i}$ \= Predicted probability of the correct class.

A higher class weight makes mistakes involving that class more costly.

Python with Keras\:

```python
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    class_weight={
        0: 1.0,
        1: 10.0
    }
)
```

The weights above are illustrative; they should be chosen based on the dataset and the cost of different errors.

**Advantages\:**

- Does not require changing dataset size.
- Gives more importance to minority classes.
- Easy to implement in many neural network frameworks.

**Disadvantages\:**

- Incorrect weights may lead to too many false positives.
- Requires careful evaluation and tuning.
- Does not create new minority-class information.

### 4.4 Evaluation Metrics for Imbalanced Data

Accuracy is not sufficient for evaluating imbalanced classification problems.

We should understand the confusion matrix and other metrics.

**Confusion Matrix**

Example for fraud detection

# TP

True Positive

Fraud correctly detected

# FP

False Positive

Normal marked as fraud

# FN

False Negative

Fraud missed by model

# TN

True Negative

Normal correctly identified

**1\. Precision**

Precision tells us how many predicted positive samples are actually positive.

$$
Precision=\frac{TP}{TP+FP}
$$

**2\. Recall**

Recall tells us how many actual positive samples were correctly identified.

$$
Recall=\frac{TP}{TP+FN}
$$

**3\. F1-Score**

F1-score is the harmonic mean of precision and recall.

$$
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

**4\. ROC-AUC and PR-AUC**

- ROC-AUC measures the model's ability to distinguish between classes across different classification thresholds.
- PR-AUC summarizes the precision-recall trade-off and is often especially informative when the positive class is rare.

For highly imbalanced datasets, PR-AUC is often more useful than ROC-AUC for assessing minority-class detection.

### 4.5 Important Rules When Handling Imbalanced Data

1. Split the dataset into training, validation, and test sets before applying oversampling or undersampling.
2. Apply resampling only to the training data.
3. Keep validation and test data in their original class distributions.
4. Use stratified splitting where appropriate.
5. Evaluate using precision, recall, F1-score, and PR-AUC rather than accuracy alone.

---

---

## 5. Comparison of All Data-Related Approaches

| Approach | Main purpose | Example techniques |
|--------|--------|--------|
| Data Preprocessing | Improve data quality | Cleaning, encoding, scaling, imputation |
| Data Augmentation | Increase data diversity | Rotation, flipping, Mixup, noise injection |
| Feature Selection | Keep useful input features | Correlation, RFE, L1 regularization |
| Handling Imbalanced Data | Improve minority-class learning | Oversampling, undersampling, SMOTE, class weighting |

## 6. How These Approaches Work Together

Consider building a neural network to predict whether a bank transaction is fraudulent.

We can apply all four approaches in the following order\:

## Fraud Detection: Data Preparation Pipeline

1. **Raw Dataset**
   
   Transaction records containing missing values, duplicates, and numerical and categorical features.
2. **Data Preprocessing**
   
   Remove duplicate records, handle missing values, encode categorical features, and scale numerical features.
3. **Feature Selection**
   
   Select relevant features such as transaction amount, transaction time, location, and account activity.
4. **Handle Imbalanced Data**
   
   Use class weighting or SMOTE to improve the representation of fraudulent transactions in training.
5. **Data Augmentation (if appropriate)**
   
   Generate valid variations of training examples when the data type and domain allow it.
6. **Neural Network Training**
   
   Train the network and monitor validation performance.
7. **Model Evaluation**
   
   Evaluate using precision, recall, F1-score, and PR-AUC on untouched test data.

The order of some operations can vary depending on the dataset and technique. For example, feature selection and resampling should be fitted within the training process to avoid data leakage.

---

## 7. Advantages of Data-Related Approaches

These approaches can improve neural network performance in several ways\:

1. **Better data quality\:** Removes errors, inconsistencies, and unnecessary noise.
2. **Improved generalization\:** Helps the network perform better on unseen data.
3. **Reduced overfitting\:** Augmentation and appropriate feature selection can reduce memorization.
4. **Faster training\:** Removing unnecessary features and reducing redundant data can improve computational efficiency.
5. **Better classification\:** Handling class imbalance improves the model's ability to recognize minority classes.
6. **More reliable predictions\:** Proper preprocessing and evaluation make model results more trustworthy.

## 8. Limitations of Data-Related Approaches

- Data preprocessing requires understanding of the dataset.
- Incorrect augmentation can generate misleading training examples.
- Feature selection may remove useful information.
- Oversampling can increase overfitting.
- Undersampling can discard important information.
- Synthetic samples may not accurately represent real-world observations.
- Poor handling of data leakage can produce misleading evaluation results.

---

## 9. Important Exam Questions

Prepare these questions after studying this topic.

**Short-answer questions (5 marks)\:**

1. Define data preprocessing and explain its importance in neural networks.
2. What is data augmentation? Explain any four augmentation techniques.
3. Differentiate between feature selection and feature extraction.
4. What is imbalanced data? Why is it a problem in classification?
5. Explain SMOTE.
6. Differentiate between oversampling and undersampling.

**Long-answer questions (10–15 marks)\:**

1. Explain the different data-related approaches used to improve neural network performance.
2. Explain data augmentation techniques for image, text, audio, and time-series data.
3. Describe the different feature selection methods with examples.
4. Explain various techniques for handling imbalanced datasets and discuss their advantages and disadvantages.
5. Explain how data preprocessing improves the training and generalization of neural networks.

## 10. Quick Revision

**Remember these four concepts**

(1\. Preprocessing)

**Make the data clean and usable.**

Missing values, duplicates, outliers, encoding, scaling, and splitting.

---

(2\. Augmentation)

**Make the training data more diverse.**

Rotation, flipping, cropping, noise, Mixup, and CutMix.

---

(3\. Feature Selection)

**Keep the most useful information.**

Filter methods, wrapper methods, and embedded methods.

---

(4\. Imbalanced Data)

**Give minority classes sufficient attention.**

Oversampling, undersampling, SMOTE, and class weighting.

