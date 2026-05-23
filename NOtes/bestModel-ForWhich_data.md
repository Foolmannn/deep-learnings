## Summary of major machine learning model types and the best kind of data to use them for:


### 1. Supervised Learning
Used when you have **labeled data** (input-output pairs) for prediction or classification.

*   **Linear Regression**: Best for predicting **continuous values** (e.g., house prices) with **linear relationships** and **low feature complexity**.
*   **Logistic Regression**: Ideal for **binary classification** (e.g., spam detection) with **interpretable results** and **moderate-sized datasets**.
*   **Decision Trees & Random Forest**: Excel with **mixed data types** (numerical and categorical), **non-linear relationships**, and when **interpretability** is key. Random Forest reduces overfitting.
*   **Support Vector Machine (SVM)**: Strong for **small-to-medium datasets** with **high-dimensional features** (e.g., text classification), especially with clear class separation.
*   **Neural Networks (Deep Learning)**: Best for **very large datasets** and **complex, unstructured data** like images (CNNs), sequences/text (RNNs, Transformers), or speech.

### 2. Unsupervised Learning
Used with **unlabeled data** to discover hidden patterns.

*   **K-Means Clustering**: Best for **customer segmentation** or grouping similar data points when you know the number of groups (`k`) and have **well-separated clusters**.
*   **Principal Component Analysis (PCA)**: Ideal for **dimensionality reduction**, **noise reduction**, and **data visualization** in high-dimensional datasets.

### 3. Reinforcement Learning
Used for **sequential decision-making** in dynamic environments (e.g., robotics, game AI), learning from rewards/penalties.

### 4. Deep Learning (Subset of Supervised/Unsupervised)
*   **Convolutional Neural Networks (CNNs)**: Best for **image and video data**.
*   **Recurrent Neural Networks (RNNs) / LSTMs**: Best for **time-series data** and **sequential data** like text or speech.
*   **Transformers**: State-of-the-art for **Natural Language Processing (NLP)** tasks like translation and text generation.

