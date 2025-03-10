# Batch Normalization in Deep Learning

## Overview
This project demonstrates the effect of Batch Normalization in a simple feedforward neural network. The dataset used consists of concentric circles, making it a binary classification problem.

## Dataset
- The dataset is loaded from a CSV file: `concertriccir2.csv`
- It contains two input features (`X`, `Y`) and a target class (`class`).

## Dependencies
Ensure you have the following libraries installed before running the script:
```bash
pip install numpy pandas matplotlib tensorflow keras
```

## Code Explanation
### 1. Load Dataset
- The dataset is read using `pandas`.
- Data visualization is done using `matplotlib` to understand the distribution of points.

### 2. Model Without Batch Normalization
- A simple neural network with two hidden layers using ReLU activation.
- The final layer uses a sigmoid activation function for binary classification.
- Compiled using `binary_crossentropy` loss and `adam` optimizer.

### 3. Model With Batch Normalization
- A similar structure, but with `BatchNormalization` layers added after `Dense` layers.
- Helps stabilize training and improve convergence speed.

### 4. Training & Comparison
- Both models are trained for 200 epochs using an 80-20 training-validation split.
- The validation accuracy of both models is plotted for comparison.

## Results
- The model with Batch Normalization generally achieves better validation accuracy.
- It also converges faster compared to the model without Batch Normalization.

## Running the Code
Run the Python script to train the models and visualize the results:
```python
python batch_normalization.py
```

## Author
BHAVYA
