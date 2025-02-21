## **Hinge Loss Perceptron Explanation**

This Python script implements a **Perceptron model with Hinge Loss** to classify two types of data points and draws a **decision boundary** (a straight line) to separate them.

---

### **How It Works**

#### **1. Importing Required Libraries**
```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
```
- `make_classification` → Generates a synthetic dataset for classification.
- `numpy` (`np`) → Used for mathematical operations.
- `matplotlib.pyplot` (`plt`) → Used for plotting graphs.

---

#### **2. Creating a Dataset**
```python
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,
                           hypercube=False, class_sep=15)
```
- **Creates 100 points** with two features (X and Y coordinates).
- **`class_sep=15`** ensures the two classes are well-separated.

---

#### **3. Visualizing the Data**
```python
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
```
- Plots the data points with different colors for each class.

---

#### **4. Perceptron Model with Hinge Loss**
```python
def perceptron(X, y):
    w1 = w2 = b = 1  # Initialize weights and bias
    lr = 0.1  # Learning rate
    
    for j in range(1000):  # Train for 1000 iterations
        for i in range(X.shape[0]):
            z = w1 * X[i][0] + w2 * X[i][1] + b  # Compute weighted sum
            
            # Hinge Loss Condition
            if z * y[i] < 0:
                w1 = w1 + lr * y[i] * X[i][0]
                w2 = w2 + lr * y[i] * X[i][1]
                b = b + lr * y[i]
                
    return w1, w2, b
```
- **Trains the Perceptron using Hinge Loss**:
  - If `z * y[i] < 0`, the prediction is wrong → update weights.
  - Adjusts `w1`, `w2`, and `b` based on the learning rate (`lr`).

---

#### **5. Training the Model**
```python
w1, w2, b = perceptron(X, y)
```
- Runs the perceptron function and retrieves final weights.

---

#### **6. Plotting the Decision Boundary**
```python
m = -(w1 / w2)  # Slope of decision boundary
c = -(b / w2)   # Intercept

x_input = np.linspace(-3, 3, 100)  # Creates X values
y_input = m * x_input + c  # Computes corresponding Y values

plt.figure(figsize=(10,6))
plt.plot(x_input, y_input, color='red', linewidth=3)  # Draws decision boundary
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)  # Original points
plt.ylim(-3, 2)  # Adjusts Y-axis limits
```
- Computes the decision boundary `y = mx + c` and plots it.
- Shows the **red line** separating the two classes.

---

## **README.md (For the Project)**
```markdown
# Hinge Loss Perceptron Classifier

This project implements a **Perceptron model with Hinge Loss** for binary classification. The model is trained on a synthetic dataset and visualizes the decision boundary.

## **How It Works**
1. Generates a dataset with `make_classification()`.
2. Plots the data with two different colors.
3. Trains a **Perceptron with Hinge Loss**.
4. Draws a **decision boundary** separating the two classes.

## **Installation**
Make sure you have Python installed and install the required libraries:
```bash
pip install numpy matplotlib scikit-learn
```

## **Usage**
Run the script:
```bash
python perceptron.py
```

## **What You Will See**
- A **scatter plot** showing data points.
- A **red line (decision boundary)** separating two classes.

## **Limitations**
- The Perceptron only works for **linearly separable** data.
- Does not work well for complex patterns.

## **Improvement Ideas**
- Try using **feature scaling** with `StandardScaler()`.
- Test different **learning rates (`lr`)**.
- Use **SVM with kernels** for non-linearly separable data.

## **License**
This project is for learning purposes only.
```

