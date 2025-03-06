# Regularization in Neural Networks

This repository demonstrates the effect of L1 and L2 regularization in neural networks using TensorFlow and Keras. The project uses a synthetic dataset (`make_moons`) to visualize decision boundaries and analyze weight distributions.

## Overview
Regularization techniques like L1 help prevent overfitting by adding penalties to the loss function, encouraging sparsity in weights. This notebook compares a standard neural network with one that includes L1 regularization.

## Requirements
Install the necessary dependencies before running the notebook:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn mlxtend
```

## Usage
1. Run the script to:
   - Generate and visualize the dataset (`make_moons`)
   - Train a standard neural network
   - Train a network with L1 regularization
   - Compare decision boundaries and loss curves
   - Analyze weight distributions

2. Execute the script step by step to observe the impact of regularization on training.

## Dataset
- The dataset is generated using `make_moons`, which creates a two-class classification problem with noise.
- `matplotlib.pyplot.scatter` is used to visualize the dataset before training.

## Model Architecture
Two models are created:
1. **Standard Model:**
   - Fully connected layers with ReLU activation.
   - Trained with Adam optimizer and binary cross-entropy loss.

2. **Regularized Model:**
   - Similar architecture but with L1 regularization (`kernel_regularizer=tensorflow.keras.regularizers.l1(0.001)`).

## Results & Observations
- **Decision Boundaries:**
  - The `plot_decision_regions` function is used to visualize decision regions for both models.
- **Loss Comparison:**
  - Training and validation loss are plotted to compare convergence.
- **Weight Analysis:**
  - Weight distributions are analyzed using seaborn (`sns.boxplot` and `sns.distplot`).
  - Regularization leads to more constrained weight values, preventing extreme variations.

## Conclusion
L1 regularization encourages sparsity, leading to improved generalization and robustness. This project highlights its effect by comparing decision boundaries, loss curves, and weight distributions.

## License
This project is open-source under the MIT License.

