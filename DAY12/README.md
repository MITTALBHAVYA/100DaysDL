# Early Stopping in Neural Networks

## Overview
**Early Stopping** is a regularization technique used in training deep neural networks to prevent overfitting. It stops the training process when the validation loss stops improving, ensuring the model generalizes well on unseen data. This repository contains an experiment demonstrating the impact of early stopping using TensorFlow and Keras.

## Requirements
To run the code, install the required dependencies:

```sh
pip install numpy pandas matplotlib seaborn tensorflow mlxtend scikit-learn
```

## Code Explanation

1. **Dataset Generation:**
   - The `make_circles` dataset is used to create a binary classification problem.
   - The dataset is split into **training** and **testing** sets using `train_test_split`.
   - A scatter plot is generated to visualize the dataset.

2. **Training Without Early Stopping:**
   - A simple neural network with two layers (256 neurons and an output neuron) is built using the **ReLU** activation function.
   - The model is compiled with the **Adam optimizer** and **binary cross-entropy loss**.
   - The model is trained for **2500 epochs** without early stopping.
   - The **loss curves** for training and validation data are plotted to observe overfitting.
   - A decision boundary is plotted using `plot_decision_regions`.

3. **Training With Early Stopping:**
   - A deeper network is built with an additional hidden layer of 50 neurons.
   - The **EarlyStopping** callback is implemented to monitor `val_loss` with the following parameters:
     - `patience=400`: Stops training if validation loss does not improve for 400 epochs.
     - `min_delta=0.00001`: Minimum change in loss to be considered as an improvement.
     - `restore_best_weights=False`: Keeps the last trained weights instead of restoring the best ones.
   - The model is trained again for **2500 epochs**, but early stopping prevents unnecessary training.
   - The **loss curves** are plotted to show where training stopped.
   - The final decision boundary is visualized.

## Key Observations
- **Without Early Stopping:**
  - The model continues training even after validation loss stops improving, leading to potential overfitting.
  - The loss curve shows an increasing gap between training and validation loss.
  
- **With Early Stopping:**
  - Training stops early, preventing overfitting and saving computational resources.
  - The model achieves similar accuracy with better generalization.
  - The validation loss remains stable, avoiding unnecessary training.

## Conclusion
Early stopping helps improve model efficiency by stopping training at the optimal point, reducing overfitting and saving computation time. This experiment demonstrates how setting appropriate patience and delta values can enhance neural network performance.

## References
- [TensorFlow Early Stopping Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
- [Keras Callbacks](https://keras.io/api/callbacks/early_stopping/)
- [Understanding Overfitting](https://en.wikipedia.org/wiki/Overfitting)

