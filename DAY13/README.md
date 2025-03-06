# Feature Scaling in Machine Learning

## Overview
Feature scaling is a crucial step in preprocessing data for machine learning models. It ensures that all features contribute equally to the learning process, preventing some features from dominating due to differences in scale. This project demonstrates feature scaling using the `StandardScaler` from `sklearn.preprocessing` and evaluates its impact on a simple neural network model.

## Dataset
The dataset used is `Social_Network_Ads.csv`, which contains user data for a social network and whether they made a purchase (binary classification). The dataset is preprocessed by selecting relevant columns before training the model.

## Dependencies
Ensure you have the following libraries installed:
```sh
pip install numpy pandas seaborn tensorflow scikit-learn matplotlib
```

## Code Workflow
1. **Load and Preprocess Data:**
   - Read the dataset using `pandas`.
   - Select the relevant features and target variable.
   - Split the data into training and testing sets using `train_test_split`.

2. **Train a Neural Network without Feature Scaling:**
   - Define a simple feedforward neural network using `keras`.
   - Train the model on raw (unscaled) data.
   - Plot validation accuracy.

3. **Apply Feature Scaling using StandardScaler:**
   - Scale the features using `StandardScaler`.
   - Train the neural network on scaled data.
   - Compare validation accuracy after scaling.

## Results
- A scatter plot of the features before and after scaling is generated using `seaborn.scatterplot`.
- The validation accuracy is plotted before and after applying feature scaling to observe improvements.
- The model is expected to converge faster and perform better when feature scaling is applied.

## Conclusion
Feature scaling significantly impacts neural network training, improving convergence and accuracy by ensuring all features contribute equally. This project highlights the importance of scaling in machine learning workflows.

