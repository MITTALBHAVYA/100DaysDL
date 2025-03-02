Neural Network for Placement Prediction

Overview

This project implements a simple 2-layer neural network to predict student placement based on their CGPA and IQ scores. The model is trained using logistic regression, forward propagation, and a basic parameter update method.

Dataset

A small dataset is created using Pandas with the following columns:

cgpa: Student's CGPA (Cumulative Grade Point Average)

iq: Student's IQ level

placed: Binary label (1 if placed, 0 otherwise)

Example dataset:

| cgpa | iq  | placed |
|------|---- |--------|
| 8    | 8   | 1      |
| 7    | 9   | 1      |
| 6    | 10  | 0      |
| 5    | 5   | 0      |

Implementation Details

1. Parameter Initialization

The function initialize_parameters(layer_dims) initializes weights (W) and biases (b) for each layer.

W is initialized with small values to avoid symmetry issues.

b is initialized to zero.

2. Forward Propagation

L_layer_forward(X, parameters): Computes the network output using matrix multiplications and a sigmoid activation function.

The equation for each layer is:


 (Sigmoid Activation)

3. Loss Function

The model uses binary cross-entropy loss:



4. Parameter Update

update_parameters(parameters, y, y_hat, A1, X): Adjusts weights and biases using a basic gradient update rule.

Learning rate is set to 0.0001.

5. Training Loop

The model runs for 50 epochs.

At each epoch, it updates parameters based on all training samples.

Prints the average loss per epoch.

Running the Code

Dependencies

Ensure you have Python and the following libraries installed:

pip install numpy pandas

Execute the Script

Run the Python script:

python placement_nn.py

Potential Improvements

Better Weight Initialization: Use np.random.randn() instead of fixed values.

Proper Gradient Descent Update: Follow standard multi-layer backpropagation.

Train-Test Split: Evaluate the model on unseen data.

Use of ML Libraries: Implement the model using TensorFlow or PyTorch for better efficiency.

Conclusion

This project demonstrates a basic neural network for binary classification using logistic regression. The model can be further improved for real-world applications.