Neural Network Implementation

Overview

This project implements a basic neural network from scratch using NumPy and Pandas. The neural network is trained using a simple dataset that contains CGPA, IQ, and LPA (likely a target variable representing salary or some outcome measure). The network consists of two layers and updates parameters using a basic gradient descent approach.

Dependencies

Ensure you have the following Python libraries installed:

pip install numpy pandas

Dataset

The dataset is stored in a Pandas DataFrame with the following columns:

cgpa: Cumulative Grade Point Average

iq: Intelligence Quotient

lpa: Target variable (possibly salary in LPA)

df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]],columns=['cgpa','iq','lpa'])

Functions

initialize_parameters(layer_dims)

Initializes weight matrices with small values (0.1) and bias terms as zero.

Returns a dictionary containing the initialized parameters.

liner_forward(A_prev, W, b)

Performs forward propagation for a single layer.

Computes Z = W.T * A_prev + b.

L_layer_forward(X, parameters)

Implements forward propagation through the entire network.

Iterates over layers and applies the forward function.

update_parameters(parameters, y, y_hat, A1, X)

Updates network parameters using gradient descent.

Adjusts weights and biases based on the error.

Training Process

The neural network is trained over multiple epochs:

A forward pass is performed.

The parameters are updated using a simple weight adjustment rule.

Loss is computed and printed after each epoch.

Training Execution

parameters = initialize_parameters([2,2,1])
epochs = 5

for i in range(epochs):
    Loss = []
    for j in range(df.shape[0]):
        X = df[['cgpa', 'iq']].values[j].reshape(2,1)
        y = df[['lpa']].values[j][0]

        y_hat, A1 = L_layer_forward(X, parameters)
        y_hat = y_hat[0][0]

        update_parameters(parameters, y, y_hat, A1, X)
        Loss.append((y - y_hat) ** 2)
    
    print('Epoch - ', i+1, 'Loss - ', np.array(Loss).mean())

Potential Improvements

Implement activation functions like ReLU or Sigmoid.

Use a proper gradient descent update instead of manual weight adjustments.

Implement backpropagation for better learning.

Use a library like TensorFlow or PyTorch for scalability.

Conclusion

This basic neural network is a starting point for understanding forward propagation and parameter updates using a dataset. Further improvements can be made by introducing non-linear activations and optimizing the learning process.