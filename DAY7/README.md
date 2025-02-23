Explanation of the Code

This Python script implements a neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. Let's break down the code step by step.
1. Importing Required Libraries

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

    tensorflow: A machine learning framework used for deep learning tasks.
    keras: A high-level API for building neural networks in TensorFlow.
    Sequential: A linear stack of layers that forms a simple feedforward neural network.
    Dense: A fully connected (dense) layer where each neuron is connected to all neurons in the previous layer.
    Flatten: Converts multi-dimensional inputs into a one-dimensional vector.

2. Loading the MNIST Dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).
    X_train and X_test: Images (grayscale, size 28x28 pixels).
    y_train and y_test: Labels (corresponding digit for each image).

X_test.shape
y_train

    X_test.shape outputs the shape of the test dataset, which should be (10000, 28, 28).
    y_train shows the labels of the training dataset.

3. Displaying a Sample Image

import matplotlib.pyplot as plt
plt.imshow(X_train[1])

    plt.imshow(X_train[1]) displays the second image in the training dataset.

4. Data Normalization

X_train = X_train / 255
X_test = X_test / 255

    Pixel values in the images range from 0 to 255.
    To improve model performance, we normalize the data to the range 0 to 1 by dividing by 255.

5. Building the Neural Network

model = Sequential()

    Creates a sequential model where layers are added one after another.

model.add(Flatten(input_shape=(28,28)))

    Flatten converts the 28x28 image into a 1D array of 784 pixels to be processed by dense layers.

model.add(Dense(128, activation='relu'))

    Dense(128, activation='relu'): A fully connected layer with 128 neurons and ReLU activation function.

model.add(Dense(32, activation='relu'))

    Another dense layer with 32 neurons and ReLU activation function.

model.add(Dense(10, activation='softmax'))

    Final output layer with 10 neurons, one for each digit (0-9).
    Softmax activation function converts the output into probability distribution over 10 classes.

model.summary()

    Displays the model architecture, number of parameters, and layer information.

6. Compiling the Model

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    Loss Function: 'sparse_categorical_crossentropy' (used for multi-class classification with integer labels).
    Optimizer: 'Adam' (efficient optimization algorithm).
    Metrics: ['accuracy'] to track the classification accuracy.

7. Training the Model

history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

    Trains the model for 25 epochs.
    Uses 80% of training data for training and 20% for validation.

8. Making Predictions

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

    model.predict(X_test): Outputs probability distribution for each digit.
    argmax(axis=1): Picks the class (digit) with the highest probability.

9. Evaluating Model Performance

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

    Calculates the accuracy by comparing predictions (y_pred) with actual labels (y_test).

10. Plotting Training History

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

    Plots training loss, validation loss, training accuracy, and validation accuracy over 25 epochs.

11. Predicting a Single Image

model.predict(X_test[3].reshape(1,28,28)).argmax(axis=1)

    Predicts the digit for the 4th image in the test dataset.

README.md File

Here’s a README for the project:

# MNIST Handwritten Digit Classification

This project implements a deep learning model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Dataset
The MNIST dataset consists of:
- **60,000 training images** and **10,000 test images** of handwritten digits (0-9).
- Images are grayscale with a size of **28x28 pixels**.

## Model Architecture
The neural network consists of:
1. **Flatten Layer**: Converts 28x28 images into a 1D array of 784 values.
2. **Dense Layer (128 neurons, ReLU activation)**: First hidden layer.
3. **Dense Layer (32 neurons, ReLU activation)**: Second hidden layer.
4. **Dense Layer (10 neurons, Softmax activation)**: Output layer for 10 digit classes.

## Installation
Make sure you have Python and the required dependencies installed:

```bash
pip install tensorflow matplotlib scikit-learn

Running the Model

Run the Python script:

python mnist_classification.py

Training

The model is trained for 25 epochs with an 80-20 train-validation split.
Evaluation

The model's accuracy is calculated using Scikit-Learn:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

Results

The training and validation accuracy/loss are plotted using Matplotlib.
Prediction

To predict a digit from the test dataset:

model.predict(X_test[3].reshape(1,28,28)).argmax(axis=1)

Author

    Bhavya Mittal

---

This README provides a structured overview of the project, including installation, running instructions, and results.

