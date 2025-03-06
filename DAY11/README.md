# Vanishing Gradient Problem

## Overview
The **Vanishing Gradient Problem** is a well-known issue in deep neural networks, particularly when using activation functions like **sigmoid** or **tanh**. This problem occurs when gradients become increasingly small as they propagate back through layers during backpropagation, leading to minimal weight updates and preventing deep networks from learning effectively.

This repository contains a simple experiment to visualize and understand the **Vanishing Gradient Problem** by training two deep neural networks on the `make_moons` dataset:
1. A deep neural network with **sigmoid** activation functions.
2. A deep neural network with **ReLU** activation functions.

By comparing the weight updates, we can observe the impact of activation functions on the gradient flow and confirm how sigmoid activations contribute to the vanishing gradient problem.

## Requirements
To run the code, you need the following Python packages:

```sh
pip install numpy pandas matplotlib tensorflow keras scikit-learn
```

## Code Explanation

1. **Dataset Generation:**
   - The `make_moons` dataset is generated with 250 samples, adding some noise.
   - The dataset is then split into **training** and **testing** sets.
   
2. **Deep Neural Network with Sigmoid Activation:**
   - A deep network (10 layers of 10 neurons each) is created using the **sigmoid** activation function.
   - The model is compiled using `binary_crossentropy` loss and `adam` optimizer.
   - The model is trained for 100 epochs.
   - Initial and final weights of the first layer are compared to compute the gradient and percentage change.
   
3. **Deep Neural Network with ReLU Activation:**
   - Another deep network with the same architecture is built, but using **ReLU** activation instead of sigmoid.
   - The same training process is repeated.
   - The weight updates are compared to analyze how ReLU helps mitigate the vanishing gradient problem.

## Key Observations

- **Sigmoid Activation:**
  - The gradients diminish significantly, especially in deeper layers, leading to **small weight updates**.
  - The percent change in weights is minimal, indicating poor learning.
  
- **ReLU Activation:**
  - The gradients remain more stable and do not shrink as much.
  - The network learns better and updates weights more effectively.

## Conclusion
This experiment demonstrates that **sigmoid activation** functions suffer from the **vanishing gradient problem**, making it harder for deep networks to learn. **ReLU activation** mitigates this issue by allowing better gradient propagation and learning efficiency.

## References
- "Understanding the Vanishing Gradient Problem" - [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

