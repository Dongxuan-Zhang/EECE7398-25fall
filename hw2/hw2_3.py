import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def MSE(y_hat, y):
    return (y - y_hat) ** 2

def MSE_derivative(y_hat, y):
    return 2 * (y - y_hat)

# input: x1, x2, x3, x4
x1, x2, x3, x4 = -0.3, 4.9, 1.1, -2.7

# weight: w1, w2, w3, w4, w5, w6
w1, w2, w3, w4, w5, w6 = -1.7, 0.1, -0.6, -1.8, -0.2, 0.5

# forward propagation: h1 and h2
s1 = w1 * x1 + w2 * x2
h1 = sigmoid(s1)

s2 = w3 * x3 + w4 * x4
h2 = sigmoid(s2)

# forward propagation: y_hat
s3 = w5 * h1 + w6 * h2
y_hat = sigmoid(s3)

# backward propagation

# true label: y = 0.7
y = 0.7

# loss function: MSE
loss = MSE(y_hat, y)

# output layer
delta_3 = MSE_derivative(y_hat, y) * sigmoid_derivative(s3)

# hidden layer
delta_1 = delta_3 * w5 * sigmoid_derivative(s1)
delta_2 = delta_3 * w6 * sigmoid_derivative(s2)

# calculate the gradient
dw1 = delta_1 * x1
dw2 = delta_1 * x2
dw3 = delta_2 * x3
dw4 = delta_2 * x4
dw5 = delta_3 * h1
dw6 = delta_3 * h2

# print the result
print(f"y_hat = {y_hat:.4f}")
print(f"gradient:")
print(f"dw1 = {dw1:.4f}, dw2 = {dw2:.4f}")
print(f"dw3 = {dw3:.4f}, dw4 = {dw4:.4f}")
print(f"dw5 = {dw5:.4f}, dw6 = {dw6:.4f}")