import numpy as np

# weight
W = np.array([[-0.57, 1.24, -3.37, 6.43],
              [-5.53, -1.13, -8.05, 3.21],
              [4.23, 0.98, -2.53, -7.67],
              [-2.31, -1.84, 6.93, -8.66]])

# input
input_1 = np.array([[1.52],[2.63],[5.37],[4.94]])
input_2 = np.array([[8.87],[1.25],[4.49],[0.12]])
input_3 = np.array([[3.22],[4.63],[3.55],[5.41]])
input_4 = np.array([[1.38],[0.63],[2.90],[8.52]])

# true label
true_label_1 = 0
true_label_2 = 0
true_label_3 = 1
true_label_4 = 3

# softmax loss
def softmax_loss(W, input_data, true_label):
    product = np.dot(W, input_data)
    return -np.log10(np.exp(product[true_label]) / np.sum(np.exp(product)))

loss_1 = softmax_loss(W, input_1, true_label_1)
loss_2 = softmax_loss(W, input_2, true_label_2)
loss_3 = softmax_loss(W, input_3, true_label_3)
loss_4 = softmax_loss(W, input_4, true_label_4)

def softmax_loss_avg(W, input_data_list, true_labels):
    total_loss = 0
    for input_data, true_label in zip(input_data_list, true_labels):
        loss = softmax_loss(W, input_data, true_label)
        total_loss += loss
    return total_loss / len(input_data_list)

input_data_list = [input_1, input_2, input_3, input_4]
true_labels = [true_label_1, true_label_2, true_label_3, true_label_4]

loss_avg = softmax_loss_avg(W, input_data_list, true_labels)

# print the loss
print("loss 1:", np.round(loss_1, 4))
print("loss 2:", np.round(loss_2, 4))
print("loss 3:", np.round(loss_3, 4))
print("loss 4:", np.round(loss_4, 4))
print("average loss:", np.round(loss_avg, 4))

# numerical method to calculate the gradient
def compute_numerical_gradient(W, input_data_list, true_labels, epsilon=1e-7):
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += epsilon
            W_minus = W.copy()
            W_minus[i, j] -= epsilon
            
            loss_plus = softmax_loss_avg(W_plus, input_data_list, true_labels)
            loss_minus = softmax_loss_avg(W_minus, input_data_list, true_labels)
            
            grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    return grad

# calculate the gradient of each sample
grad = compute_numerical_gradient(W, input_data_list, true_labels)

# print the average gradient
print("average gradient:")
print(np.round(grad, 4))

# vanila gradient descent
def vanila_gradient_descent(W, input_data_list, true_labels, alpha, num_iterations):
    for _ in range(num_iterations):
        grad = compute_numerical_gradient(W, input_data_list, true_labels)
        W -= alpha * grad
    return W

learing_rate = 0.2
W_vanila = vanila_gradient_descent(W, input_data_list, true_labels, learing_rate, num_iterations=1) 

# print the vanila gradient descent
print("vanila gradient descent:")
print(np.round(W_vanila, 4))

