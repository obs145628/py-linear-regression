import numpy as np



'''
Compute cost function
@param x matrix of size (set_len, input_len)
@param y vector of size (set_len) - expected values
@param w vector of size (input_len) - weights
@param b vector of size (1) - bias
@return cost value
'''
def cost_function(X, y, w, b):
    y_hat = np.dot(X, w) + b
    res = 0.5 * np.sum((y - y_hat) ** 2)
    return res


def evaluate(X, y, w, b):
    print('Cost: ' + str(cost_function(X, y, w, b)))


'''
Apply stochastic gradient descent on the whole training set of size set_len
@param x matrix of size (set_len, input_len)
@param y vector of size (set_len) - expected values
@param w vector of size (input_len) - weights
@param b vector of size (1) - bias
@param lr - learning rate
@return vector (input_len) updated weights, updated biais
'''
def sgd(X, y, w, b, lr):
    y_hat = np.dot(X, w) + b
    dW = np.dot(X.T, y_hat - y)
    dB = sum(y_hat - y)
    w = w - lr * dW
    b = b - lr * dB
    return w, b


'''
@param x matrix of size (set_len, input_len)
@param y vector of size (set_len) - expected values
@param epochs - number of epochs of learning
@param lr - learning rate

Run the training for several epochs.
After each epochs, the wieghts are tested on the training set
'''
def train(X, y, epochs, lr):

    w = np.random.randn(X.shape[1])
    b = np.random.randn(1)

    #Training
    for i in range(1, epochs + 1):
        print('Epoch :' + str(i))
        w, b = sgd(X, y, w, b, lr)
        evaluate(X, y, w, b)

    return w, b
