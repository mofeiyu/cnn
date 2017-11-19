def sgd(learning_rate, parameters, grads, L):
    for l in range(L):
        l = str(l + 1)
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    return parameters