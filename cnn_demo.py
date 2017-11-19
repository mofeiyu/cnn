from cnn.prepare_data import ReaderWrapper
from cnn.test_accuracy import test_accuracy
from cnn.train_accuracy import train_accuracy
from cnn.cost_function import cost_function

from cnn.mini_batch import mini_batch_data
from cnn.initialization_ramdon import initialize_ramdon
from cnn.adam import AdamOptimizer

import cnn.conv as conv
import cnn.pooling as pool
import cnn.fc_linear as fc
import cnn.relu as relu
import cnn.softmax as softmax
  
class CNNModel:
    def __init__(self):
        self._reader = ReaderWrapper()
 
    def train(self, stride = 1, fliter = 5, pad = 2,para_dims = ([5,5,1,8],[5,5,8,16],[7*7*16,16],[16,10]), learning_rate = 0.0001, num_iterations = 30):
        train_X, train_Y = self._reader.get_training_data()
        self._para_dims = para_dims
        self._stride = stride
        self._fliter = fliter
        self._pad = pad
        self._learning_rate = learning_rate    
        self._parameters = initialize_ramdon(para_dims)
        self._L = len(para_dims)
        costs = []
        accuracys = []
        self._grads = {}
        self._caches = []
        self._AL = None
        self._mean = None
        self._var = None
        adam = AdamOptimizer(self._learning_rate, para_dims)
        for i in range(num_iterations):
            mini_batchs = mini_batch_data(train_X, train_Y,1000)
            hit_count = 0         
            for X, Y in mini_batchs:
                self._forward(X)
                cost = cost_function(self._AL, Y, self._L)
                self._backward(Y)
                self._parameters = adam.update_parameters(self._parameters, self._grads, self._L)
                h_c =  train_accuracy(self._AL, Y)
                hit_count += h_c
                costs.append(cost)
            print ('iter %d cost %f, h_c = %d, hit_ratio = %.2lf%%' % (i, cost, hit_count, hit_count * 100.0 / train_X.shape[0]))
            accuracy = hit_count * 100.0 / train_X.shape[0]
            accuracys.append(accuracy)
        return costs, accuracys
    
    def _predict(self, A, Y):  
        for l in range(self._L):
            if len(self._para_dims[l]) != 2:
                Z, _ = conv.forward(A, self._parameters['W' + str(l+1)], self._parameters['b' + str(l+1)], self._stride, self._pad)
                R = relu.forward(Z)
                A = pool.forward(R, self._fliter, self._stride)
            else:
                Z, _ = fc.forward(A, self._parameters['W' + str(l+1)], self._parameters['b' + str(l+1)])
                A = relu.forward(Z)
        Z, _ = fc.forward(A, self._parameters['W' + str(self._L)], self._parameters['b' + str(self._L)])
        AL, _ = softmax.forward(Z) 
        cost = cost_function(AL, Y, self._L, self._parameters)
        hit_count = test_accuracy(AL, Y)
        print ('cost %f, hit_count = %d, hit_ratio = %.2lf%%' % (cost, hit_count, hit_count * 100.0 / A.shape[0]))
        return (cost, hit_count * 100.0 / A.shape[0])
        
    def test(self):
        X, Y = self._reader.get_test_data()
        return self._predict(X, Y)
        
    def validation(self):
        X, Y = self._reader.get_validation_data()
        return self._predict(X, Y)

    def _forward(self, X):
        caches = []
        A = X
        for l in range(self._L-1):
            if len(self._para_dims[l]) != 2:
                Z, cache = conv.forward(A, self._parameters['W' + str(l+1)], self._parameters['b' + str(l+1)], self._stride, self._pad)
                R = relu.forward(Z)
                caches.append((cache, Z))
                A = pool.forward(R, 2, 2)
            else:
                Z, cache = fc.forward(A, self._parameters['W' + str(l+1)], self._parameters['b' + str(l+1)])
                A = relu.forward(Z)
                caches.append((cache, Z))
        Z, cache = fc.forward(A, self._parameters['W' + str(self._L)], self._parameters['b' + str(self._L)])
        AL = softmax.forward(Z)  
        caches.append((cache, Z))
        self._caches = caches
        self._AL = AL

    def _backward(self, Y):
        grads = {}
        current_cache = self._caches[-1]
        dZ = softmax.backward(self._AL, Y)
        dA_prev_temp, dW_temp, db_temp = fc.backward(dZ, current_cache)    
        grads["dA" + str(self._L)] = dA_prev_temp
        grads["dW" + str(self._L)] = dW_temp
        grads["db" + str(self._L)] = db_temp
        
        for l in reversed(range(self._L-1)):
            current_cache = self._caches[l]
            if len(self._para_dims[l]) == 2:
                dZ = relu.backward(grads["dA" + str(l+2)], current_cache)
                dA_prev_temp, dW_temp, db_temp = fc.backward(dZ, current_cache)
                grads["dA" + str(l + 1)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
            else:
                dA = pool.backward(grads["dA" + str(l+2)], current_cache, 2, 2)
                dZ = relu.backward(dA, current_cache)
                dA_prev_temp, dW_temp, db_temp = conv.backward(dZ, current_cache, self._stride, self._pad)
                grads["dA" + str(l + 1)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp                
        self._grads = grads

def main():
    cnn_model = CNNModel()
    cnn_model.train()
    cnn_model.validation()
    cnn_model.test()
    
if __name__ == '__main__':
    main()