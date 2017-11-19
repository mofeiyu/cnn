#encode=utf-8

import numpy as np

class AdamOptimizer():
    def __init__(self, learning_rate,para_dims):
        self._learning_rate = learning_rate
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._epsilon = 1e-8
        self._t = 0
        L = len(para_dims)
        self._v = {}
        self._s = {}
        for l in range(0, L):
            if len(para_dims[l]) == 4:
                self._v["dW" + str(l + 1)] = np.zeros((para_dims[l][0],para_dims[l][1],para_dims[l][2],para_dims[l][3]))
                self._v["db" + str(l + 1)] = np.zeros((para_dims[l][3], 1))
                self._s["dW" + str(l + 1)] = np.zeros((para_dims[l][0],para_dims[l][1],para_dims[l][2],para_dims[l][3]))
                self._s["db" + str(l + 1)] = np.zeros((para_dims[l][3], 1))                
            else:   
                self._v["dW" + str(l + 1)] = np.zeros((para_dims[l][1],para_dims[l][0]))
                self._v["db" + str(l + 1)] = np.zeros((para_dims[l][1],1))
                self._s["dW" + str(l + 1)] = np.zeros((para_dims[l][1],para_dims[l][0]))
                self._s["db" + str(l + 1)] = np.zeros((para_dims[l][1],1))
    
    def update_parameters(self, parameters, grads, L):
        self._t += 1
        for l in range(L):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._v[dW] = self._beta1 * self._v[dW] + (1 - self._beta1) * grads[dW]
            self._v[db] = self._beta1 * self._v[db] + (1 - self._beta1) * grads[db]
            v_dW_corrected = self._v[dW] / (1 - self._beta1 ** self._t)
            v_db_corrected = self._v[db] / (1 - self._beta1 ** self._t)
            # update s
            self._s[dW] = self._beta2 * self._s[dW] + (1 - self._beta2) * (grads[dW] ** 2)
            self._s[db] = self._beta2 * self._s[db] + (1 - self._beta2) * (grads[db] ** 2)
            s_dW_corrected = self._s[dW] / (1 - self._beta2 ** self._t)
            s_db_corrected = self._s[db] / (1 - self._beta2 ** self._t)
            # update parameters
            parameters['W' + str(l+1)] -= self._learning_rate * v_dW_corrected / (s_dW_corrected ** 0.5 + self._epsilon)
            parameters['b' + str(l+1)] -= self._learning_rate * v_db_corrected / (s_db_corrected ** 0.5 + self._epsilon)
        return parameters
