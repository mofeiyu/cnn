import numpy as np

def initialize_ramdon(para_dims):
    parameters = {}
    L = len(para_dims)
    for l in range(0,L):
        if len(para_dims[l]) == 4:
            parameters['W' + str(l+1)] = np.random.randn(para_dims[l][0],para_dims[l][1],para_dims[l][2],para_dims[l][3])/10
            parameters['b' + str(l+1)] = np.zeros((para_dims[l][3],1))
        else :
            parameters['W' + str(l+1)] = np.random.randn(para_dims[l][1],para_dims[l][0])/10
            parameters['b' + str(l+1)] = np.zeros((para_dims[l][1],1))
    return parameters
