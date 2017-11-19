import numpy as np

def forward(Z):
    max_z = np.max(Z, axis = 0)
    S = np.exp(Z - max_z)/(1e-8 + np.sum(np.exp(Z - max_z), axis = 0, keepdims = True))
    return S

def backward(AL, Y):
    dZ = AL - Y
    return dZ