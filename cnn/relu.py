import numpy as np

def forward(Z):
    temp = Z > 0
    return Z * temp

def backward(dA, cache):
    (_, Z) = cache
    r =  Z > 0
    r = np.multiply(dA,r)
    return r