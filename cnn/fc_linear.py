import numpy as np

def forward(A, W, b):
    if len(np.shape(A)) == 4:
        (m, n_H, n_W, n_C) = np.shape(A)
        A = A.reshape(n_H*n_W*n_C, m)
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache

def backward(dZ, cache):   
    ((A_prev, W, _),_)= cache
    m = A_prev.shape[1] 
    dW = np.dot(dZ, A_prev.T)*1/m
    db = np.sum(dZ, axis = 1, keepdims = True)*1/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db
