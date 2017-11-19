import numpy as np

def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values =(0,0))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(W, a_slice_prev)
    Z = np.sum(np.sum(s))
    Z = Z + b
    return Z

def forward(A_prev, W, b, stride, pad):
    (m, n_H_prev, n_W_prev, _) = np.shape(A_prev)
    (f, _, _, n_C) = np.shape(W)
    n_H = int((n_H_prev + 2 * pad - f)/ stride + 1)
    n_W = int((n_W_prev + 2 * pad - f)/ stride + 1)
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)        
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]                               # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    vert_start = h*stride
                    vert_end = h*stride + f - 1
                    horiz_start = w*stride
                    horiz_end = w*stride + f - 1
                    a_slice_prev = a_prev_pad[vert_start:vert_end + 1,horiz_start:horiz_end + 1,:]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[c])         
    cache = (A_prev, W, b)
    return Z, cache

def backward(dZ, cache, stride, pad):
    ((A_prev, W, b), _) = cache
    (f, _, _, _) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros(A_prev.shape)                          
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape) 
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    for i in range(m):                       # loop over the training examples
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] +=  W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    return dA_prev, dW, db
