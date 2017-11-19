import numpy as np
import cnn.relu as relu
def forward(A_prev, f, stride):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for h in range(n_H):                     # loop on the vertical axis of the output volume
        for w in range(n_W):                 # loop on the horizontal axis of the output volume
            for c in range (n_C):            # loop over the channels of the output volume
                vert_start = h*stride
                vert_end = h*stride + f - 1
                horiz_start = w*stride
                horiz_end = w*stride + f - 1
                a_prev_slice = A_prev[:,vert_start:vert_end+1,horiz_start:horiz_end+1,c]
                A[:, h, w, c] = np.max(a_prev_slice)
    return A  
def create_mask_from_window(x):
    mask = ( x == np.max(x))
    return mask
def backward(dA, cache, f, stride):
    (_,Z) = cache
    A_prev = relu.forward(Z)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    if len(np.shape(dA)) == 2:
        dA = dA.reshape(m, (n_H_prev-f)/stride + 1,(n_W_prev-f)/stride + 1, n_C_prev)
    _, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    for h in range(n_H):                   # loop on the vertical axis
        for w in range(n_W):               # loop on the horizontal axis
            for c in range(n_C):           # loop over the channels (depth)
                vert_start = h*stride
                vert_end = h*stride + f
                horiz_start = w*stride
                horiz_end = w*stride + f
                a_prev_slice =  A_prev[:,vert_start:vert_end,horiz_start:horiz_end,c]
                mask = create_mask_from_window(a_prev_slice)
                dA_tmp = dA[:,h,w,c].reshape((m,1,1)) 
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, c] +=  np.multiply(mask, dA_tmp)
    return dA_prev