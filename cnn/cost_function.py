import numpy as np

def cost_function(AL,Y, L = None,  parameters = None):
    m = AL.shape[1]
    if Y.shape != AL.shape:
        x = np.zeros(AL.shape)
        for y in Y:
            x[y][Y[y]] = 1
        Y = x  
    cost = np.sum(np.multiply(np.log(AL + 1e-8), Y)) * (-1.0) / m      
    return cost