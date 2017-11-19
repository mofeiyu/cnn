import numpy as np 

def mini_batch_data(X, Y, batch_size):
    mini_batchs = []
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    #print m, permutation[:10]
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[:, permutation].reshape((10,m))
    batch_num = m / batch_size + (m % batch_size > 0) 
    for j in range(batch_num):
        if j != batch_num-1:
            Xj,Yj = shuffled_X[j*batch_size:(j+1)*batch_size,:,:,:],shuffled_Y[:,j*batch_size:(j+1)*batch_size]
        else :
            Xj,Yj = shuffled_X[j*batch_size:,:,:,:],shuffled_Y[:,j*batch_size:]
        mini_batchs.append((Xj, Yj))
    return mini_batchs