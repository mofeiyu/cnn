import logging

def test_accuracy(AL, Y):
    hit_count = 0
    for j in range(AL.shape[1]):
        max_val = -1e8
        max_index = 0
        for i in range(AL.shape[0]):
            if AL[i][j] > max_val:
                max_val = AL[i][j] 
                max_index = i
        if Y[j] == max_index:
            hit_count += 1
    logging.debug('hit_count=%d, AL.shape[1]=%d, hit_ratio=%.2lf%%',
                 hit_count,
                 AL.shape[1],
                 hit_count * 100. / AL.shape[1]) 
    return hit_count