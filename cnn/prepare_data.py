#encode=utf-8

import os
import gzip
import cPickle
import numpy as np

class Dataset:
    def __init__(self, X, Y):
        self.X = X.T
        self.Y = Y.T
        
    def data_size(self):
        return len(self.X)
    
class MinistDataset:
    def __init__(self):
        self._input_shape = (784)
        self._result_shape = (10)
        self._load_data_wrapper()
        
    def _load_data(self):
        data_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/mnist.pkl.gz'
        f = gzip.open(data_path, 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)
    
    def _vectorized_result(self, j):
        e = np.zeros(self._result_shape)
        e[j] = 1
        return e
    
    def _load_data_wrapper(self):
        tr_d, va_d, te_d = self._load_data()
        training_inputs = [np.reshape(x, self._input_shape) for x in tr_d[0]]
        training_results = [self._vectorized_result(y) for y in tr_d[1]]
        self.training_data = Dataset(np.array(training_inputs), np.array(training_results))
        validation_inputs = [np.reshape(x, self._input_shape) for x in va_d[0]]
        self.validation_data = Dataset(np.array(validation_inputs), np.array(va_d[1]))
        test_inputs = [np.reshape(x, self._input_shape) for x in te_d[0]]
        self.test_data = Dataset(np.array(test_inputs), np.array(te_d[1]))
        
class ReaderWrapper:
    def __init__(self, batch_size=128):
        self._batch_size = batch_size
        self._dataset = MinistDataset()
        
    def get_training_data(self):
        (_,m) = np.shape(self._dataset.training_data.X)
        training_data_X = self._dataset.training_data.X.reshape(m,28,28,1)
        return training_data_X, self._dataset.training_data.Y
    
    def get_ong_data(self):
        training_data_X = self._dataset.training_data.X[0].reshape(1,28,28,1)
        return training_data_X, self._dataset.training_data.Y[0]
        
    def get_validation_data(self):
        (_,m) = np.shape(self._dataset.validation_data.X)
        validation_data_X = self._dataset.validation_data.X.reshape(m,28,28,1)
        return validation_data_X, self._dataset.validation_data.Y
    
    def get_test_data(self):
        (_,m) = np.shape(self._dataset.test_data.X)
        test_data_X = self._dataset.test_data.X.reshape(m,28,28,1)
        return test_data_X, self._dataset.test_data.Y

if __name__ == '__main__':
    reader = ReaderWrapper()
    indexs = reader.get_mini_batchs()
    print indexs[0][0]
    