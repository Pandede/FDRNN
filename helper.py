import numpy as np


class AverageMeter:
    def __init__(self, epoch=0, iteration=0):
        '''
        Record the metrics while the neural network is training, e.g. loss, accuracy
        
        @Params:
        epoch (int): Number of epoch
        iteration (int): Number of batch, should be ceil(total/batch_num)
        
        @Return:
        None
        '''
        assert isinstance(epoch, int), 'Epoch must be integer'
        assert isinstance(iteration, int), 'Iteration must be integer'

        self.epoch = epoch
        self.iteration = iteration

        if epoch > 0 and iteration > 0:
            self.__history = np.zeros((epoch, iteration))
        else:
            print('Appending mode activated')
            self.__history = [[]]

        self.e_counter = 0
        self.i_counter = 0

    @property
    def history(self):
        if isinstance(self.__history, list):
            return np.array(self.__history[:self.e_counter])
        return self.__history

    def append(self, value):
        '''
        Append the metrics each iteration
        
        @Params:
        value: value of metric
        
        @Return:
        None
        '''
        if self.epoch > 0 and self.iteration > 0:
            assert self.e_counter < self.epoch, \
                'Too many epoch, index out of bound'
            assert self.i_counter < self.iteration, \
                'Too many iteration, index out of bound'
            self.__history[self.e_counter, self.i_counter] = value
        else:
            self.__history[self.e_counter].append(value)
        self.i_counter += 1

    def step(self):
        '''
        Append the iterations each epoch
        '''
        if self.epoch == 0 and self.iteration == 0:
            self.__history.append([])
        self.e_counter += 1
        self.i_counter = 0

    def extend(self, am):
        '''
        Concatenate the history of AverageMeter
        am (AverageMeter): Another AverageMeter object
        '''
        self.__history = np.concatenate((self.history, am.history), axis=0)

    def get_average(self, indices=None):
        '''
        @Return:
        the mean of epochs in history
        '''
        if indices is None:
            return np.mean(self.__history[:self.e_counter], axis=1)
        if indices == -1:
            return np.mean(self.__history[self.e_counter, :self.i_counter])
        if isinstance(indices, (tuple, list, int)):
            return np.mean(self.__history[indices])
        raise IndexError('Unknown indices, expected tuple, list or int')
