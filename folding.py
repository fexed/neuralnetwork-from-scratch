from typing import Iterator
from utils import shuffle as _shuffle
import numpy as np

class FoldingStrategy(): 
    def __init__(): 
        return

class FoldingCycle(): 
    def __next__():
        return

class Holdout(FoldingStrategy): 
    def __init__(self,  val_size=0, ):
        if val_size >= 1:
            raise Exception("Wrong splitting coefficient")
        
        self.val_size = val_size


    def __call__(self, X, Y, shuffle = True):
        if shuffle: 
            X, Y = _shuffle(X,Y)

        split = int(len(X) * (1 - self.val_size))

        self.X_TR = X[0:split]
        self.Y_TR = Y[0:split]
        self.X_VAL = X[split:-1] if split else []
        self.Y_VAL = Y[split:-1] if split else []

        return self


    def __iter__(self) -> Iterator[FoldingCycle]:  
        return iter([(self.X_TR, self.Y_TR, self.X_VAL, self.Y_VAL)])


    def data(self):
        return self[0]


# This only supports CV, nested one is not implemented yet.
class KFold(FoldingStrategy):
    """ KFold CV data structure to directly iterate on """

    def __init__(self, k, val_size = 1):
        self.k = k
        self.val_size = val_size


    def __call__(self, X, Y, shuffle = True):
        if shuffle: 
            X, Y = _shuffle(X,Y)

        n = len(X)//self.k

        self.folds = [ ]

        for i in range(self.k): 
            self.folds.append([X[i*n: (i+1)*n], Y[i*n: (i+1)*n]]) 

        return self


    def __iter__(self): 
        return FoldIterator(self.folds, self.k, self.val_size,)


class FoldIterator():
    def __init__(self, folds, k, val_size): 
        self.i = 0
        self.k = k

        self.folds = folds

        self.val_size = val_size
        self.tr_size = len(folds) - val_size


    def _split_folds(self, start, elems):
        X_TR = []
        Y_TR = []

        X_VAL = []
        Y_VAL = []
        
        end = start + elems
        for i in range(len(self.folds)): 
            if i < end - self.k or ( i >= start and i < end ) : 
                X_VAL.append(self.folds[i][0])
                Y_VAL.append(self.folds[i][1])
            else:
                X_TR.append(self.folds[i][0])
                Y_TR.append(self.folds[i][1])
        
        return np.concatenate(X_TR), np.concatenate(Y_TR), np.concatenate(X_VAL), np.concatenate(Y_VAL)


    def __next__(self) -> FoldingCycle: 
        if self.i >= self.k: 
            raise StopIteration
        else:
            self.i += 1
            return self._split_folds(self.i, self.val_size)
