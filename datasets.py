import numpy as np
from metrics import Task
from preprocessing import one_hot_encoding

class Dataset():
    def __init__(self, name, task = Task.REGRESSION): 
        self.name = name
        self.task = task

        self.X_TR, self.Y_TR = self.readfile(self.train_suffix )
        self.X_TS, self.Y_TS = self.readfile(self.test_suffix)

        self.tr_size = len(self.X_TR)
        self.ts_size = len(self.X_TS)

    def size(self): 
        return (self.input_size, self.output_size)


class Monk(Dataset):
    def __init__(self, n):
        self.path = f'datasets/MONK/monks-{n}.'

        self.train_suffix = 'train'
        self.test_suffix = 'test'

        self.input_size = 6
        self.output_size = 1

        super().__init__(f"MONK{n}", Task.BINARY_CLASSIFICATION)

    def readfile(self, set):
        file = open(self.path + set, 'r')

        x = []
        y = []
        
        for line in file.readlines():
            vals = line.split(" ")

            x.append([list(map(lambda x_i: int(x_i), vals[2:8]))])
            y.append([[int(vals[1])]])

        x = np.array(x)
        y = np.array(y)

        return x, y

    def getAll(self, one_hot = False): 

        if one_hot == True: 
            self.X_TR, self.input_size = one_hot_encoding(self.X_TR)
            self.X_TS, self.input_size = one_hot_encoding(self.X_TS )

        return self.X_TR, self.Y_TR, self.X_TS, self.Y_TS


class CUP(Dataset): 
    def __init__(self):        
        self.path = f'datasets/CUP/ML-CUP21-'

        self.train_suffix = 'TR'
        self.test_suffix = 'TS'

        self.input_size = 10
        self.output_size = 2

        super().__init__("CUP", Task.REGRESSION)

    def readfile(self, set):
        file = open(f'{self.path}{set}.csv', 'r')

        x = []
        y = []
        
        for line in file.readlines():
            if (line.startswith("#")):
                continue
            vals = line.split(",")

            x.append([list(map(lambda x_i: float(x_i), vals[1:11]))])

            if set == self.train_suffix :
                y.append([list(map(lambda y_i: float(y_i), vals[11:13]))])

        x = np.array(x)
        y = np.array(y)

        return x, y

    def getAll(self): 
        return self.X_TR, self.Y_TR, self.X_TS, self.Y_TS

