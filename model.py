from time import time
from metrics import Metric 
import pickle
import os

class Model(): 
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger

        self.create_model_folder()

        self.evaluated = False

    def predict(self, _): 
        pass

    def evaluate(self, X, Y, loss = None, metric: Metric = None): 
        output = self.predict(X)

        self.ts_loss = loss.compute(output, Y)
        self.ts_metric = metric.compute(output, Y)

        self.evaluated = True
        self.save()
        

    def results(self):
        print("")
        print(f"TR_LOSS:_{self.tr_loss}")
        print(f"VAL_LOSS:_{self.val_loss}")
        
        print(f"TR_METRIC_{self.tr_metric}")
        print(f"VAL_METRIC:_{self.val_metric}")

        if self.evaluated:
            print(f"TS_LOSS:_{self.ts_loss}")
            print(f"TS_METRIC:_{self.ts_metric}")

    def save(self, model_type):
        filename = f'{self.path}logs/{model_type}.pkl'

        with open(filename, "wb") as savefile:
            pickle.dump(self.__dict__, savefile)


    def load(self, filename):
        """ Loads the neural network from a pickle """

        with open(filename, "rb") as savefile:
            newmodel = pickle.load(savefile)

        self.__dict__.clear()  # clear current net
        self.__dict__.update(newmodel)


    def create_model_folder(self, overwrite = False): 
        suffix = time() if not overwrite else ''
        self.path = f'_MODELS/{self.name}/{suffix}/'

        os.makedirs(f'{self.path}/plots')
        os.makedirs(f"{self.path}/logs" )