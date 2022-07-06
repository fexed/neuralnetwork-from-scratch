from abc import abstractmethod
from time import time
import os

class Model(): 
    def __init__(self, name, ):
        self.name = name

    def create_model_folder(self, overwrite = False): 
        suffix = time() if not overwrite else ''
        self.path = f'_MODELS/{self.name}/{suffix}/'

        os.makedirs(f'{self.path}/plots')
        os.makedirs(f"{self.path}/logs" )

    @abstractmethod
    def predict(self, X): 
        pass

    @abstractmethod
    def export(self):
        pass

    def evaluate(self, X_TS, Y_TS, metric): 
        pass
    