from time import time
from metrics import Metric 
import os

class Model(): 
    def __init__(self, name, ):
        self.name = name

    def predict(self, X): 
        pass

    def export(self):
        pass

    def evaluate(self, X_TS, Y_TS, metric: Metric): 
        pass

    def create_model_folder(self, overwrite = False): 
        suffix = time() if not overwrite else ''
        self.path = f'_MODELS/{self.name}/{suffix}/'

        os.makedirs(f'{self.path}/plots')
        os.makedirs(f"{self.path}/logs" )