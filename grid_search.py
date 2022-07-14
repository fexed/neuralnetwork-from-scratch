
from datasets import Dataset
from model import Model
from logger import GridSearchLogger
from mlp import MLP
from time import time
import os

class GridSearch(): 
    def __init__(self, dataset: Dataset, model_type: type[Model], verbose=True):
        self.dataset = dataset
        self.model_type = self.model_type
        self.verbose = verbose

        if(model_type == MLP): 
            self.set_space = self.__init_MLP_search_space__

        #suffix = time() if not overwrite else ''
        self.path = f'_GRID_SEARCHCES/{self.dataset.name}_{self.model_type}/'
        
        self.create_search_folders()

        self.logger = GridSearchLogger()

 
    def preview(self):    
        return


    def start(self):
        return 
    

    def create_search_folders(self):
        if not os.path.exists(self.path):
            os.makedirs(f'{self.path}/plots')
            os.makedirs(f"{self.path}/logs" )
    

    def __init_MLP_search_space__(self, architecture_space, hyperparameter_space, metric):
        models = []
        for architecture in architecture_space: 
            for hyperparameters in hyperparameter_space: 
                models.append(MLP(architecture, hyperparameters, verbose=self.verbose, make_folder=False))
