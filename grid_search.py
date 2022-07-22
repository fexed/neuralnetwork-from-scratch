import numpy as np
import pickle
from datasets import Dataset
from folding import FoldingStrategy, Holdout
from losses import MEE
from model import Model
from logger import GridSearchLogger
from mlp import MLP
from time import time
import os

class GridSearch(): 
    def __init__(self, name, dataset: Dataset, model_type: type[Model], verbose=True):
        self.name = name
        self.dataset = dataset
        self.model_type = model_type
        self.verbose = verbose

        if(model_type == MLP): 
            self.set_space = self.__init_MLP_search_space__

        suffix = time() 
        self.path = f'_GRID_SEARCHES/{self.name}_{self.dataset.name}/{suffix}/'


    def create_model_folders(self, suffix):
        model_path = f'{self.path}/{suffix}'
        if not os.path.exists(model_path):
            os.makedirs(f'{model_path}/plots')
            os.makedirs(f'{model_path}/logs')  
        return model_path


    def start(self, metric = MEE(), folding_strategy: FoldingStrategy = Holdout(0.2), repetitions = 1): 
        #TODO Implement gruid search logger.
        # self.logger.preview() Print a preview of the starting grid seach.
        self.results = []

        folding_cycles = folding_strategy(*self.dataset.getTR(), shuffle=True)

        for f, fc in enumerate(folding_cycles):
            X_TR, Y_TR, X_VAL, Y_VAL = fc
            fold_result = []

            for i, model in enumerate(self.models):
                model_path = self.create_model_folders(f'{i}_{f}')

                model.train(X_TR, Y_TR ,X_VAL, Y_VAL, metric , plot_folder = model_path + '/')
                model.save(model_path)
                fold_result.append(model.val_metric)

            self.results.append(fold_result)
            
        self.searched = True
        #@TODO Plot result matrix somewhere


    def top_results(self, n):
        indexes = np.argpartition(np.array(self.results), -n)[-n:]

        print("Best models:")
        for i, ind in enumerate(zip(indexes)): 
            print(f"{i+1}): Index: {ind}, Model: {self.results[ind]}")
        

    def save(self):
        filename = f'{self.path}results_{self.name}.pkl'

        with open(filename, "wb") as savefile:
            pickle.dump(self.results, savefile)


    def __init_MLP_search_space__(self, architecture_space, hyperparameter_space):
        self.models = []
        model_idx = 0
        for architecture in architecture_space: 
            #print(architecture)
            for hyperparameters in hyperparameter_space: 
                #@TODO Apply constraints here if necesary.
                self.models.append(MLP(f'_{model_idx}', architecture, hyperparameters, verbose=self.verbose, make_folder=False))
                model_idx += 1
        
        return self
