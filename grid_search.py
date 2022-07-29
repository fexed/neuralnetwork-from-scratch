import numpy as np
import pickle
from datasets import Dataset
from folding import FoldingStrategy, Holdout
from metrics import MeanEuclideanError, Metric
from model import Model
from logger import GridSearchLogger
from mlp import MLP
from time import time
import os

 
class GridSearch(): 
    def __init__(self, name, dataset: Dataset, model_type, verbose=True):
        self.name = name
        self.verbose = verbose
        self.dataset = dataset
        self.model_type = model_type

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


    def start(self, metric: Metric = MeanEuclideanError(), folding_strategy: FoldingStrategy = Holdout(0.2)): 
        self.logger.search_preview(folding_strategy) 

        self.metric = metric
        self.results = []

        folding_cycles = folding_strategy(*self.dataset.getTR(), shuffle=True)
        
        for i, model in enumerate(self.models):
            fold_result = []
            for f, fc in enumerate(folding_cycles):
                model_path = self.create_model_folders(f'{i}_{f}')
                
                model.train(*fc, metric , plot_folder = model_path + '/')
                model.save(model_path)

                fold_result.append(model.val_metric)

            self.results.append([i, np.mean(fold_result), np.std(fold_result)])
            
        self.searched = True
        self.logger.end_message()
        self.save_result_matrix()
       


    def save_result_matrix(self, format='txt', matrix = None):
        mat = np.matrix(self.results if matrix is None else matrix)
        with open(f'{self.path}/RESULTS.{format}','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
    

    def top_results(self, n, save = True):
        best_models = []
        
        indexes = np.array(self.results)[:, 1].argsort()
        ordered_indexes = indexes if self.metric.is_loss else reversed(indexes)

        if save: 
            text_file = open(f'{self.path}BEST_MODELS.txt', "w")
            text_file.write(f"Best models according to {self.metric}\n")

        for i in range(n): 
            j = ordered_indexes[i]
            row = self.results[j]
            i_model = self.logger.top_result_line(i+1, j, row[1], row[2])
            best_models.append(i_model )
            
            if save:
                text_file.write(f"{i_model}\n")
        
        if save:     
            text_file.close()

        return best_models


    def __init_MLP_search_space__(self, architecture_space, hyperparameter_space):
        self.models = []
        model_idx = 0

        for architecture in architecture_space: 
            for hyperparameters in hyperparameter_space: 
                self.models.append(MLP(f'{model_idx}_', architecture, hyperparameters, verbose=self.verbose, make_folder=False))
                model_idx += 1

        self.logger = GridSearchLogger(self.name, self.dataset.name, self.dataset.cardinality(), self.model_type, len(self.models), self.verbose)

        return self
