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

        if not os.path.exists(self.path):
             os.makedirs(f'{self.path}')

    def create_model_folders(self, suffix, plots = False):
        model_path = f'{self.path}/{suffix}'
        if not os.path.exists(model_path):

            if plots: 
                os.makedirs(f'{model_path}/plots')

            os.makedirs(f'{model_path}/logs')

        return model_path


    def start(self, metric: Metric = MeanEuclideanError(), folding_strategy: FoldingStrategy = Holdout(0.2), restart_from = 0, plots=False): 
        self.logger.search_preview(folding_strategy) 

        self.metric = metric
        self.results = []

        folding_cycles = folding_strategy(*self.dataset.getTR(), shuffle=True) 
        os.path.join(self.path, 'RESULTS.txt')
       
        for i, model in enumerate(self.models[restart_from: -1]):
            fold_result = []
            epochs = []
            for f, fc in enumerate(folding_cycles):
                model_path = self.create_model_folders(f'{i+restart_from}_{f}', plots)
                
                hists = model.train(*fc, metric, plot_folder = model_path + '/' if plots else None) #,)
                model.save(model_path)
                
                fold_result[0].append(model.tr_metric)
                fold_result[1].append(model.val_metric)
                
                epochs.append(len(hists[0]))
                model.reset()
            
            self.results.append([i+restart_from, np.mean(fold_result[0]), np.std(fold_result[0]), np.mean(fold_result[1]), np.std(fold_result[1]),np.mean(epochs), np.std(epochs)])

            result_file = open(f'{self.path}RESULTS.txt', "a")
            result_file.write(f"{self.results[i]}\n")
            result_file.close()
            
        self.searched = True
        self.logger.end_message()
        self.save_result_matrix()
       


    def save_result_matrix(self, format='txt', matrix = None):
        mat = np.matrix(self.results if matrix is None else matrix)
        with open(f'{self.path}/FINAL_RESULTS.{format}','wb') as f:
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
