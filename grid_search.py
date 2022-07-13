from datasets import Dataset
from model import Model
from time import time
import os

class Range(): 
    def __init__(self, searchable_type, config): 
        return iter(searchable_type.range(**config)) #@JUST an intuition, Need to be implemented

class Constraints():
    def __init__(self):
        return None

class SearchSpace(): 
    def __init__(self, model_type, search_ranges):
        self.model_type = model_type
        self.search_ranges = search_ranges

        self.cardinality = 1

        self.iterable = []
        self.fixed = []

        for range in self.search_ranges: 
            self.cardinality *= len(range)
            if len(range) > 1: 
                self.iterable.append(range) 
            else: 
                self.fixed.append(range)


    def varying_parameters(self): 
        return len(self.iterable)


    def __set_MLP_setup__():
        return None


class GridSearch(): 
    def __init__(self, dataset: Dataset, model: type, search_space: SearchSpace, metrics):
        self.dataset = dataset
        self.model = self.model
        self.search_space = search_space

        #suffix = time() if not overwrite else ''
        self.path = f'_GRID_SEARCHCES/{self.model.name}/'
        
        self.create_search_folders()


    def preview(self):    
        return


    def start(self):
        return 
    

    def create_search_folders(self):
        if not os.path.exists(self.path):
            os.makedirs(f'{self.path}/plots')
            os.makedirs(f"{self.path}/logs" )