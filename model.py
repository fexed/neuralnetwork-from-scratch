from time import time
from metrics import Metric 
import pickle
import os

class Model(): 
    def __init__(self, name, logger, description = "", make_folder=True):
        self.name = name
        self.description = description
        self.logger = logger
        
        if(make_folder):
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

        self.logger.test_results(self.ts_loss, self.ts_metric, metric)


    def save(self, model_type, custom_path = None):
        path = self.path if custom_path is None else custom_path
        filename = f'{path}/logs/{model_type}'

        with open(filename + '.pkl', "wb") as savefile:
            pickle.dump(self.__dict__, savefile)

        text_file = open(filename + ".txt", "w")
 
        #write string to file
        text_file.write(self.name)
        text_file.write(self.description)
        
        #close file
        text_file.close()


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


    def __str__(self): 
        return f"Model: {self.name}"

    def reset():
        pass