from activationfunctions import Identity, Sigmoid
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MSE
from mlp import MLP
from utils import shuffle
from metrics import MeanEuclideanError, MeanSquaredError
from regularizators import Thrun, L2
from datasets import CUP

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP()

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()

#X_TR, Y_TR = shuffle(X_TR, X_TR)
#This line causes a bizarre runtime error...

input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 30, output_size], 
    activations = [Sigmoid(), Identity()], 
    loss = MSE(), 
    initializations = ["xavier"]
)
  
hyperparameters = [
    Epochs(200),
    LearningRate(0.005),
    BatchSize(50),
]

model = MLP("CUP_hodlout", architecture, hyperparameters)
model.train(X_TR[0:1000], Y_TR[0:1000], X_TR[1000:-1], Y_TR[1000:-1], metric = MeanSquaredError(), verbose=True)

#model.evaluate(X_TS, Y_TS)
model.results()

