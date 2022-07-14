from activationfunctions import Identity, LeakyReLU, ReLU, Sigmoid, Tanh
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MEE, MSE
from mlp import MLP
from metrics import MeanEuclideanError, MeanSquaredError
from regularizators import Thrun, L2
from datasets import CUP
from weight_initialization import He, Xavier

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP()

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()

#X_TR, Y_TR = shuffle(X_TR, X_TR)
#This line causes a bizarre runtime error...

input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 100, output_size], 
    activations = [Sigmoid(), Identity()], 
    loss = MSE(), 
    initializations = [Xavier()]
)
  
hyperparameters = [
    Epochs(200),
    LearningRate(0.0001),
    BatchSize(50),
]

model = MLP("CUP_hodlout", architecture, hyperparameters)
model.train(X_TR[0:1000], Y_TR[0:1000], X_TR[1000:-1], Y_TR[1000:-1], metric = MEE(), verbose=True)

#model.evaluate(X_TS, Y_TS)
model.results()

