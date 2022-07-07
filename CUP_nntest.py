from activationfunctions import Sigmoid
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MSE
from mlp import MLP
from metrics import Accuracy
from regularizators import Thrun, L2
from datasets import CUP

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP()

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()

input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 30, output_size], 
    activations = [Sigmoid()], 
    loss = MSE(), 
    initializations = ["xavier"]
)
  
hyperparameters = [
    Epochs(200),
    LearningRate(0.005),
    BatchSize(1000),
]

model = MLP("CUP_hodlout", architecture, hyperparameters)
model.train(X_TR[0:1000], Y_TR[0:1000], X_TR[1000:-1], Y_TR[1000:-1], metric = Accuracy(), verbose=True)

#model.evaluate(X_TS, Y_TS)
model.results()

