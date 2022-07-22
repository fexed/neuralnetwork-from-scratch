from activationfunctions import Identity, Sigmoid, Tanh
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MEE, MSE
from mlp import MLP
from metrics import MeanSquaredError
from regularizators import L2
from datasets import CUP
from utils import shuffle
from weight_initialization import He

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP()

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
X_TR, Y_TR = shuffle(X_TR, Y_TR)


input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 20, 50, 20, output_size], 
    activations = [Tanh(), Sigmoid(), Tanh(), Identity()], 
    loss = MEE(), 
    initializations = [He()]
)
  
hyperparameters = [
    Epochs(400),
    LearningRate(0.0001),
    BatchSize(300),
    L2(0.0005),
    #Momentum(0.0005),
    #Dropout(0.8)
]

model = MLP("CUP_hodlout", architecture, hyperparameters)
model.train(X_TR[0:1200], Y_TR[0:1200], X_TR[1200:-1], Y_TR[1200:-1], metric = MeanSquaredError(), verbose=True)
