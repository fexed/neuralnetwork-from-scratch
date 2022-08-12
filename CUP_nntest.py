from activationfunctions import Identity, Sigmoid, Tanh
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MEE, MSE
from mlp import MLP
from metrics import MeanEuclideanError, MeanSquaredError
from regularizators import L2
from datasets import CUP
from utils import shuffle
from weight_initialization import Xavier, He, NormalizedXavier

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP()

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
X_TR, Y_TR = shuffle(X_TR, Y_TR)


input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 60, 50, 35, output_size], 
    activations = [Tanh(), Tanh(), Tanh(), Identity()], 
    loss = MEE(), 
    initializations = [Xavier(), He(), NormalizedXavier(), Xavier() ]
)
  
hyperparameters = [
    Epochs(400),
    LearningRate(0.00001),
    BatchSize(175),
    L2(0.0001)
]

model = MLP("CUP_holdout", architecture, hyperparameters)
model.train(X_TR[0:1000], Y_TR[0:1000], X_TR[1000:1200], Y_TR[1000:1200], metric = MeanSquaredError(), verbose=True)
model.evaluate(X_TR[1200:-1], Y_TR[1200:-1], loss=MEE(), metric=MeanSquaredError())
