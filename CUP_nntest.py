from activationfunctions import Identity, Sigmoid, Tanh
from architecture import Architecture
from hyperparameter import BatchSize, EarlyStopping, Epochs, LearningRate, Momentum, NesterovMomentum
from losses import MEE, MSE
from mlp import MLP
from metrics import MeanEuclideanError, MeanSquaredError
from regularizators import L2
from datasets import CUP
from utils import shuffle, tr_vl_split
from weight_initialization import Xavier, He, NormalizedXavier
from folding import KFold
import numpy as np

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP(internal_split=True)

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
X_TR, Y_TR = shuffle(X_TR, Y_TR)
X_TR, X_VL, Y_TR, Y_VL = tr_vl_split(X_TR, Y_TR, ratio=0.25)


input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 42, 42, output_size], 
    activations = [Tanh(), Tanh(), Identity()], 
    loss = MSE(), 
    initializations = [He()]
)
  
hyperparameters = [
    Epochs(800),
    LearningRate(0.00005),
    BatchSize(128),
    Momentum(0.001),
    L2(2.5e-5),
    EarlyStopping(50)
]

folding_strategy=KFold(4)
folding_cycles = folding_strategy(X_TR, Y_TR, shuffle=True)
results = []
for f, fc in enumerate(folding_cycles):
    model = MLP("CUP_holdout", architecture, hyperparameters)
    hists = model.train(*fc, metric = MeanEuclideanError(), verbose=True, plot_folder='')
    results.append(model.tr_metric)

print("\n\n****TESTING NETWORK ON CUP" )
print(np.mean(results))
print(np.std(results))
