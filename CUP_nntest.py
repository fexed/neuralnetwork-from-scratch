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

input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 42, 42, output_size], 
    activations = [Tanh(), Tanh(), Identity()], 
    loss = MSE(), 
    initializations = [He()]
)
  
hyperparameters = [
    Epochs(220),
    LearningRate(0.0001),
    BatchSize(128),
    Momentum(0.001),
    L2(0.000025)
]

results = []
for i in range(10):
    X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
    X_TR, Y_TR = shuffle(X_TR, Y_TR)
    model = MLP("CUP_finaltest", architecture, hyperparameters)
    hists = model.train(X_TR, Y_TR, metric = MeanEuclideanError(), verbose=True, plot_folder="")
    results.append(model.tr_metric)

print("\n\n****RESULTS" )
print("MEE mean:", np.mean(results))
print("MEE  dev:", np.std(results))


print("\n\n****FINAL CUP RESULTS" )
_CUP = CUP(internal_split=False)
X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
model = MLP("CUP_finaltest", architecture, hyperparameters)
hists = model.train(X_TR, Y_TR, metric = MeanEuclideanError(), verbose=True, plot_folder="")
results = model.predict(X_TS)
output_file = open("datasets/CUP/arbitraryvalues_ML-CUP21-TS.csv", "w")
output_file.write("#Federico Matteoni, Riccardo Parente, Sergio Latrofa\n"+
                  "#arbitraryvalues\n"+
                  "#ML CUP21\n"+
                  "#22/08/2022\n")
count = 1
for data in results:
    output_file.write(str(count)+","+str(data[0][0])+","+str(data[0][1])+"\n")
    count += 1
output_file.close()