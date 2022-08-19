from activationfunctions import Identity, Sigmoid, Tanh
from architecture import Architecture
from hyperparameter import BatchSize, EarlyStopping, Epochs, LearningRate, Momentum, NesterovMomentum
from losses import MEE, MSE
from mlp import MLP
from metrics import MeanEuclideanError, MeanSquaredError
from regularizators import L2
from datasets import CUP
from utils import shuffle
from weight_initialization import Xavier, He, NormalizedXavier

print("\n\n****TESTING NETWORK ON CUP" )

_CUP = CUP(internal_split=True)

X_TR, Y_TR, X_TS, Y_TS = _CUP.getAll()
X_TR, Y_TR = shuffle(X_TR, Y_TR)


input_size, output_size = _CUP.size()

architecture = Architecture(MLP).define(
    units= [input_size, 40, 40, output_size], 
    activations = [ Tanh(), Tanh(), Identity()], 
    loss = MSE(), 
    initializations = [He() ]
)
  
hyperparameters = [
    Epochs(800),
    LearningRate(0.00005),
    BatchSize(128),
    Momentum(0.0005),
    L2(2.5e-5),
    EarlyStopping(50)
]

model = MLP("CUP_holdout", architecture, hyperparameters)
model.train(X_TR[0:1000], Y_TR[0:1000], X_TR[1000:1200], Y_TR[1000:1200], metric = MeanEuclideanError(), verbose=True, plot_folder='')
model.evaluate(X_TR[1200:-1], Y_TR[1200:-1], loss=MSE(), metric=MeanEuclideanError())
