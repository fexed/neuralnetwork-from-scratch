from architecture import Architecture
from datasets import CUP
from folding import Holdout, KFold
from grid_search import GridSearch
from hyperparameter import BatchSize, Epochs, HyperParameter, LearningRate, Momentum, NesterovMomentum
from metrics import MeanEuclideanError 
from mlp import MLP
from losses import MEE, MSE
from regularizators import L2, Regularization
from activationfunctions import Identity, ReLU, Sigmoid, Tanh
from search_space import SearchSpace
from weight_initialization import NormalizedXavier, RandomUniform

cup = CUP(internal_split=True)

MIN_LAYERS, MAX_LAYERS = 2, 6
MIN_UNITS, MAX_UNITS = 10, 100
UNITS_INCR = 30

units_space = []
for L in range(MIN_LAYERS, MAX_LAYERS + 1):
    for u in range(MIN_UNITS, MAX_UNITS + 1, UNITS_INCR):
        units_space.append([u] * L)

architecture_space = Architecture(MLP).search_space(
    io_sizes= (cup.input_size, cup.output_size),
    loss=MSE(),
    hidden_units= [[20, 50, 20]],
    activation=[[Tanh()]],
    initialization=[[RandomUniform()]],
    last_activation=Identity()
)

hyperparameter_space = SearchSpace([
    Epochs.search_space([400]),
    LearningRate.search_space([0.0001 ]), #0.001 ,0.005, 0.01
    BatchSize.search_space([256]), #32, 64, 128, 
    #*[Momentum.search_space([0, 0.01, 0.001]), NesterovMomentum.search_space([0.01, 0.001])],
    Regularization.search_space(L2, [0, 0.0001, ]) #0.005
])

gs = GridSearch("MIRACLE", cup, MLP, verbose=True).set_space(architecture_space, hyperparameter_space)
gs.start(metric=MeanEuclideanError(), folding_strategy=KFold(4))
gs.top_results(2)