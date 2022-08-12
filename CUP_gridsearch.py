from architecture import Architecture
from datasets import CUP
from folding import KFold
from grid_search import GridSearch
from hyperparameter import BatchSize, Epochs, LearningRate, Momentum, NesterovMomentum, RandomizedMomentum
from metrics import MeanEuclideanError 
from mlp import MLP
from losses import MSE
from regularizators import L2, Regularization
from activationfunctions import Identity, Sigmoid
from search_space import SearchSpace
from weight_initialization import  He, Xavier

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
    hidden_units= units_space,
    activation=[[Sigmoid()]],
    initialization=[[Xavier()], [He()]],
    last_activation=Identity()
)

hyperparameter_space = SearchSpace([
    Epochs.search_space([600]),
    LearningRate.search_space([0.00001, 0.0001, 0.001 ]),
    BatchSize.search_space([64 ,128, 256]), 
    RandomizedMomentum.search_space([0, 0.001, 0.0001]),
    Regularization.search_space(L2, [ 0.00001,  0.0001, 0.001])
])

gs = GridSearch("MIRACLE", cup, MLP, verbose=True).set_space(architecture_space, hyperparameter_space)
gs.start(metric=MeanEuclideanError(), folding_strategy=KFold(4), plots=False)
gs.top_results(200)