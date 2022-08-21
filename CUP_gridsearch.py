from architecture import Architecture
from datasets import CUP
from folding import KFold
from grid_search import GridSearch
from hyperparameter import BatchSize, Epochs, LearningRate, Momentum, EarlyStopping, RandomizedMomentum
from metrics import MeanEuclideanError 
from mlp import MLP
from losses import MSE
from regularizators import L2, Regularization
from activationfunctions import Identity, Sigmoid, Tanh
from search_space import SearchSpace
from weight_initialization import He, Xavier

cup = CUP(internal_split=True)

#MIN_LAYERS, MAX_LAYERS = 2, 3
MIN_UNITS, MAX_UNITS = 35, 45
UNITS_INCR = 5

units_space = []
for u in range(MIN_UNITS, MAX_UNITS + 1, UNITS_INCR):
    units_space.append([u] * 2)

architecture_space = Architecture(MLP).search_space(
    io_sizes= (cup.input_size, cup.output_size),
    loss=MSE(),
    hidden_units=units_space,
    activation=[[Tanh()]],
    initialization=[[He()]],
    last_activation=Identity()
)

hyperparameter_space = SearchSpace([
    Epochs.search_space([800]),
    LearningRate.search_space([0.001, 0.0005, 0.00025]),
    BatchSize.search_space([ 128 ]), 
    RandomizedMomentum.search_space([0.0005, 0.00075, 0.00025 ]),
    Regularization.search_space(L2, [0.000075, 0.00005, 0.000025]),
    EarlyStopping.search_space([50])
])

gs = GridSearch("ANXIETY", cup, MLP, verbose=True).set_space(architecture_space, hyperparameter_space)
gs.start(metric=MeanEuclideanError(), folding_strategy=KFold(4), plots=False)
gs.top_results(200)