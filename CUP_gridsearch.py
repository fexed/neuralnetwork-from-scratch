from architecture import Architecture
from datasets import CUP
from folding import Holdout, KFold
from grid_search import GridSearch
from hyperparameter import BatchSize, Epochs, HyperParameter, LearningRate, Momentum
from metrics import MeanEuclideanError 
from mlp import MLP
from losses import MEE, MSE
from activationfunctions import Identity, Sigmoid, Tanh
from search_space import SearchSpace
from weight_initialization import NormalizedXavier, Xavier, He

cup = CUP()

MIN_LAYERS, MAX_LAYERS = 3, 3
MIN_UNITS, MAX_UNITS = 20, 50
UNITS_INCR = 10

units_space = []
activation_space = [[Sigmoid()]]
initialization_space = [[Xavier()]]
for L in range(MIN_LAYERS, MAX_LAYERS + 1):
    for u in range(MIN_UNITS, MAX_UNITS + 1, UNITS_INCR):
        units_space.append([u] * L)

architecture_space = Architecture(MLP).search_space(
    io_sizes= (cup.input_size, cup.output_size),
    loss=MSE(),
    hidden_units=[ [60, 50, 35], [60, 50, 35], [60, 50, 35], [60, 50, 35] ],
    activation=[[Tanh(), Tanh(), Tanh(), Identity()] ],
    initialization=[[Xavier(), He(), NormalizedXavier(), Xavier() ]],
    last_activation=Identity()
)

hyperparameter_space = SearchSpace([
    Epochs.search_space([100]),
    LearningRate.search_space([0.00001]),
    BatchSize.search_space([175])
])

gs = GridSearch("MIRACLE", cup, MLP, verbose=True).set_space(architecture_space, hyperparameter_space)
gs.start(metric=MeanEuclideanError(), folding_strategy=KFold(2,1))
gs.top_results(2)