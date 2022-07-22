from architecture import Architecture
from datasets import CUP
from folding import Holdout, KFold
from grid_search import GridSearch
from hyperparameter import BatchSize, Epochs, HyperParameter, LearningRate, Momentum 
from mlp import MLP
from losses import MEE, MSE
from activationfunctions import Identity, Sigmoid, Tanh
from search_space import SearchSpace
from weight_initialization import NormalizedXavier, Xavier, He

cup = CUP()

architecture_space = Architecture(MLP).search_space(
    io_sizes= (cup.input_size, cup.output_size),
    loss=MSE(),
    hidden_units=[ [20, 50, 20] ],
    activation=[[Tanh(), Sigmoid(), Tanh(), Identity()], [Tanh(), Tanh(), Tanh(), Identity()]],
    initialization=[[Xavier(), He(), NormalizedXavier(), Xavier() ]],
    #last_activation=Identity()
)

hyperparameter_space = SearchSpace([
    Epochs.search_space([100]),
    LearningRate.search_space([0.00001]),
    BatchSize.search_space([175])
])

gs = GridSearch("MIRACLE", cup, MLP, verbose=True).set_space(architecture_space, hyperparameter_space)
gs.start(metric=MEE(), folding_strategy=KFold(7, 2))