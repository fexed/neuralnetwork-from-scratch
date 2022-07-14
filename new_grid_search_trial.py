from architecture import Architecture
from datasets import CUP
from hyperparameter import BatchSize, Epochs, HyperParameter, LearningRate, Momentum 
from mlp import MLP
from losses import MSE
from activationfunctions import Identity, Tanh
from search_space import SearchSpace
from weight_initialization import Xavier, He

cup = CUP()

architecture_search_space = Architecture(MLP).search_space(
    io_sizes= (cup.input_size, cup.output_size),
    loss=MSE(),
    hidden_units=[ [20, 10], [20, 10, 5]],
    activation=[Tanh()],
    initialization=[Xavier(), He()],
    last_activation=Identity()
)


hy_space = SearchSpace([
    Epochs.search_space([2, 3]),
    LearningRate.search_space( [0.01, 0.05] ),
    BatchSize.search_space( [ 1, cup.tr_size/8 ,cup.tr_size ])
])

print(hy_space.cardinality)


