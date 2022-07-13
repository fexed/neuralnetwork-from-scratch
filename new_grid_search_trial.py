from architecture import Architecture 
from mlp import MLP
from losses import MSE
from activationfunctions import Identity, Tanh
from weight_initialization import Xavier, He

search_space = Architecture(MLP).range(
    io_sizes= (10, 2),
    loss=MSE(),
    hidden_units=[ [20, 10], [20, 10, 5]],
    activation=[Tanh()],
    initialization=[Xavier(), He()],
    last_activation=Identity()
)

print(search_space)