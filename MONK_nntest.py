from activationfunctions import Identity, Sigmoid
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MSE
from mlp import MLP
from metrics import Accuracy
from regularizators import Thrun, L2
from datasets import Monk
from weight_initialization import RandomUniform

monk = 3
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

MONK = Monk(monk)

X_TR, Y_TR, X_TS, Y_TS = MONK.getAll(one_hot=True)
input_size, output_size = MONK.size()

architecture = Architecture(MLP).define(
    units= [input_size, 4, output_size], 
    activations = [Sigmoid()], 
    loss = MSE(), 
    initializations = [RandomUniform()]
)
  
hyperparameters = [
    Epochs(200),
    LearningRate(0.03),
    BatchSize(len(X_TR)),
    L2(0.005)
]

model = MLP("MONK" + str(monk), architecture, hyperparameters)
model.train(X_TR, Y_TR, X_TS, Y_TS, metric = Accuracy(), verbose=True)

model.evaluate(X_TS, Y_TS, loss= MSE(), metric = Accuracy())