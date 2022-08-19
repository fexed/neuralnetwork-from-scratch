from activationfunctions import Identity, Sigmoid
from architecture import Architecture
from hyperparameter import BatchSize, EarlyStopping, Epochs, LearningRate, Momentum
from losses import MSE, BinaryCrossentropy
from mlp import MLP
from metrics import Accuracy
from regularizators import Thrun, L2
from datasets import Monk
from weight_initialization import RandomUniform
import numpy as np

monk = 3
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

MONK = Monk(monk)

X_TR, Y_TR, X_TS, Y_TS = MONK.getAll(one_hot=True)
input_size, output_size = MONK.size()

architecture = Architecture(MLP).define(
    units= [input_size, 3, output_size], 
    activations = [Sigmoid()], 
    loss = BinaryCrossentropy(), 
    initializations = [RandomUniform()]
)
  
hyperparameters = [
    Epochs(400),
    LearningRate(0.025),
    BatchSize(len(X_TR)),
    Momentum(0.4),
    L2(0.002)
]

res = []
for _ in range(10): 
    model = MLP("MONK" + str(monk), architecture, hyperparameters)
    hist = model.train(X_TR, Y_TR, X_TS, Y_TS, metric = Accuracy(), second_metric=MSE(), verbose=True, plot_folder='')

    model.evaluate(X_TS, Y_TS, loss= MSE(), metric = Accuracy())
    res.append(np.array(hist)[:, -1])

res = np.array(res)
print(res.shape)
print(np.mean(res, axis=0), np.std(res, axis=0))

    