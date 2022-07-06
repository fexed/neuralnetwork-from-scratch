from platform import architecture
from activationfunctions import Sigmoid
from architecture import Architecture
from hyperparameter import BatchSize, Epochs, LearningRate
from losses import MSE
from mlp import MLP
from metrics import Accuracy
from dataset_loader import load_monk
from regularizators import Thrun, L2

monk = 3
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

architecture = Architecture(MLP).define(
    units= [input_size, 4, 1], 
    activations = [Sigmoid()], 
    loss = MSE(), 
    initializations = ["xavier"]
)
  
hyperparameters = [
    Epochs(200),
    LearningRate(0.03),
    BatchSize(len(X_TR)),
    L2(0.005)
]

model = MLP("MONK" + str(monk), architecture, hyperparameters)
model.train(X_TR, Y_TR, X_TS, Y_TS, metric = Accuracy(), verbose=True)

model.evaluate(X_TS, Y_TS)
model.results()
