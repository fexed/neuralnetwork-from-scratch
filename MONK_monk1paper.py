from random import shuffle
from activationfunctions import Sigmoid
from losses import BinaryCrossentropy, MSE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from metrics import Accuracy
from utils import multiline_plot
from dataset_loader import load_monk
from regularizators import L2, Thrun
import numpy as np

monk = 1
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# Training

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# training
net = Network("MONK" + str(monk), MSE(), regularizator=L2(l=0.005))
net.add(FullyConnectedLayer(input_size, 3, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(3, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history, test_history, metric_history, metric_test_history = net.training_loop(X_TR, Y_TR, X_validation=X_TS, Y_validation=Y_TS, epochs=400, learning_rate=0.15, verbose=True, metric=Accuracy())

# evaluating
accuracy = Accuracy().compute(net, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
multiline_plot(title="MONK1 MSE", legend_names=["Training MSE", "Test Set MSE"], histories=[history, test_history], ylabel="MSE", xlabel="Epochs", savefile="MONK1_PAPER_MSE", showlegend=True, showgrid=True, alternateDots=True)
multiline_plot(title="MONK1 Accuracy", legend_names=["Training Accuracy", "Test Set Accuracy"], histories=[metric_history, metric_test_history], ylabel="Accuracy", xlabel="Epochs", savefile="MONK1_PAPER_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)
# saving the net
net.savenet("models/MONK1PAPER.pkl")
