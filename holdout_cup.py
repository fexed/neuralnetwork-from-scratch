from random import random
from activationfunctions import Tanh
from losses import MSE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_and_save, tr_vl_split, compare
import numpy as np
import matplotlib.pyplot as plot
from preprocessing import continuous_standardizer, min_max_normalizer
from dataset_loader import load_cup

print("\n\n****TESTING NETWORK ON CUP" )


# Training

# training set loading +
X, Y = load_cup()

# preprocessing
X, n_min, n_max = min_max_normalizer(X)
X, means, std = continuous_standardizer(X)


X_TR,  X_VAL, Y_TR, Y_VAL = tr_vl_split(X, Y, ratio=0.2)

# training
net = Network("CUP", MSE()))
net.add(FullyConnectedLayer(10, 25, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 2,  initialization_func="normalized_xavier"))
net.summary()
history, validation_history = net.training_loop(X_TR, Y_TR ,X_validation=X_VAL, Y_validation=Y_VAL, epochs=150, learning_rate=0.001, verbose=True, batch_size=1)

# Model evaluation


# evaluating
out = net.predict(X_VAL)
accuracy = 0
print(Y_VAL[0], " ", out[0])
for i in range(len(out)):
    # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
    if (compare(Y_VAL[i][0][0], out[i][0][0]) and compare(Y_VAL[i][0][1], out[i][0][1])): accuracy += 1
accuracy /= len(out)
accuracy *= 100

print("\n\nAccuracy on CUP validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

# plotting data
plot_and_save(title="CUP model evaluation", history=history, validation_history=validation_history, ylabel="Loss", xlabel="Epochs", savefile="CUP_HOLDOUT")
