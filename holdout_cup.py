from random import random
from activationfunctions import tanh, tanh_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_loss
import numpy as np
import matplotlib.pyplot as plot
from preprocessing import continuous_standardizer, min_max_normalizer
from dataset_loader import load_cup
from sklearn.model_selection import train_test_split

print("\n\n****TESTING NETWORK ON CUP" )


def compare(a, b, tollerance=1e-03):
    return abs(a - b) <= tollerance * max(abs(a), abs(b))

# Training

# training set loading + 
X, Y = load_cup()

# preprocessing
X, n_min, n_max = min_max_normalizer(X)
X, means, std = continuous_standardizer(X)
 

X_TR,  X_VAL, Y_TR, Y_VAL = train_test_split(X, Y, test_size=0.2, random_state=18) 

# training
net = Network("CUP", MSE, MSE_prime)
net.add(FullyConnectedLayer(10, 25, tanh, tanh_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, tanh, tanh_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, tanh, tanh_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(25, 25, tanh, tanh_prime, initialization_func="normalized_xavier"))
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
plot_loss(title="CUP model evaluation", history=history, validation_history=validation_history, ylabel="Loss", xlabel="Epochs", savefile="CUP_HOLDOUT")

