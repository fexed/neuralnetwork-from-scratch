from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_loss
import numpy as np
import matplotlib.pyplot as plot
from preprocessing import one_hot_encoding


monk = 2
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# Training

# training set loading
monkfile = open("datasets/MONK/monks-" + str(monk) + ".train", "r")
xtr = []
ytr = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    ytr.append([[int(vals[1])]])
X = np.array(xtr)
Y = np.array(ytr)

# preprocessing
X, input_size = one_hot_encoding(X)

# training
net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime)
net.add(FullyConnectedLayer(input_size, 10, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(10, 10, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(10, 1, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
net.summary()
history = net.training_loop(X, Y, epochs=1000, learning_rate=0.05, verbose=True, early_stopping=50)

# Model evaluation

# test set loading
monkfile = open("datasets/MONK/monks-" + str(monk) + ".test", "r")
xts = []
yts = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xts.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    yts.append([[int(vals[1])]])
X = np.array(xts)
Y = np.array(yts)

# preprocessing
X, input_size = one_hot_encoding(X)

# evaluating
out = net.predict(X)
accuracy = 0
for i in range(len(out)):
    val = 0 if out[i].item() < 0.5 else 1
    if (Y[i].item() == val): accuracy += 1
accuracy /= len(out)
accuracy *= 100
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
plot_loss(title="MONK2 model evaluation", history=history, ylabel="Loss", xlabel="Epochs", savefile="MONK1TEST")

# saving the net
net.savenet("models/MONK2TESTED_1L_10U_0.05LR_normxavier.pkl")
