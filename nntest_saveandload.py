import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import time


monk = 1

ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
print("\n\n****MONK" + str(monk))
monkfile = open("datasets/MONK/monks-" + str(monk) + ".train", "r")
xtr = []
ytr = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    ytr.append([[int(vals[1])]])
X = np.array(xtr)
Y = np.array(ytr)
xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Training set of " + str(X.size) + " elements")
net = Network("MONK" + str(monk) + " test", binary_crossentropy, binary_crossentropy_prime)
net.add(FullyConnectedLayer(6, 10, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(10, 1, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
# train
net.summary()
history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.01, verbose=True, early_stopping=25)

# accuracy on validation set
out = net.predict(xvl)
accuracy = 0
for i in range(len(out)):
    val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
    if (yvl[i].item() == val): accuracy += 1
accuracy /= len(out)
accuracy *= 100
print("\n\nAccuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

suffix = "MONK" + str(monk) + "_" + ts

net.savenet("models/" + suffix + ".pkl")

newnet = Network("TESTNET")
newnet.summary()
newnet.loadnet("models/" + suffix + ".pkl")
newnet.summary()
out = newnet.predict(xvl)
accuracy = 0
for i in range(len(out)):
    val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
    if (yvl[i].item() == val): accuracy += 1
accuracy /= len(out)
accuracy *= 100
print("\n\nAccuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")
