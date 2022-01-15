import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
import matplotlib.pyplot as plot
from preprocessing import one_hot_encoding


monk = 1
print("\n\n****TESTING NETWORK ON MONK" + str(monk))
monkfile = open("datasets/MONK/monks-" + str(monk) + ".train", "r")
xtr = []
ytr = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    ytr.append([[int(vals[1])]])
X = np.array(xtr)
Y = np.array(ytr)

X, input_size = one_hot_encoding(X)
net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime, momentum=0.8)
net.add(FullyConnectedLayer(input_size, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
net.add(FullyConnectedLayer(20, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
net.add(FullyConnectedLayer(20, 1, sigmoid, sigmoid_prime, initialization_func="xavier"))
history = net.training_loop(X, Y, epochs=1000, learning_rate=0.1, verbose=True, early_stopping=50)


monkfile = open("datasets/MONK/monks-" + str(monk) + ".test", "r")
xts = []
yts = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xts.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    yts.append([[int(vals[1])]])
X = np.array(xts)
Y = np.array(yts)
X, input_size = one_hot_encoding(X)
out = net.predict(X)
accuracy = 0
for i in range(len(out)):
    val = 0 if out[i].item() < 0.5 else 1
    if (Y[i].item() == val): accuracy += 1
accuracy /= len(out)
accuracy *= 100
fig, ax = plot.subplots()
ax.plot(history)
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
ax.set_title("MONK1 Test")

print("\n\nAccuracy on the test set: {:.4f}%".format(accuracy))

plot.gca().margins(x=0)
fig.set_size_inches(18.5, 10.5)
plot.savefig("plots/MONKTEST.png")
plot.clf()
net.savenet("models/MONKTESTED_1L_20U_xavier_momentum0.8_LR0.1.pkl")
