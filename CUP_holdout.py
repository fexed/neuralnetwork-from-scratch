from random import random
from activationfunctions import LeakyReLU, ReLU, Sigmoid, Tanh
from losses import MEE
from layers import FullyConnectedLayer
from metrics import MeanEuclideanError
from neuralnetwork import Network
from utils import plot_and_save, tr_vl_split, compare
from regularizators import L2
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
net = Network("CUP", MEE(), regularizator=L2(l = 1e-05))
net.add(FullyConnectedLayer(10, 23, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(23, 2,  initialization_func="normalized_xavier"))
net.summary()
history, validation_history = net.training_loop(X_TR, Y_TR, X_validation=X_VAL, Y_validation=Y_VAL, epochs=1000, learning_rate=0.0025, verbose=True, batch_size=1, early_stopping=50)

# Model evaluation


# evaluating
out = net.predict(X_VAL)
accuracy = 0
acc1=0
acc2=0

# print(Y_VAL[0], " ", out[0])
for i in range(len(out)):
    # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
    if (compare(Y_VAL[i][0][0], out[i][0][0]) and compare(Y_VAL[i][0][1], out[i][0][1])): accuracy += 1

    #Delete from here
    if compare(Y_VAL[i][0][0], out[i][0][0]): acc1 +=1
    if compare(Y_VAL[i][0][1], out[i][0][1]): acc2 +=1

acc1 /= len(out)
acc1 *= 100

acc2 /= len(out)
acc2 *= 100
#to here

accuracy /= len(out)
accuracy *= 100

MEE = MeanEuclideanError().compute(net, X_VAL, Y_VAL)

print("\n\nAccuracy on CUP validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

print("\nAccuracy on VAL1", acc1 ,"%")
print("Accuracy on VAL2", acc2, "%")
print("MEE", MEE)

# plotting data
plot_and_save(title="CUP model evaluation", history=history, validation_history=validation_history, ylabel="Loss", xlabel="Epochs", savefile="CUP_HOLDOUT")
#plot_and_save(title="CUP target1 evaluation", history=history, validation_history=validation_history, ylabel="Loss", xlabel="Epochs", savefile="CUP_HOLDOUT")
#plot_and_save(title="CUP target2 evaluation", history=history, validation_history=validation_history, ylabel="Loss", xlabel="Epochs", savefile="CUP_HOLDOUT")

#net.savenet("models/CUP_currentbest.pkl")