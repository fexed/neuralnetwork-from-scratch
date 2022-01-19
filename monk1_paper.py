from activationfunctions import Sigmoid
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_and_save
from dataset_loader import load_monk

monk = 1
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# Training

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)

# training
net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime, momentum=0.8)
net.add(FullyConnectedLayer(input_size, 3, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(3, 3, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(3, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history = net.training_loop(X_TR, Y_TR, epochs=1000, learning_rate=0.1, verbose=True, early_stopping=50)

# Model evaluation

# test set loading
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# evaluating
out = net.predict(X_TS)
accuracy = 0
for i in range(len(out)):
    val = 0 if out[i].item() < 0.5 else 1
    if (Y_TS[i].item() == val): accuracy += 1
accuracy /= len(out)
accuracy *= 100
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
plot_and_save(title="MONK1 model evaluation", history=history, ylabel="Loss", xlabel="Epochs", savefile="MONK1TEST")

# saving the net
net.savenet("models/MONK1TESTED_1L_20U_0.8M_0.1LR_xavier.pkl")
