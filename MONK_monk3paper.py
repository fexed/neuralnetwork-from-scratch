from activationfunctions import Sigmoid
from losses import BinaryCrossentropy, MSE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from metrics import Accuracy
from utils import multiline_plot
from dataset_loader import load_monk
from regularizators import L2, Thrun

monk = 3
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# Training

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# training
net = Network("MONK" + str(monk), MSE(), regularizator=Thrun(0.0005))
net.add(FullyConnectedLayer(input_size, 4, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(4, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history, test_history, metric_history, metric_test_history = net.training_loop(X_TR, Y_TR, X_validation=X_TS, Y_validation=Y_TS, epochs=300, learning_rate=0.25, verbose=True, metric=Accuracy())

# Model evaluation

# test set loading
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# evaluating
accuracy = Accuracy().compute(net, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
multiline_plot(title="MONK3 MSE (regularized)", legend_names=["Training MSE", "Test Set MSE"], histories=[history, test_history], ylabel="MSE", xlabel="Epochs", savefile="MONK3_PAPER_REG_MSE", showlegend=True, showgrid=True, alternateDots=True)
multiline_plot(title="MONK3 Accuracy (regularized)", legend_names=["Training Accuracy", "Test Set Accuracy"], histories=[metric_history, metric_test_history], ylabel="Accuracy", xlabel="Epochs", savefile="MONK3_PAPER_REG_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)

# saving the net
net.savenet("models/MONK3PAPER.pkl")