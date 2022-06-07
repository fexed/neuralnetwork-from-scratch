from activationfunctions import Sigmoid
from losses import BinaryCrossentropy, MSE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from metrics import Accuracy
from utils import multiline_plot
from dataset_loader import load_monk
from regularizators import L2, Thrun

monk = 2
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# training
net = Network("MONK" + str(monk), MSE())
net.add(FullyConnectedLayer(input_size, 2, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(2, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history, test_history, metric_history, metric_test_history = net.training_loop(X_TR, Y_TR, X_validation=X_TS, Y_validation=Y_TS, 
    epochs=200, learning_rate=0.05, verbose=True, metric=Accuracy(), batch_size=len(X_TR))

# evaluating
accuracy = Accuracy().compute(net, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
multiline_plot(title="MONK2: MSE", legend_names=["Training set", "Test set"], histories=[history, test_history], 
    ylabel="MSE", xlabel="Epochs", savefile="MONK2_PAPER_MSE", showlegend=True, showgrid=True, alternateDots=True)
multiline_plot(title="MONK2: Accuracy", legend_names=["Training set", "Test set "], histories=[metric_history, metric_test_history], 
    ylabel="Accuracy", xlabel="Epochs", savefile="MONK2_PAPER_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)

# saving the net
net.savenet("models/MONK3PAPER.pkl")
