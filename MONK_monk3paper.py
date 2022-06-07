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

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# training
net = Network("MONK" + str(monk), MSE())
net.add(FullyConnectedLayer(input_size, 4, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(4, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history, test_history, metric_history, metric_test_history = net.training_loop(X_TR, Y_TR, X_validation=X_TS, Y_validation=Y_TS, 
    epochs=150, learning_rate=0.05, verbose=True, metric=Accuracy(), batch_size=len(X_TR))

# evaluating
accuracy = Accuracy().compute(net, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
multiline_plot(title="MONK3: MSE", legend_names=["Training set", "Test set"], histories=[history, test_history], 
    ylabel="MSE", xlabel="Epochs", savefile="MONK3_PAPER_MSE", showlegend=True, showgrid=True, alternateDots=True)
multiline_plot(title="MONK3: Accuracy", legend_names=["Training set", "Test set "], histories=[metric_history, metric_test_history], 
    ylabel="Accuracy", xlabel="Epochs", savefile="MONK3_PAPER_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)

# saving the net
net.savenet("models/MONK3PAPER.pkl")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# training with regularization
netr = Network("MONK" + str(monk) + " - Regularized", MSE(), regularizator=L2())
netr.add(FullyConnectedLayer(input_size, 4, Sigmoid(), initialization_func="xavier"))
netr.add(FullyConnectedLayer(4, 1, Sigmoid(), initialization_func="xavier"))
netr.summary()
history_r, test_history_r, metric_history_r, metric_test_history_r = netr.training_loop(X_TR, Y_TR, X_validation=X_TS, Y_validation=Y_TS, 
    epochs=150, learning_rate=0.05, verbose=True, metric=Accuracy(), batch_size=len(X_TR))

# evaluating
accuracyr = Accuracy().compute(netr, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracyr))

# plotting data
multiline_plot(title="MONK3 with regularization: MSE", legend_names=["Training set", "Test set"], histories=[history_r, test_history_r], 
    ylabel="MSE", xlabel="Epochs", savefile="MONK3_PAPER_REG_MSE", showlegend=True, showgrid=True, alternateDots=True)
multiline_plot(title="MONK3 with regularization: Accuracy", legend_names=["Training set", "Test set"], histories=[metric_history_r, metric_test_history_r], 
    ylabel="Accuracy", xlabel="Epochs", savefile="MONK3_PAPER_REG_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)

# saving the net
netr.savenet("models/MONK3PAPER_REG.pkl")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# compare reg and not reg in the same plot/
multiline_plot(title="MONK3: MSE", 
    legend_names=[ "Training set", "Test set", "Training set regularized", "Test set regularized"], 
    histories=[ history, test_history, history_r, test_history_r], ylabel="MSE", xlabel="Epochs", 
    savefile="MONK3_PAPER_COMP_MSE", showlegend=True, showgrid=True, alternateDots=True)

multiline_plot(title="MONK3: Accuracy", 
    legend_names=["Training set", "Test set", "Training set regularized", "Test set regularized"], 
    histories=[metric_history, metric_test_history, metric_history_r, metric_test_history_r], ylabel="Accuracy", xlabel="Epochs", 
    savefile="MONK3_PAPER_COMP_ACCURACY", showlegend=True, showgrid=True, alternateDots=True)

