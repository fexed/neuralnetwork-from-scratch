from activationfunctions import Tanh
from losses import MEE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from dataset_loader import load_cup

print("\n\n****CUP")
X, Y = load_cup(verbose=False, file="training_full")
# train
net = Network("CUP", MEE(), nesterov=True, momentum=0.25)
net.add(FullyConnectedLayer(10, 23, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(23, 2, initialization_func="normalized_xavier"))
net.summary()
net.training_loop(X, Y, epochs=1549, learning_rate=0.00125, verbose=True, batch_size=16)
#test
xtest = load_cup(verbose=False, file="blind")
ytest = net.predict(xtest);
output_file = open("datasets/CUP/arbitraryvalues_ML-CUP21-TS.csv", "w")
output_file.write("#Federico Matteoni, Riccardo Parente, Sergio Latrofa\n"+
                  "#arbitraryvalues\n"+
                  "#ML CUP21\n"+
                  "#/06/2022\n")
count = 1
for data in ytest:
    output_file.write(str(count)+","+str(data[0][0])+","+str(data[0][1])+"\n")
    count += 1
output_file.close()
