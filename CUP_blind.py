from activationfunctions import Tanh
from losses import MEE
from layers import FullyConnectedLayer
from training import Network
from dataset_loader import load_cup
from regularizators import L2


print("\n\n****CUP")
X, Y = load_cup(verbose=False, file="training_full")
# train
net = Network("CUP", MEE(), regularizator=L2(l=0.0001))
net.add(FullyConnectedLayer(10, 21, Tanh(), initialization_func="normalized_xavier"))
net.add(FullyConnectedLayer(21, 2, initialization_func="normalized_xavier"))
net.summary()
net.training_loop(X, Y, epochs=200, learning_rate=0.00125, verbose=True, batch_size=1)

# test
xtest = load_cup(verbose=False, file="blind")[0]
ytest = net.predict(xtest);
output_file = open("datasets/CUP/arbitraryvalues_ML-CUP21-TS.csv", "w")
output_file.write("#Federico Matteoni, Riccardo Parente, Sergio Latrofa\n"+
                  "#arbitraryvalues\n"+
                  "#ML CUP21\n"+
                  "#05/06/2022\n")
count = 1
for data in ytest:
    output_file.write(str(count)+","+str(data[0][0])+","+str(data[0][1])+"\n")
    count += 1
output_file.close()