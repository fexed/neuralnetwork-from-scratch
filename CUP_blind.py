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
xtest, _ = load_cup(verbose=False, file="test")
ytest = net.predict(xtest);

output_file = open("datasets/CUP/team-name_ML-CUP21-TS.csv", "w")
count = 1
output_file.write("#Federico Matteoni, Riccardo Parente, Sergio Latrofa\n"+
                  "#team-name\n"+
                  "#ML CUP21 v1\n"+
                  "# June 2022\n")
for data in ytest:
    output_file.write(count+","+data[0]+","+data[1]+"\n")
    count += 1
output_file.close()

