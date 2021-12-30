import numpy as np
from sklearn.model_selection import train_test_split
from grid_search import grid_search


monkfile = open("/home/fexed/ML/fromscratch/datasets/MONK/monks-1.train", "r")
xtr = []
ytr = []
for line in monkfile.readlines():
    vals = line.split(" ")
    xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
    ytr.append([[int(vals[1])]])
X = np.array(xtr)
Y = np.array(ytr)
xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)

grid_search(6, 1, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=[1,2,3,4,5], units=[5,10,15,20], learning_rates=[0.005, 0.01, 0.1], batch_sizes=[None, xtr.size()], epochs=500)

# Best: 0.00005 with {'layers': 5, 'units': 15, 'learning_rate': 0.01}
