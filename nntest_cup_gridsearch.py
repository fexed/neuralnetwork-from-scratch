import numpy as np
from sklearn.model_selection import train_test_split
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer

result_file = open("datasets/CUP/grid_search/results.txt", "w")
X, Y = load_cup(verbose=True, test=False)
X, n_min, n_max = min_max_normalizer(X)
X, means, std = continuous_standardizer(X)
xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)

res = grid_search(10, 2, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=list(range(3, 4)), units=list(range(28, 32)), learning_rates=[0.001], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0], regularizators=[None, L2], regularization_lambdas=[0.005, 0.15], epochs=1000, verbose=False, early_stopping=150)
print("CUP\n")
result_file.write("CUP:\n")
for i in range (0, 10):
    result_file.write(str(i+1)+": "+str(res[i]))
    print("\t" + str(res[i]))
result_file.close()
