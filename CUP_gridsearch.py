import numpy as np
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import tr_vl_split


X, Y = load_cup(verbose=True, file="training")

#res = grid_search(10, 2, X, Y, layers=[0], units=[23], learning_rates=[0.0025], batch_sizes=[16, 32, 64], init_functions=["normalized_xavier"], momentums=[0.25, 0.5, 0.75, 0.85, 0.9, 0.99], regularizators=[None], dropouts=[0], nesterovs=[0], epochs=22500000, verbose=False, early_stopping=150)
res = grid_search(10, 2, X, Y, layers=[0], units=list(range(21, 24, 1)), learning_rates=[0.00125, 0.001, 0.000625], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0], regularizators=[None, L2(l = 1e-4), L2(l = 1e-5)], dropouts=[0], nesterovs=[0], epochs=1500, verbose=False, early_stopping=100)
result_file = open("datasets/CUP/grid_search/results.txt", "w")
print("CUP\n")
result_file.write("CUP:\n")
for i in range (0, 10):
    result_file.write(str(i+1)+": "+str(res[i])+"\n")
    print("\n" + str(res[i]))
result_file.close()
