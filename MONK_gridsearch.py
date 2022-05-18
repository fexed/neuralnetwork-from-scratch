import numpy as np
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_monk
from utils import tr_vl_split

result_file = open("datasets/MONK/grid_search/results.txt", "w")
for monk in range(1,4):
    X, Y, input_size = load_monk(monk, use_one_hot=True)
    #xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio = 0.2)

    res = grid_search(input_size, 1, X, Y, layers=[0,1,2], units=list(range(15, 20)), learning_rates=[0.005, 0.01, 0.1], batch_sizes=[1], init_functions=["xavier", "normalized_xavier"], momentums=[0, 0.8, 0.99], regularizators=[None, L2], epochs=1000, verbose=False)
    print("MONK "+str(monk)+"\n")
    result_file.write("MONK "+str(monk)+":\n")
    for i in range (0, 10):
        result_file.write(str(i+1)+": "+str(res[i]))
        print("\t" + str(res[i]))
result_file.close()
# Best: 'loss': 3.956661378148177e-05, 'layers': 1, 'units': 20, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.8, 'regularizator': None
# Top 10:
        #{'loss': 3.956661378148177e-05, 'layers': 1, 'units': 20, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.8, 'regularizator': None}
        #{'loss': 4.277376054781649e-05, 'layers': 1, 'units': 16, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.8, 'regularizator': None}
        #{'loss': 4.8792562482848936e-05, 'layers': 1, 'units': 16, 'learning_rate': 0.005, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.99, 'regularizator': None}
        #{'loss': 5.65974452412603e-05, 'layers': 1, 'units': 18, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.8, 'regularizator': None}
        #{'loss': 6.078985624791223e-05, 'layers': 1, 'units': 17, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.8, 'regularizator': None}
        #{'loss': 6.329797667950901e-05, 'layers': 1, 'units': 19, 'learning_rate': 0.1, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.8, 'regularizator': None}
        #{'loss': 7.269136816978436e-05, 'layers': 1, 'units': 18, 'learning_rate': 0.005, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.99, 'regularizator': None}
        #{'loss': 8.163203326838232e-05, 'layers': 1, 'units': 17, 'learning_rate': 0.005, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.99, 'regularizator': None}
        #{'loss': 8.606551589452759e-05, 'layers': 1, 'units': 12, 'learning_rate': 0.005, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.99, 'regularizator': None}
        #{'loss': 0.00011595615090431107, 'layers': 1, 'units': 19, 'learning_rate': 0.005, 'batch_size': 1, 'init_function': 'xavier', 'momentum': 0.99, 'regularizator': None}
