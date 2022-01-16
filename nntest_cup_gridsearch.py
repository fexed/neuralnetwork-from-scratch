import numpy as np
from sklearn.model_selection import train_test_split
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_cup

result_file = open("datasets/CUP/grid_search/results.txt", "w")
X, Y = load_cup(verbose=True, test=False)
xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)

res = grid_search(10, 1, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=[0,1,2], units=list(range(5, 20)), learning_rates=[0.001, 0.005, 0.01], batch_sizes=[1], init_functions=["xavier", "normalized_xavier"], momentums=[0, 0.8, 0.99], regularizators=[None, L2], epochs=1000, verbose=True)
print("CUP\n")
result_file.write("CUP:\n")
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
