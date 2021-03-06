# Best results
### MONK1
|Date|Accuracy|Hidden layers and units|Learning rate|Momentum|Init. function|Activ. function|Notes|
|---|---|---|---|---|---|---|---|
|16/01/2022|100%|[20]|0.1|0.8|Xavier|Sigmoid| |
|14/01/2022|71.63%|[20]|0.1|0.8|Xavier|Sigmoid| |
|11/01/2022|74.80%|[15, 15, 15]|0.01| | | | | |

### MONK2
|Date|Accuracy|Hidden layers and units|Learning rate|Momentum|Init. function|Activ. function|Notes|
|---|---|---|---|---|---|---|---|
|16/01/2022|100%|[10]|0.05| |Norm. Xavier|Sigmoid| |
|14/01/2022|88.10%|[20]|0.1|0.8|Xavier| | | |
|11/01/2022|59.52%|[15, 15, 15]|0.01| | | | | |

### MONK3
|Date|Accuracy|Hidden layers and units|Learning rate|Momentum|Init. function|Activ. function|Notes|
|---|---|---|---|---|---|---|---|
|16/01/2022|94.21%|[10]|0.01| |Xavier|Sigmoid| |
|14/01/2022|90.67%|[20]|0.1|0.8|Xavier| | | |
|11/01/2022|74.17%|[15, 15, 15]|0.01| | | | | |

### CUP
|Date|Accuracy|MEE|Loss|Hidden layers and units|Learning rate|Momentum|Init. function|Activ. function|Notes|
|---|---|---|---|---|---|---|---|---|---|
|17/05/2022|0% 2.37%|0.07951|1.1123 (MEE)|[23]|0.0025| |Norm. Xavier|Tanh|L2(l = 1e-05)|
|13/05/2022|0.67% 2.37%|0.09735|0.6836 (MSE)|[25,25]|0.001| |Norm. Xavier|Tanh| |
|10/05/2022|0%| |0.9846 (MSE)|[30]|0.001|0.85|He|Tanh|Quick baseline test|
