import numpy as np

def one_hot_encoding(input): 
    encoded_input = []
    maxs = []
    mins = []
    
    # Calcolate range of integers representing the categorical 
    for feature in np.transpose(input):
        maxs.append(max(feature[0]))
        mins.append(min(feature[0]))

    offsets = np.array(maxs) - np.array(mins) + 1 
    # calculate the number of bits needed
    bits = sum(offsets)

    # compute the mapping of each pattern
    for pattern in input: 
        one_hot_input = np.full(bits, 0)
        feat_idx = 0
        for feat, offset in zip(pattern[0]-np.array(mins), offsets): 
            one_hot_input[feat_idx + feat] = 1
            feat_idx += offset
        encoded_input.append([one_hot_input])
             
    return np.array(encoded_input), bits


