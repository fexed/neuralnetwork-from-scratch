import numpy as np


# Returns encoded input and number of new features (bits needed) 
def one_hot_encoding(input): 
    encoded_input = []
    maxs = []
    mins = []
    
    # Compute range of integers representing the categorical 
    for feature in np.transpose(input):
        maxs.append(max(feature[0]))
        mins.append(min(feature[0]))

    offsets = np.array(maxs) - np.array(mins) + 1 
    # Compute the number of bits needed
    bits = sum(offsets)

    # Compute the mapping of each pattern
    for pattern in input: 
        one_hot_input = np.full(bits, 0)
        feat_idx = 0
        for feat, offset in zip(pattern[0]-np.array(mins), offsets):
            one_hot_input[feat_idx + feat] = 1
            feat_idx += offset
        encoded_input.append([one_hot_input])

    return np.array(encoded_input), bits

# Returns standardized input, menas and standard deviations of each feature (pre standardization)
def continuous_standardizer(input): 
    standardized_input = []
    mean = [] 
    std = []
    
    # Compute mean and standard deviation of each feature
    for feature in np.transpose(input):
        mean.append(np.mean(feature[0]))
        std.append(np.std(feature[0]))

    mean = np.array(mean)
    std = np.array(std)
    
    # Scale each pattern of the dataset
    for pattern in input:
        standardized_input.append((pattern - mean)/std)
    
    return np.array(standardized_input), mean, std

# Returns normalized input, min and max of each fdeature (pre normalization)
def min_max_normalizer(input): 
    normalized_input = []
    maxs = []
    mins = []
    
    # Compute min and max of each feature
    for feature in np.transpose(input):
        mins.append(min(feature[0]))
        maxs.append(max(feature[0]))

    mins = np.array(mins)
    maxs = np.array(maxs)
    diffs = maxs - mins
    
    # Normalized each pattern of the dataset
    for pattern in input:
        normalized_input.append((pattern - mins)/diffs )
             
    return np.array(normalized_input), mins, maxs
