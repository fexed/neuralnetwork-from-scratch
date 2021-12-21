class KFold:
    # KFold data structure to easily get the folds of a given dataset
    def __init__(self, K, dataset):
        self.K = K

        # builds the first fold, made from the first elements_per_fold elements
        # of the dataset for the validation set and the rest of the dataset for
        # the training set
        self.current_fold = 0
        self.folds = []
        self.elements_per_fold = len(dataset)//self.K
        for i in range(0, len(dataset), self.elements_per_fold):
            self.folds.append(dataset[i : i + self.elements_per_fold])
        self.trset = [elem for sublist in self.folds[1:] for elem in sublist]
        self.vlset = self.folds[0]

    def current_folds(self):
        return self.trset, self.vlset

    def next_fold(self):
        self.current_fold += 1
        if (self.current_fold < self.K):
            # the validation set will be the current_fold-th fold, and the
            # training set will be the remaining elements of the dataset
            self.vlset = self.folds[self.current_fold]
            self.trset = [elem for sublist in self.folds[0:self.current_fold] for elem in sublist]
            self.trset.extend([elem for sublist in self.folds[self.current_fold+1:] for elem in sublist])
            return self.trset, self.vlset
        else:
            # no more folds left
            # TODO: better error handling? Not important for now
            raise ValueError
