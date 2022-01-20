class KFold:
    """ KFold data structure to easily get the folds of a given dataset """

    def __init__(self, K, X, Y):
        self.K = K

        # builds the folds, dividing the dataset in K parts
        self.current_fold = -1  # call next_fold to build the first fold
        self.x_folds = []
        self.y_folds = []
        self.elements_per_fold = len(X)//self.K
        for i in range(0, len(X), self.elements_per_fold):
            self.x_folds.append(X[i : i + self.elements_per_fold])
            self.y_folds.append(Y[i : i + self.elements_per_fold])


    def hasNext(self):
        return self.current_fold+1<self.K


    def next_fold(self):
        """ Returns the next fold, or ValueError if there are no folds left """
        
        self.current_fold += 1
        if (self.current_fold < self.K):
            # the validation set will be the current_fold-th fold, and the
            # training set will be the remaining elements of the dataset
            self.x_vlset = self.x_folds[self.current_fold]
            self.y_vlset = self.y_folds[self.current_fold]
            self.x_trset = [elem for sublist in self.x_folds[0:self.current_fold] for elem in sublist]
            self.x_trset.extend([elem for sublist in self.x_folds[self.current_fold+1:] for elem in sublist])
            self.y_trset = [elem for sublist in self.y_folds[0:self.current_fold] for elem in sublist]
            self.y_trset.extend([elem for sublist in self.y_folds[self.current_fold+1:] for elem in sublist])
            return self.x_trset, self.x_vlset, self.y_trset, self.y_vlset
        else:
            # no more folds left
            # TODO: better error handling? Not important for now
            raise ValueError
