class KFold:
    def __init__(self, K, dataset):
        self.K = K

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
            self.vlset = self.folds[self.current_fold]
            self.trset = [elem for sublist in self.folds[0:self.current_fold] for elem in sublist]
            self.trset.extend([elem for sublist in self.folds[self.current_fold+1:] for elem in sublist])
            return self.trset, self.vlset
        else:
            raise ValueError
