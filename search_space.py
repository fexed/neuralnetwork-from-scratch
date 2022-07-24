import itertools

class SearchSpace():
    def __init__(self, subspaces):
        self.cardinality = 1

        self.iterable = []
        self.fixed = []

        for i, subspace in enumerate(subspaces):
            if len(subspace) > 1: 
                self.iterable.append(subspace)
                self.cardinality *= len(subspace)
            elif len(subspace) == 0: 
                self.subspaces.pop(i)
            else:
                self.fixed.append(subspace)

        self.subspaces = subspaces
        self.elements = [ elems for elems in itertools.product(*subspaces)]

    def __iter__(self): 
        return self.elements.__iter__()
