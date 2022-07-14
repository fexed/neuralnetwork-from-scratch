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
        self.elements = itertools.product(*subspaces)

    def __iter__(self): 
        return SpaceIterator(self.elements)


class SpaceIterator():
    def __init__(self, elements) :
        self.elements = elements

    def __next__(self): 
        return self.elements.__next__()