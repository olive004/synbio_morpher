

class Mutations():

    def __init__(self, source, positions, targets, algorithm='random') -> None:
        self.source = source
        self.positions = positions
        self.targets = targets
        self.count = len(positions)
        self.algorithm = algorithm


class Evolver():

    def __init__(self) -> None:
        pass

    def mutate(self, data, algorithm, **specs):
        mutator = self.get_mutator()
        
        return mutator(data)

    def get_mutator(self, algorithm):

        def mutator():
            pass
        pass


