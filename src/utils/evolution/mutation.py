

class Mutations():

    mapping = {
        "A": {
            "C": 0,
            "G": 1,
            "T": 2
        },
        "C": {
            "A": 3,
            "G": 4,
            "T": 5
        },
        "G": {
            "A": 6,
            "C": 7,
            "T": 8
        },
        "T": {
            "A": 9,
            "C": 10,
            "G": 11
        }
    }

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


