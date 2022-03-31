

from src.utils.data.loaders.loader import DataLoader


class DataManager():
    def __init__(self, source=None, identities=None, data=None):
        self.loader = DataLoader()
        self.data = data
        if data is None:
            self.data = self.loader.load_data(source) 
        self.sample_names = list(self.data.values())

        self.identities = self.convert_names_to_idxs(identities, self.sample_names)

    @staticmethod
    def convert_names_to_idxs(names_table: dict, source: list) -> dict:
        indexed_identities = {}
        for name_type, name in names_table.items():
            if name in source:
                indexed_identities[name_type] = source.index(name)
        return indexed_identities

    def get_data(self, idx):
        if idx not in self.sample_names:
            return self.data[self.sample_names[idx]]
        else:
            return self.data[idx]

    @property
    def size(self):
        return len(self.data)

    @size.getter
    def size(self):
        return len(self.data)

    @property
    def sample_names(self):
        return self._sample_names

    @sample_names.getter
    def sample_names(self):
        if type(self.data) == dict:
            return list(self.data.keys())
        else:
            import numpy as np
            return list(np.range(len(self.data)))

    @sample_names.setter
    def sample_names(self, value):
        if type(value) == list or type(value) == dict:
            self._sample_names = value
        else:
            raise ValueError(f'Wrong type "{type(value)}" for ' +
                             'setting data sample_names to {value}')

    def __repr__(self) -> str:
        return str(self.data)
