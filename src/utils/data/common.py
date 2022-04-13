

class Data():

    def __init__(self, loaded_data: dict, identities: dict = {}, source_files=None) -> None:
        self.data = loaded_data
        self.source = source_files
        self.sample_names = self.make_sample_names()

        self.identities = self.convert_names_to_idxs(identities, self.sample_names)

    def get_data_by_idx(self, idx):
        if idx not in self.sample_names:
            return self.data[self.sample_names[idx]]
        else:
            return self.data[idx]

    def add_data(self, name, data):
        self.data[name] = data
        self.sample_names.append(name)

    @staticmethod
    def convert_names_to_idxs(names_table: dict, source: list) -> dict:
        """ Output and input species definition """
        indexed_identities = {}
        for name_type, name in names_table.items():
            if name in source:
                indexed_identities[name_type] = source.index(name)
        return indexed_identities

    def make_sample_names(self):
        if type(self.data) == dict:
            return list(self.data.keys())
        raise ValueError(f'Unrecognised loaded data type {type(self.data)}.')

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