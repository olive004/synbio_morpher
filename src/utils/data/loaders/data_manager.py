

from src.utils.data.loaders.loader import DataLoader


class DataManager():
    def __init__(self, source=None, identities=None, data=None):
        self.loader = DataLoader()
        self.data = data
        if data is None:
            self.data = self.loader.load_data(source, identities) 

    def get_data(self, idx):
        if idx not in self.sample_names:
            return self.data[self.sample_names[idx]]
        else:
            return self.data[idx]

    def __repr__(self) -> str:
        return str(self.data)
