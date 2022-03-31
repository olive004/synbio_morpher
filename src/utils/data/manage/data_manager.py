

from src.utils.data.loaders.generic import DataLoader


class DataManager():
    def __init__(self, source=None, identities=None, data=None):
        self.loader = DataLoader()
        self.data = data
        if data is None:
            self.data = self.loader.load_data(source, identities) 

    def __repr__(self) -> str:
        return str(self.data)
