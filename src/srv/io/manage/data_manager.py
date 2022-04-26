

from src.srv.io.loaders.data_loader import DataLoader


class DataManager():
    def __init__(self, filepath: str = None, identities=None, data=None):
        self.loader = DataLoader()
        self.source = filepath
        self.data = data
        if data is None:
            self.data = self.loader.load_data(filepath, identities=identities)

    def __repr__(self) -> str:
        return str(self.data)
