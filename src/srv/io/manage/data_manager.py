

from src.srv.io.loaders.data_loader import DataLoader


class DataManager():
    def __init__(self, filepath: str = None, identities=None, data=None):
        self.loader = DataLoader()
        self.source = filepath
        self.data = data
        if data is None and filepath:
            self.data = self.loader.load_data(filepath, identities=identities)
        elif data is None and filepath is None:
            raise ValueError('Either a data filepath or raw data must be supplied.')

    def __repr__(self) -> str:
        return str(self.data)
