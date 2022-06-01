

import logging
from src.srv.io.loaders.data_loader import DataLoader
from src.utils.data.common import Data


class DataManager():
    def __init__(self, filepath: str = None, identities: dict = None, data=None, sample_names: list = None):
        self.loader = DataLoader()
        self.source = filepath
        self.data = data
        if data is None and filepath:
            self.data = self.loader.load_data(filepath, identities=identities)
        elif data is None and filepath is None:
            self.data = Data(data, identities=identities, sample_names=sample_names)
            logging.warning(
                f'No data: either a data filepath or raw data must be supplied to DataManager')
            # raise ValueError('Either a data filepath or raw data must be supplied.')

    def __repr__(self) -> str:
        return str(self.data)
