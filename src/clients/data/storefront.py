

from src.srv.io.manage.data_manager import DataManager


class DataFront():

    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def retrieve(self):
        return DataManager()