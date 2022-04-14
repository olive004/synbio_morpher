

from functools import partial

from src.utils.data.common import Data


class DataLoader():

    def __init__(self) -> None:
        pass

    def get_loader(self, filepath: str):
        from src.utils.data.data_format_tools.common import determine_data_format
        data_type = determine_data_format(filepath)
        loader = self.get_loader_by_dtype(data_type)
        return loader

    @staticmethod
    def get_loader_by_dtype(data_type):
        if data_type == "fasta":
            from src.utils.data.data_format_tools.manipulate_fasta \
                import load_seq_from_FASTA
            return partial(load_seq_from_FASTA, as_type='dict')
        else:
            raise NotImplementedError(
                "Other filetypes than fasta not supported yet.")

    def load_data(self, filepath: str, identities):
        loader = self.get_loader(filepath)
        return Data(loader(filepath), identities, source_files=filepath)
