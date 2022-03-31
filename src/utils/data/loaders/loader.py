from functools import partial


class DataLoader():

    def __init__(self) -> None:
        pass

    def load_data(self, source: str):
        from src.utils.data.data_format_tools.common import determine_data_format
        filepath = source
        data_type = determine_data_format(filepath)
        loader = self.get_loader(data_type)
        return loader(filepath)

    @staticmethod
    def get_loader(self, data_type):
        if data_type == "fasta":
            from src.utils.data.data_format_tools.manipulate_fasta \
                import load_seq_from_FASTA
            return partial(load_seq_from_FASTA, as_type='dict')
        else:
            raise NotImplementedError(
                "Other filetypes than fasta not supported yet.")