from functools import partial


class DataManager():
    def __init__(self, source):
        self.data = self.load_data(source)

    def load_data(self, source):
        from src.utils.data.data_format_tools.common import determine_data_format
        filepath = source
        data_type = determine_data_format(filepath)
        loader = self.get_loader(data_type)
        return loader(filepath)

    def get_loader(self, data_type):
        if data_type == "fasta":
            from src.utils.data.data_format_tools.manipulate_fasta \
                import load_seq_from_FASTA
            return partial(load_seq_from_FASTA, as_type='dict')
        else:
            raise NotImplementedError(
                "Other filetypes than fasta not supported yet.")

    @property
    def size(self):
        return len(self.data)

    @property
    def sample_names(self):
        if type(self.data) == dict:
            return list(self.data.values())
        else:
            import numpy as np
            return list(np.range(len(self.data)))
