from functools import partial
from src.utils.system_definition.configurations import load_json_as_dict


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
