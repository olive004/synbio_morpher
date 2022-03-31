from functools import partial


class Data():

    def __init__(self, loaded_data, identities) -> None:
        self.raw = loaded_data
        self.sample_names = self.make_sample_names()

        self.identities = self.convert_names_to_idxs(identities, self.sample_names)

    @staticmethod
    def convert_names_to_idxs(names_table: dict, source: list) -> dict:
        indexed_identities = {}
        for name_type, name in names_table.items():
            if name in source:
                indexed_identities[name_type] = source.index(name)
        return indexed_identities

    def make_sample_names(self):
        if type(self.raw) == dict:
            return list(self.raw.values())
        raise ValueError(f'Unrecognised loaded data type {type(self.raw)}.')


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

    def load_data(self, source: str, identities):
        loader = self.get_loader(source)
        return Data(loader(source), identities)
