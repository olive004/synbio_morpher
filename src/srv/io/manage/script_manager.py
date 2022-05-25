


from typing import List, Tuple


class Ensembler():

    def __init__(self, data_writer, subscripts: List[Tuple] = None) -> None:
        self.data_writer = data_writer
        self.subscripts = subscripts

    def run(self):
        for (script, config_filepath) in self.subscripts:
            self.data_writer.update_ensemble(script.__name__)
            script(config_filepath, self.data_writer)
