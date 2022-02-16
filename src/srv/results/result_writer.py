

class ResultWriter():
    def __init__(self) -> None:
        self.results = []

    def add_result(self, result, visualisation_type, vis_func, **vis_kwargs):
        """ visualisation_type: 'plot', 'graph' """
        result_entry = self.curate_result(
            result, visualisation_type, vis_func, **vis_kwargs)
        self.results.append(result_entry)

    def curate_result(self, result, visualisation_type, vis_func, **vis_kwargs):
        result_entry = {
            'data': result,
            'visualisation_type': visualisation_type,
            'vis_func': vis_func,
            'vis_kwargs': vis_kwargs
        }
        return result_entry
