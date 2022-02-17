

class ResultWriter():
    def __init__(self) -> None:
        self.results = []

    def add_result(self, result, category, vis_func, **vis_kwargs):
        """ category: 'time_series', 'graph' """
        result_entry = self.curate_result(
            result, category, vis_func, **vis_kwargs)
        self.results.append(result_entry)

    def curate_result(self, result, category, vis_func, **vis_kwargs):
        metrics = []
        if category == 'time_series':
            from src.utils.metrics.plotting import Timeseries
            metrics = Timeseries(result).generate_analytics()
        result_entry = {
            'data': result,
            'category': category,
            'metrics': metrics,
            'vis_func': vis_func,
            'vis_kwargs': vis_kwargs
        }
        return result_entry
