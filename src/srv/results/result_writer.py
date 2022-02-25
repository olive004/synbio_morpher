

class ResultWriter():
    def __init__(self) -> None:
        self.results = {}

    def add_result(self, result, category, vis_func, name=None, **vis_kwargs):
        """ category: 'time_series', 'graph' """
        name = f'Result_{len(self.results.keys())}' if not name else name
        result_entry = self.curate_result(
            result, category, vis_func, name, **vis_kwargs)
        self.results[name] = result_entry

    def get_result(self, key):
        return self.results.get(key, None)

    def curate_result(self, result, category, vis_func, name, **vis_kwargs):
        metrics = []
        if category == 'time_series':
            from src.srv.results.metrics.plotting import Timeseries
            metrics = Timeseries(result).generate_analytics()
        result_entry = {
            'data': result,
            'category': category,
            'metrics': metrics,
            'name': name,
            'vis_func': vis_func,
            'vis_kwargs': vis_kwargs
        }
        return result_entry

    def make_report(self, keys, source: dict, new_report: bool):
        filename = 'report.txt'
        with open(filename, 'w') as fn:
            for writeable in keys:
                fn.write(f'{writeable}: \n' + str(source.get(writeable, '')) + '\n')

    def write_metrics(self, result: dict, new_report=False):
        metrics = result.get('metrics', {})
        if 'first_derivative' in metrics.keys():
            result['vis_func'](metrics['first_derivative'],
                               new_vis=new_report, save_name=f'{result["name"]}_first_derivative',
                               **result['vis_kwargs'])
        writeables = ['steady_state', 'fold_change']
        self.make_report(writeables, metrics, new_report)

    def write_all(self, new_report=False):

        for name, result in self.results.items():
            result['vis_func'](
                result['data'], new_vis=new_report, **result['vis_kwargs'])
            self.write_metrics(result, new_report=new_report)
