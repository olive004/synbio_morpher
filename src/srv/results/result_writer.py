

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

    def make_report(self, keys, source: dict, new_report=False):
        filename = 'report.txt'
        for writeable in keys:
            with open(filename, 'w') as fn:
                fn.write(f'{writeable}: \n' + str(source.get(writeable, '')))

    def write_metrics(self, result: dict, new_report):
        metrics = result.get('metrics', {})
        if 'first_derivative' in metrics.keys():
            result['vis_func'](metrics['first_derivative'],
                               new_vis=new_report, save_name='test_first_derivative',
                               **result['vis_kwargs'])
        writeables = ['steady_state', 'fold_change']
        self.make_report(writeables, metrics, new_report)

    def write_all(self, new_report=False):

        for result in self.results:
            result['vis_func'](
                result['data'], new_vis=new_report, **result['vis_kwargs'])
            self.write_metrics(result, new_report=new_report)
