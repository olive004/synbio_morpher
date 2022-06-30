

from src.utils.misc.type_handling import assert_uniform_type


class Result():
    def __init__(self, name, result_data, category, vis_func, save_numerical_vis_data=False,
                 vis_kwargs=None, analytics_kwargs=None) -> None:
        self.name = name
        self.data = result_data
        self.category = category
        self.save_numerical_vis_data = save_numerical_vis_data
        self.vis_func = vis_func
        self.vis_kwargs = vis_kwargs
        self.analytics_kwargs = analytics_kwargs

        self.analytics = {}
        if category == 'time_series':
            from src.src.utils.results.analytics.timeseries import Timeseries
            analytics_kwargs = analytics_kwargs if analytics_kwargs is not None else {}
            self.analytics = Timeseries(
                result_data).generate_analytics(**analytics_kwargs)

    def __repr__(self):
        str_rep = [f'\n\nResult {self.name}\n']
        for k, v in self.__dict__.items():
            str_rep.append(f'{k}: {v}\n')
        return ''.join(str_rep)


class ResultCollector():

    def __init__(self) -> None:

        self.results = {}

    def add_result(self, result_data, category, vis_func, name, save_numerical_vis_data=False,
                   vis_kwargs=None, analytics_kwargs=None):
        """ category: 'time_series', 'graph' """
        name = f'Result_{len(self.results.keys())}' if not name else name
        result_entry = Result(name, result_data, category,
                              vis_func, save_numerical_vis_data, vis_kwargs, analytics_kwargs)
        self.results[name] = result_entry

    def get_result(self, key):
        return self.results.get(key, None)

    def pool_results(self, results: dict):
        assert_uniform_type(results.values, Result)
        self.results.update(results)

    def reset(self):
        self.results = {}
