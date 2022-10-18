

from copy import deepcopy
import logging
from src.utils.misc.type_handling import assert_uniform_type


class Result():
    def __init__(self, name, result_data, category, vis_func, save_numerical_vis_data=False,
                 vis_kwargs=None, analytics_kwargs=None, analytics=None) -> None:
        self.name = name
        self.data = result_data
        self.category = category
        self.save_numerical_vis_data = save_numerical_vis_data
        self.vis_func = vis_func
        self.vis_kwargs = vis_kwargs
        self.analytics_kwargs = analytics_kwargs

        self.analytics = analytics
        if category == 'time_series':
            from src.utils.results.analytics.timeseries import Timeseries
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

    def add_result(self, data, category: str, vis_func, name: str, save_numerical_vis_data: bool = False,
                   vis_kwargs: dict = None, analytics_kwargs: dict = None, analytics=None) -> None:
        """ category: 'time_series', 'graph' """
        name = f'Result_{len(self.results.keys())}' if not name else name
        result_entry = Result(name, data, category, vis_func, save_numerical_vis_data,
                              vis_kwargs, analytics_kwargs, analytics)
        self.results[name] = result_entry

    def add_modified_duplicate_result(self, key, **add_kwargs):
        result = deepcopy(self.get_result(key))
        result_kwargs = result.__dict__
        result_kwargs.update(add_kwargs)
        self.add_result(**result_kwargs)

    def get_result(self, key) -> Result:
        result = self.results.get(key, None)
        if result is None:
            logging.warning(
                f'The keyword {key} did not return a result from {self}.')
        return result

    def pool_results(self, results: dict):
        assert_uniform_type(results.values, Result)
        self.results.update(results)

    def reset(self):
        self.results = {}
