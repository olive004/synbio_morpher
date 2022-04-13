

from functools import partial
import json
import logging
import os
from src.srv.results.metrics.analytics import Analytics
from src.utils.data.manage.writer import DataWriter
from src.utils.misc.string_handling import add_outtype, make_time_str
from src.utils.misc.type_handling import assert_uniform_type


class Result():
    def __init__(self, name, result_data, category, vis_func, **vis_kwargs) -> None:
        self.name = name
        self.data = result_data
        self.category = category
        self.vis_func = vis_func
        self.vis_kwargs = vis_kwargs

        self.metrics = {}
        self.analytics = Analytics(result_data, category)
        if category == 'time_series':
            from src.srv.results.metrics.plotting import Timeseries
            self.metrics = Timeseries(result_data).generate_analytics()


class ResultWriter(DataWriter):
    def __init__(self, purpose, out_location=None) -> None:
        super().__init__(purpose, out_location)
        self.results = {}

    def add_result(self, result_data, category, vis_func, name, **vis_kwargs):
        """ category: 'time_series', 'graph' """
        name = f'Result_{len(self.results.keys())}' if not name else name
        if 'out_path' in vis_kwargs.keys():
            vis_kwargs['out_path'] = os.path.join(
                self.write_dir, vis_kwargs['out_path'])
        result_entry = Result(name, result_data, category,
                              vis_func, **vis_kwargs)
        self.results[name] = result_entry

    def get_result(self, key):
        return self.results.get(key, None)

    def make_metric_visualisation(self, result, keys, source: dict, new_report: bool):
        for plotable in keys:
            if plotable in source.keys():
                out_name = f'{result.name}_{plotable}'
                result.vis_kwargs['out_path'] = os.path.join(
                    self.write_dir, out_name)
                result.vis_func(source.get(plotable),
                                new_vis=new_report,
                                **result.vis_kwargs)

    def make_report(self, keys, source: dict, new_report: bool, out_name='report', out_type='json'):
        if new_report:
            out_name = out_name + '_' + make_time_str()
        write_dict = {}
        for writeable in keys:
            write_dict[writeable] = source.get(writeable, '')
        self.output(out_type, out_name, overwrite=not(new_report), **{'data': write_dict})

    def pool_results(self, results: dict):
        assert_uniform_type(results.values, Result)
        self.results.update(results)

    def write_metrics(self, result: Result, new_report=False):
        metrics = result.metrics
        plotables = ['first_derivative']
        writeables = ['steady_state', 'fold_change']
        self.make_metric_visualisation(result, plotables, metrics, new_report)
        self.make_report(writeables, metrics, new_report)

    def write_all(self, new_report=False):

        for name, result in self.results.items():
            result.vis_func(
                result.data, new_vis=new_report, **result.vis_kwargs)
            self.write_metrics(result, new_report=new_report)

    def visualise(self, circuit, mode="pyvis", new_vis=False):

        out_path = os.path.join(self.write_dir, 'graph')
        if mode == 'pyvis':
            from src.srv.results.visualisation import visualise_graph_pyvis
            visualise_graph_pyvis(graph=circuit.graph,
                                  out_path=out_path, new_vis=new_vis)
        else:
            from src.srv.results.visualisation import visualise_graph_pyplot
            visualise_graph_pyplot(graph=circuit.graph, new_vis=new_vis)
