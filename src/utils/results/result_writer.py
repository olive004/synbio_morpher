

import logging
import os

import numpy as np
from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.results import Result
from src.utils.results.writer import DataWriter
from src.utils.misc.numerical import transpose_arraylike
from src.utils.misc.string_handling import make_time_str
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


class ResultWriter(DataWriter):
    def __init__(self, purpose, out_location=None) -> None:
        super().__init__(purpose, out_location)
        self.report = {}

    def make_metric_visualisation(self, result, keys, source: dict, new_report: bool):
        for plotable in keys:
            if plotable in source.keys():
                out_name = f'{result.name}_{plotable}'
                result.vis_kwargs.update({
                    'data': source.get(plotable),
                    'new_vis': new_report
                })
                self.output(out_name=out_name,
                            write_func=result.vis_func, **result.vis_kwargs)

    def make_report(self, keys: list, source: dict):

        def prettify_writeable(writeable):
            if type(writeable) == np.ndarray or type(writeable) == list:
                writeable = np.array(writeable)
                if writeable.ndim == 2 and np.shape(writeable)[1] == 1:
                    writeable = np.squeeze(writeable)
                writeable = list(writeable)
            elif type(writeable) == dict:
                for k, v in writeable.items():
                    writeable[k] = prettify_writeable(v)
            return writeable

        report = {}
        for writeable in keys:
            report[writeable] = prettify_writeable(source.get(writeable, ''))
        self.report = report

        return report

    def write_report(self, writeables: list, analytics: dict, new_report: bool, out_name: str='report', out_type: str='json'):
        report = self.make_report(writeables, analytics)

        if new_report:
            out_name = out_name + '_' + make_time_str()
        self.output(out_type, out_name, overwrite=not(
            new_report), data=report)

    def write_numerical(self, data, out_name: str):
        self.output(out_type='csv', out_name=out_name,
                    data=transpose_arraylike(data))

    def write_analytics(self, result: Result, new_report=False):
        analytics = result.analytics
        writeables = Timeseries(data=None).get_analytics_types()
        self.write_report(writeables, analytics, new_report, out_name=f'report_{result.name}')

    def write_results(self, results: dict, new_report=False, no_visualisations=False,
                      only_numerical=False, no_analytics=False, no_numerical=False):

        for _name, result in results.items():
            result.vis_kwargs.update(
                {'new_vis': new_report, 'data': result.data})

            if not no_numerical:
                self.write_numerical(
                    data=result.data, out_name=result.name + '_data')
            if not only_numerical:
                if not no_visualisations:
                    self.visualise(out_name=result.name,
                                   writer=result.vis_func, vis_kwargs=result.vis_kwargs)
                    plottables = ['first_derivative']
                    self.make_metric_visualisation(
                        result, plottables, result.analytics, new_report)
                if not no_analytics:
                    self.write_analytics(result, new_report=new_report)

    def write_all(self, circuit: BaseSystem, new_report: bool, no_visualisations: bool = False,
                  only_numerical: bool = False):
        if not no_visualisations:
            self.visualise_graph(circuit)
        self.write_results(circuit.result_collector.results,
                           new_report=new_report, no_visualisations=no_visualisations,
                           only_numerical=only_numerical)

    def visualise(self, out_name, writer, vis_kwargs):
        self.output(out_name=out_name, write_func=writer, **vis_kwargs)

    def visualise_graph(self, circuit: BaseSystem, mode="pyvis", new_vis=False):
        circuit.refresh_graph()

        out_path = os.path.join(self.write_dir, 'graph')
        if mode == 'pyvis':
            from src.utils.results.visualisation import visualise_graph_pyvis
            visualise_graph_pyvis(graph=circuit.graph,
                                  out_path=out_path, new_vis=new_vis)
        else:
            from src.utils.results.visualisation import visualise_graph_pyplot
            visualise_graph_pyplot(graph=circuit.graph, new_vis=new_vis)
