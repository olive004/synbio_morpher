

import logging
import os
from src.srv.results.results import Result
from src.utils.data.manage.writer import DataWriter
from src.utils.misc.string_handling import make_time_str
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


class ResultWriter(DataWriter):
    def __init__(self, purpose, out_location=None) -> None:
        super().__init__(purpose, out_location)

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

    def write_metrics(self, result: Result, new_report=False):
        metrics = result.metrics
        plotables = ['first_derivative']
        writeables = ['steady_state', 'fold_change']
        self.make_metric_visualisation(result, plotables, metrics, new_report)
        self.make_report(writeables, metrics, new_report)

    def write_all(self, results: dict, new_report=False):

        logging.info(f'Writing results {results}')
        for name, result in results.items():
            logging.info(f'Writing result {result}')
            result.vis_func(
                result.data, new_vis=new_report, **result.vis_kwargs)
            self.write_metrics(result, new_report=new_report)

    def visualise(self, circuit: BaseSystem, mode="pyvis", new_vis=False):

        out_path = os.path.join(self.write_dir, 'graph')
        if mode == 'pyvis':
            from src.srv.results.visualisation import visualise_graph_pyvis
            visualise_graph_pyvis(graph=circuit.graph,
                                  out_path=out_path, new_vis=new_vis)
        else:
            from src.srv.results.visualisation import visualise_graph_pyplot
            visualise_graph_pyplot(graph=circuit.graph, new_vis=new_vis)
