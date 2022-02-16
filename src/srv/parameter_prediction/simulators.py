from src.utils.parameter_prediction.interactions import InteractionData, RawSimulationHandling


class InteractionSimulator():
    def __init__(self, config_args, simulator_choice: str):

        self.simulation_handler = RawSimulationHandling(simulator_choice, config_args)

    def run(self, batch=None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """

        simulation_func = self.simulation_handler.get_simulation(allow_self_interaction)
        data = simulation_func(batch)
        data = InteractionData(data, simulation_handler=self.simulation_handler)
        return data

        