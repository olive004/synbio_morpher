

class Modeller():
    """ Common modeller class. For example deterministic vs. 
    stochastic modelling share some things """

    def __init__(self, max_time=0, time_interval=1, t0=0) -> None:
        """ Time step is the dt """
        self.t0 = t0
        self.max_time = int(max_time / time_interval)
        self.original_max_time = max_time
        self.time_interval = time_interval

    def dxdt_RNA(self):
        pass
