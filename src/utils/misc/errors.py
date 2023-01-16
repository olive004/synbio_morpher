

class ConfigError(Exception):
    """ Raised when there is a problem with the way 
    the configuration file that was used was defined. """
    pass


class ExperimentError(Exception):
    """ Raised when the script was not used in the intended
    manner, for example inputs in the wrong format """
    pass
