
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


class ConfigError(Exception):
    """ Raised when there is a problem with the way 
    the configuration file that was used was defined. """
    pass


class ExperimentError(Exception):
    """ Raised when the script was not used in the intended
    manner, for example inputs in the wrong format """
    pass
