import inspect
import logging
import numpy as np
from typing import List
# from src.utils.signal.signals import Signal, AdaptationTarget, OscillatingSignal
from src.utils.signal.signals_new import Signal
from src.utils.circuit.agnostic_circuits.circuit_manager_new import Circuit


# TODO: Delete
def get_signal_type(signal_type: str) -> Signal:
    return Signal
#     if signal_type == 'abstract':
#         return Signal
#     if signal_type == 'adaptation':
#         return AdaptationTarget
#     if signal_type == 'oscillation':
#         return OscillatingSignal
#     raise ValueError(f'Could not identify signal of type {signal_type}.')


def make_onehot_signals(circuit: Circuit, species_chosen: List[str]):
    """ Return an ndarray where 1 denotes 'signal input / output' 
    and 0 denotes the opposite"""
    onehot = np.zeros(circuit.circuit_size)
    idxs = [circuit.model.species.index(s) for s in circuit.model.species if s.name in species_chosen]
    onehot[idxs] = 1
    return onehot

def parse_sig_args(kwargs: dict, circuit: Circuit, SignalType = Signal):
    signal_init_args = inspect.getfullargspec(SignalType.__init__).args
    sig_kwargs = {}
    sig_kwargs['onehot'] = make_onehot_signals(circuit=circuit, species_chosen=kwargs['inputs'])
    sig_kwargs['time_interval'] = kwargs.get('simulation', {}).get('dt')
    for k, v in kwargs.items():
        if k in signal_init_args:
            sig_kwargs[k] = v
    if any([k not in sig_kwargs for k in signal_init_args if k != 'self']):
        logging.warning(f'Signal init kwargs missing from {sig_kwargs}')
    return sig_kwargs

