import inspect
from src.utils.signal.signals import Signal, AdaptationTarget, OscillatingSignal


def get_signal_type(signal_type: str) -> Signal:
    if signal_type == 'abstract':
        return Signal
    if signal_type == 'adaptation':
        return AdaptationTarget
    if signal_type == 'oscillation':
        return OscillatingSignal
    raise ValueError(f'Could not identify signal of type {signal_type}.')


def parse_sig_args(kwargs: dict):
    SignalType = get_signal_type(kwargs.get("signal_type"))
    signal_init_args = inspect.getfullargspec(SignalType.__init__).args
    sig_kwargs = {}
    for k, v in kwargs.items():
        if k in signal_init_args:
            sig_kwargs[k] = v
    return sig_kwargs

