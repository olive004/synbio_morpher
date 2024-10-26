import diffrax as dfx
import numpy as np
from synbio_morpher.srv.parameter_prediction.simulator import make_piecewise_stepcontrol


def get_diffrax_solver(solver_type):

    solvers = {
        'Tsit5': dfx.Tsit5,
        'Dopri5': dfx.Dopri5,
        'Dopri8': dfx.Dopri8,
        'Heun': dfx.Heun,
        'Euler': dfx.Euler,
        'Midpoint': dfx.Midpoint,
        'Ralston': dfx.Ralston,
        'Bosh3': dfx.Bosh3,
        'ImplicitEuler': dfx.ImplicitEuler,
        'Kvaerno3': dfx.Kvaerno3,
        'Kvaerno4': dfx.Kvaerno4,
        'Kvaerno5': dfx.Kvaerno5,
        'Sil3': dfx.Sil3,
        'KenCarp3': dfx.KenCarp3,
        'KenCarp4': dfx.KenCarp4,
        'KenCarp5': dfx.KenCarp5,
        'SemiImplicitEuler': dfx.SemiImplicitEuler,
        'ReversibleHeun': dfx.ReversibleHeun,
        'LeapfrogMidpoint': dfx.LeapfrogMidpoint
    }

    assert solver_type in list(solvers.keys(
    )), f'Diffrax solver option {solver_type} not found. See https://docs.kidger.site/diffrax/api/solvers/ode_solvers/ for valid solvers.'

    return solvers[solver_type]()


def make_stepsize_controller(t0, t1, dt0, dt1, choice: str, **kwargs):
    """ The choice can be either log or piecewise """
    if choice == 'log':
        return make_log_stepcontrol(t0, t1, dt0, dt1, **kwargs)
    elif choice == 'piecewise':
        return make_piecewise_stepcontrol(t0=t0, t1=t1, dt0=dt0, dt1=dt1, **kwargs)
    elif choice == 'adaptive':
        return dfx.PIDController(rtol=1e-7, atol=1e-9)
    else:
        raise ValueError(
            f'The stepsize controller option `{choice}` is not available.')


def make_log_stepcontrol(t0, t1, dt0, dt1, upper_log: int = 3):
    num = 1000
    x = np.interp(np.logspace(0, upper_log, num=num), [
        1, np.power(10, upper_log)], [dt0, dt1])
    while np.cumsum(x)[-1] < t1:
        x = np.interp(np.logspace(0, upper_log, num=num), [
            1, np.power(10, upper_log)], [dt0, dt1])
        num += 1
    ts = np.cumsum(x)
    ts[0] = t0
    ts[-1] = t1
    return dfx.StepTo(ts)
