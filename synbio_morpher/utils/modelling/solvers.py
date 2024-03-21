

def get_diffrax_solver(solver_type):
    import diffrax as dfx

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

    return solvers[solver_type]


def make_stepsize_controller(self, choice: str, **kwargs):
    """ The choice can be either log or piecewise """
    if choice == 'log':
        return self.make_log_stepcontrol(**kwargs)
    elif choice == 'piecewise':
        return make_piecewise_stepcontrol(t0=self.t0, t1=self.t1, dt0=self.dt0, dt1=self.dt1, **kwargs)
    elif choice == 'adaptive':
        return dfx.PIDController(rtol=1e-3, atol=1e-5)
    else:
        raise ValueError(
            f'The stepsize controller option `{choice}` is not available.')


def make_log_stepcontrol(self, upper_log: int = 3):
    num = 1000
    x = np.interp(np.logspace(0, upper_log, num=num), [
        1, np.power(10, upper_log)], [self.dt0, self.dt1])
    while np.cumsum(x)[-1] < self.t1:
        x = np.interp(np.logspace(0, upper_log, num=num), [
            1, np.power(10, upper_log)], [self.dt0, self.dt1])
        num += 1
    ts = np.cumsum(x)
    ts[0] = self.t0
    ts[-1] = self.t1
    return dfx.StepTo(ts)
