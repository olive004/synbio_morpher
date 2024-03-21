

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
