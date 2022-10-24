from src.utils.circuit.agnostic_circuits.base_circuit import BaseCircuit

BaseCircuit.__init__.__doc__ = """
EXAMPLE FROM PATRICK KEDGER'S DIFFRAX DOCS (SaveAt function class)

**Main Arguments:**

- `t0`: If `True`, save the initial input `y0`.
- `t1`: If `True`, save the output at `t1`.
- `ts`: Some array of times at which to save the output.
- `steps`: If `True`, save the output at every step of the numerical solver.
- `dense`: If `True`, save dense output, that can later be evaluated at any part of
    the interval $[t_0, t_1]$ via `sol = diffeqsolve(...); sol.evaluate(...)`.

**Other Arguments:**

It is less likely you will need to use these options.

- `solver_state`: If `True`, save the internal state of the numerical solver at
    `t1`.
- `controller_state`: If `True`, save the internal state of the step size
    controller at `t1`.
- `made_jump`: If `True`, save the internal state of the jump tracker at `t1`.
"""