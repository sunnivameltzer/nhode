import jax
import jax.numpy as jnp
from utilities import Params


def vector_field_double_mass_spring(t, y, args):
    model, params = args                        # model = None for true dynamics
    
    if model != None:
        raise ValueError("This vector field is only for true dynamics, not learned models.")
    
    x0, x1, v0, v1 = y                          # unpack state (x are positions and v are velocities)
    ext0 = x0 - params.L0                       # spring extensions
    ext1 = (x1 - x0) - params.L1
    F0 = -params.k0 * ext0 + params.k1 * ext1   # forces
    F1 = -params.k1 * ext1
    return jnp.array([v0, v1, F0 / params.m0, F1 / params.m1]) #[dx0/dt, dx1/dt, dv0/dt, dv1/dt]


def vector_field_hnn_potenergy(t, y, args):
    """
    y = [q0, q1, p0, p1]
    args = (model, params) where params is an instance of Params
    """
    model, params = args
    masses = jnp.array([params.m0, params.m1])

    q = y[:2]             # [q0, q1]
    p = y[2:]             # [p0, p1]

    # dq/dt = p/m
    dqdt = p / masses

    # dp/dt = -∇_q U(q)
    U = lambda q_: model(q_)                     # scalar
    dUdq = jax.grad(U)(q)                        # shape (2,)
    dpdt = -dUdq
    return jnp.concatenate([dqdt, dpdt])


def vector_field_hnn_totenergy(t, y, args):
    """
    y = [q0, q1, p0, p1]
    args = (model, params) where params is an instance of Params
    """
    model, params = args # params is not used here since we assume the masses are not known
    
    H = lambda y_: model(y_)                     # scalar, total energy
    dHdy = jax.grad(H)(y)                        # shape (4,)

    dqdt = dHdy[2:]
    dpdt = -dHdy[:2]

    return jnp.concatenate([dqdt, dpdt])


def vector_field_node_phys(t, y, args):
    """
    y = [q0, q1, p0, p1]
    args = (model, params) where params is an instance of Params
    """
    model, params = args
    masses = jnp.array([params.m0, params.m1])

    q = y[:2]             # [q0, q1]
    p = y[2:]             # [p0, p1]

    dqdt = p / masses
    dpdt = model(y)
    return jnp.concatenate([dqdt, dpdt])


def vector_field_node_vanilla(t, y, args):
    """
    y = [q0, q1, p0, p1]
    args = (model, params) where params is an instance of Params
    """
    model, params = args

    dydt = model(y)  # dydt = [dq0/dt, dq1/dt, dp0/dt, dp1/dt]
    
    return dydt