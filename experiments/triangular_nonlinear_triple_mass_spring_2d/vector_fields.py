import jax
import jax.numpy as jnp
from utilities import Params


def _pair_distance_extension(r_i, r_j, L_ij, eps=1e-12):
    """Return (dist, ext) where ext = dist - L_ij."""
    d = r_j - r_i
    dist = jnp.sqrt(jnp.dot(d, d) + eps)
    ext = dist - L_ij
    return dist, ext


def true_hamiltonian_triangular_mass_spring_2d(q, p, params: Params, eps=1e-12):
    """
    Canonical Hamiltonian for 3 masses in 2D connected in a triangle.

    q: (3,2) positions: [[x0,y0],[x1,y1],[x2,y2]]
    p: (3,2) momenta:   [[px0,py0],[px1,py1],[px2,py2]]

    Potential corresponds to force law F = k * ext^3 * u, i.e. V = (k/4)*ext^4.
    """
    r0, r1, r2 = q[0], q[1], q[2]
    p0, p1, p2 = p[0], p[1], p[2]

    # Kinetic energy: sum ||p_i||^2 / (2 m_i)
    T = (jnp.dot(p0, p0) / (2.0 * params.m0) +
         jnp.dot(p1, p1) / (2.0 * params.m1) +
         jnp.dot(p2, p2) / (2.0 * params.m2))

    # Spring extensions
    _, e01 = _pair_distance_extension(r0, r1, params.L01, eps=eps)
    _, e02 = _pair_distance_extension(r0, r2, params.L02, eps=eps)
    _, e12 = _pair_distance_extension(r1, r2, params.L12, eps=eps)

    # Potential energy: sum (k/4) * ext^4
    V = (params.k01 * (e01**4) / 4.0 +
         params.k02 * (e02**4) / 4.0 +
         params.k12 * (e12**4) / 4.0)

    return T + V


def vector_field_triangular_nonlinear_mass_spring_2d(t, y, args, eps=1e-12):
    """
    True-dynamics vector field in 2D from Hamilton's equations.

    State:
      y = [x0,y0,x1,y1,x2,y2, px0,py0,px1,py1,px2,py2]

    Returns:
      dy/dt = [dq/dt, dp/dt] where
        dq/dt =  dH/dp
        dp/dt = -dH/dq
    """
    model, params = args
    if model is not None:
        raise ValueError("True-dynamics vector field only (model must be None).")

    q = y[:6].reshape(3, 2)
    p = y[6:].reshape(3, 2)

    dH_dq, dH_dp = jax.grad(true_hamiltonian_triangular_mass_spring_2d, argnums=(0, 1))(
        q, p, params, eps
    )

    dqdt = dH_dp
    dpdt = -dH_dq

    return jnp.concatenate([dqdt.reshape(-1), dpdt.reshape(-1)])


def vector_field_hnn_potenergy(t, y, args):
    """
    y = [qx0,qy0,qx1,qy1,qx2,qy2,px0,py0,px1,py1,px2,py2]
    args = (model, params) where params is an instance of Params
    """
    model, params = args
    masses = jnp.array([params.m0, params.m0, params.m1, params.m1, params.m2, params.m2])

    q = y[:6]             # [qx0, qy0, qx1, qy1, qx2, qy2]
    p = y[6:]             # [px0, py0, px1, py1, px2, py2]

    # dq/dt = p/m
    dqdt = p / masses 

    # dp/dt = -∇_q U(q)
    U = lambda q_: model(q_)                     # scalar
    dUdq = jax.grad(U)(q)                        # shape (2,)
    dpdt = -dUdq
    return jnp.concatenate([dqdt, dpdt])


def vector_field_hnn_totenergy(t, y, args):
    """
    y = [qx0,qy0,qx1,qy1,qx2,qy2,px0,py0,px1,py1,px2,py2]
    args = (model, params) where params is an instance of Params
    """
    model, params = args # params is not used here since we assume the masses are not known
    
    H = lambda y_: model(y_)                     # scalar, total energy
    dHdy = jax.grad(H)(y)                        # shape (6,)

    dqdt = dHdy[6:]
    dpdt = -dHdy[:6]

    return jnp.concatenate([dqdt, dpdt])


def vector_field_node_phys(t, y, args):
    """
    y = [qx0,qy0,qx1,qy1,qx2,qy2,px0,py0,px1,py1,px2,py2]
    args = (model, params) where params is an instance of Params
    """
    model, params = args
    masses = jnp.array([params.m0, params.m0, params.m1, params.m1, params.m2, params.m2])

    q = y[:6]             # [qx0, qy0, qx1, qy1, qx2, qy2]
    p = y[6:]             # [px0, py0, px1, py1, px2, py2]

    dqdt = p / masses
    dpdt = model(y)
    return jnp.concatenate([dqdt, dpdt])

def vector_field_node_vanilla(t, y, args):
    """
    y = [qx0,qy0,qx1,qy1,qx2,qy2,px0,py0,px1,py1,px2,py2]
    args = (model, params) where params is an instance of Params
    """
    model, params = args

    dydt = model(y)
    return dydt

