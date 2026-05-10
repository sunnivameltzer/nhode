import jax
import jax.numpy as jnp


def _pair_separation_direction(r_i, r_j):
    """
    Return separation vector d = r_i - r_j and its squared norm.
    """
    d = r_i - r_j                          # vector from j to i
    r2 = jnp.dot(d, d)                     # squared distance + eps
    return d, r2

def vector_field_three_body_plummer(t, y, args):
    """
    Three-body Hamiltonian system with pairwise potential
        U(q) = - sum_{i<j} G m_i m_j / |q_i - q_j|

    State y:
        [x0,y0,z0, x1,y1,z1, x2,y2,z2,
         vx0,vy0,vz0, vx1,vy1,vz1, vx2,vy2,vz2]

    args = (model, params) where:
        - model must be None for true dynamics
        - params has attributes: m0, m1, m2, G
    """
    model, params, eps = args
    if model is not None:
        raise ValueError("True-dynamics vector field only (model must be None).")

    G = params.G

    # Unpack state
    x0, y0, z0, x1, y1, z1, x2, y2, z2, \
    vx0, vy0, vz0, vx1, vy1, vz1, vx2, vy2, vz2 = y

    # Position vectors
    r0 = jnp.array([x0, y0, z0])
    r1 = jnp.array([x1, y1, z1])
    r2 = jnp.array([x2, y2, z2])

    # Velocity vectors
    v0 = jnp.array([vx0, vy0, vz0])
    v1 = jnp.array([vx1, vy1, vz1])
    v2 = jnp.array([vx2, vy2, vz2])

    # Masses
    m0 = params.m0
    m1 = params.m1
    m2 = params.m2

    # Pair separations
    d01, r2_01 = _pair_separation_direction(r0, r1)
    d02, r2_02 = _pair_separation_direction(r0, r2)
    d12, r2_12 = _pair_separation_direction(r1, r2)

    # Distances and 1/r^3 factors (for 1/r potential → 1/r^2 force along d̂,
    # but since d is not normalized, we get d / r^3)
    # r01 = jnp.sqrt(r2_01)
    # r02 = jnp.sqrt(r2_02)
    # r12 = jnp.sqrt(r2_12)

    # Plummer potential softening
    r01 = (r2_01 + eps**2)**(3/2)
    r02 = (r2_02 + eps**2)**(3/2)
    r12 = (r2_12 + eps**2)**(3/2)  

    # inv_r3_01 = 1.0 / (r2_01 * r01)
    # inv_r3_02 = 1.0 / (r2_02 * r02)
    # inv_r3_12 = 1.0 / (r2_12 * r12)

    # Forces from each pair (i,j) on i:
    # F_ij_on_i = - G m_i m_j * d_ij / r_ij^3
    F01_on_0 = -G * m0 * m1 * d01 / r01
    F01_on_1 = -F01_on_0

    F02_on_0 = -G * m0 * m2 * d02 / r02
    F02_on_2 = -F02_on_0

    F12_on_1 = -G * m1 * m2 * d12 / r12
    F12_on_2 = -F12_on_1

    # Net forces
    F0 = F01_on_0 + F02_on_0
    F1 = F01_on_1 + F12_on_1
    F2 = F02_on_2 + F12_on_2

    # Accelerations
    a0 = F0 / m0
    a1 = F1 / m1
    a2 = F2 / m2

    # Time derivatives of positions and velocities
    dxdt = jnp.array([
        v0[0], v0[1], v0[2],
        v1[0], v1[1], v1[2],
        v2[0], v2[1], v2[2],
    ])

    dvdt = jnp.array([
        a0[0], a0[1], a0[2],
        a1[0], a1[1], a1[2],
        a2[0], a2[1], a2[2],
    ])

    return jnp.concatenate([dxdt, dvdt])


def vector_field_three_body_repulsive(t, y, args, eps=1e-3):
    model, params = args
    if model is not None:
        raise ValueError("True-dynamics vector field only (model must be None).")

    G = params.G

    # Unpack state
    x0, y0, z0, x1, y1, z1, x2, y2, z2, \
    vx0, vy0, vz0, vx1, vy1, vz1, vx2, vy2, vz2 = y

    # Position vectors
    r0 = jnp.array([x0, y0, z0])
    r1 = jnp.array([x1, y1, z1])
    r2 = jnp.array([x2, y2, z2])

    # Velocity vectors
    v0 = jnp.array([vx0, vy0, vz0])
    v1 = jnp.array([vx1, vy1, vz1])
    v2 = jnp.array([vx2, vy2, vz2])

    # Masses
    m0 = params.m0
    m1 = params.m1
    m2 = params.m2

    # Pair separations
    d01, r2_01 = _pair_separation_direction(r0, r1)
    d02, r2_02 = _pair_separation_direction(r0, r2)
    d12, r2_12 = _pair_separation_direction(r1, r2)

    r01 = (r2_01 + eps**2)**(3/2)
    r02 = (r2_02 + eps**2)**(3/2)
    r12 = (r2_12 + eps**2)**(3/2)  

    # Add small repulsive term at short distances
    repulsive_01 = 1 / (r2_01**0.5)
    repulsive_02 = 1 / (r2_02**0.5)
    repulsive_12 = 1 / (r2_12**0.5)

    # Forces from each pair (i,j) on i:
    # F_ij_on_i = - G m_i m_j * d_ij / r_ij^3
    F01_on_0 = -G * m0 * m1 * d01 / r01 + repulsive_01 * d01
    F01_on_1 = -F01_on_0

    F02_on_0 = -G * m0 * m2 * d02 / r02 + repulsive_02 * d02
    F02_on_2 = -F02_on_0

    F12_on_1 = -G * m1 * m2 * d12 / r12 + repulsive_12 * d12
    F12_on_2 = -F12_on_1

    # Net forces
    F0 = F01_on_0 + F02_on_0
    F1 = F01_on_1 + F12_on_1
    F2 = F02_on_2 + F12_on_2

    # Accelerations
    a0 = F0 / m0
    a1 = F1 / m1
    a2 = F2 / m2

    # Time derivatives of positions and velocities
    dxdt = jnp.array([
        v0[0], v0[1], v0[2],
        v1[0], v1[1], v1[2],
        v2[0], v2[1], v2[2],
    ])

    dvdt = jnp.array([
        a0[0], a0[1], a0[2],
        a1[0], a1[1], a1[2],
        a2[0], a2[1], a2[2],
    ])

    return jnp.concatenate([dxdt, dvdt])


def vector_field_hnn_potenergy(t, y, args):
    """
    y = [qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2,px0,py0,pz0,px1,py1,pz1,px2,py2,pz2]
    args = (model, params) where params is an instance of Params
    """
    model, params, eps = args
    masses = jnp.array([
        params.m0, params.m0, params.m0, 
        params.m1, params.m1, params.m1, 
        params.m2, params.m2, params.m2
    ])
    
    q = y[:9]             # [qx0, qy0, qz0, qx1, qy1, qz1, qx2, qy2, qz2]
    p = y[9:]             # [px0, py0, pz0, px1, py1, pz1, px2, py2, pz2]

    # dq/dt = p/m
    dqdt = p / masses 

    # dp/dt = -∇_q U(q)
    # Pass masses to model for proper preprocessing
    U = lambda q_: model(q_)                     # scalar
    dUdq = jax.grad(U)(q)                        # shape (9,)
    dpdt = -dUdq
    return jnp.concatenate([dqdt, dpdt])


def vector_field_hnn_totenergy(t, y, args):
    """
    y = [qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2,px0,py0,pz0,px1,py1,pz1,px2,py2,pz2]
    args = (model, params) where params is an instance of Params
    """
    model, params, eps = args # params is not used here since we assume the masses are not known
    
    H = lambda y_: model(y_)                     # scalar, total energy
    dHdy = jax.grad(H)(y)                        # shape (18,)

    dqdt = dHdy[9:]
    dpdt = -dHdy[:9]

    return jnp.concatenate([dqdt, dpdt])


def vector_field_node_phys(t, y, args):
    """
    y = [qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2,px0,py0,pz0,px1,py1,pz1,px2,py2,pz2]
    args = (model, params) where params is an instance of Params
    """
    model, params, eps = args
    masses = jnp.array([
        params.m0, params.m0, params.m0, 
        params.m1, params.m1, params.m1, 
        params.m2, params.m2, params.m2
    ])
    
    q = y[:9]             # [qx0, qy0, qz0, qx1, qy1, qz1, qx2, qy2, qz2]
    p = y[9:]             # [px0, py0, pz0, px1, py1, pz1, px2, py2, pz2]

    dqdt = p / masses
    dpdt = model(y)
    return jnp.concatenate([dqdt, dpdt])


def vector_field_node_vanilla(t, y, args):
    """
    y = [qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2,px0,py0,pz0,px1,py1,pz1,px2,py2,pz2]
    args = (model, params) where params is an instance of Params
    """
    model, params, eps = args

    dydt = model(y)  # dydt = [dq0/dt, dq1/dt, dp0/dt, dp1/dt]
    
    return dydt

