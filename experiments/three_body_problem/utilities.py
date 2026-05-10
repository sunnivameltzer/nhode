import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
from pathlib import Path
from typing import Any, Mapping, Optional, Union

class Params(eqx.Module):
    m0: float
    m1: float
    m2: float      
    G: float         # gravitational constant


def integrate_vector_field(ts, term, eps=0.0, rtol=1e-5, atol=1e-6, max_steps=16384):
    ts_host = jnp.asarray(ts)
    t0 = float(ts_host[0])
    t1 = float(ts_host[-1])

    term = dfx.ODETerm(term)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=ts_host)
    controller = dfx.PIDController(rtol=rtol, atol=atol)

    def run_batch(model, params: Params, y0s): 
        def solve_one(y0):
            sol = dfx.diffeqsolve(term, solver, 
                                  t0=t0, t1=t1, dt0=None,
                                  y0=y0, 
                                  args=(model, params, eps),
                                  saveat=saveat, 
                                  stepsize_controller=controller,
                                  max_steps=max_steps)
            return sol.ys
        return jax.vmap(solve_one)(y0s)
    return run_batch


# System specific
def to_canonical(ys, m0, m1, m2):
    x0, y0, z0 = ys[..., 0], ys[..., 1], ys[..., 2]
    x1, y1, z1 = ys[..., 3], ys[..., 4], ys[..., 5]
    x2, y2, z2 = ys[..., 6], ys[..., 7], ys[..., 8]
    vx0, vy0, vz0 = ys[..., 9],  ys[...,10], ys[...,11]
    vx1, vy1, vz1 = ys[...,12], ys[...,13], ys[...,14]
    vx2, vy2, vz2 = ys[...,15], ys[...,16], ys[...,17]

    # Momenta p = m v
    px0, py0, pz0 = m0 * vx0, m0 * vy0, m0 * vz0
    px1, py1, pz1 = m1 * vx1, m1 * vy1, m1 * vz1
    px2, py2, pz2 = m2 * vx2, m2 * vy2, m2 * vz2

    # Return canonical vector: [positions, momenta]
    return jnp.stack([
        x0, y0, z0,
        x1, y1, z1,
        x2, y2, z2,
        px0, py0, pz0,
        px1, py1, pz1,
        px2, py2, pz2,
    ], axis=-1)

def split_train_val(key, data, frac=0.85):
    N = data.shape[0]
    idx = jax.random.permutation(key, N)
    ntr = int(frac * N)
    return data[idx[:ntr]], data[idx[ntr:]]

def make_minibatches(key, ys, batch):
    N = ys.shape[0]
    permutation = jax.random.permutation(key, N)
    for i in range(0, N, batch):
        yield ys[permutation[i:i+batch]]

def save_model(
    model_name: str,
    ckpt_dir: Union[str, Path],
    model: Any,
    aux: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save an Equinox model and optional aux data (e.g., masses, step, opt_state).
    Files written:
      <ckpt_dir>/<model_name>_model.eqx
      <ckpt_dir>/<model_name>_aux.eqx  (only if aux is provided)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / f"{model_name}_model.eqx"
    eqx.tree_serialise_leaves(model_path, model)

    if aux is not None:
        aux_path = ckpt_dir / f"{model_name}_aux.eqx"
        eqx.tree_serialise_leaves(aux_path, aux)

def load_model(
    model_name: str,
    ckpt_dir: Union[str, Path],
    model_like: Any,
    aux_like: Optional[Mapping[str, Any]] = None,
):
    """
    Load an Equinox model and optional aux data.

    Args:
        model_name: Name used when saving.
        ckpt_dir: Directory containing the checkpoint files.
        model_like: A 'skeleton' instance with the SAME architecture
                    (e.g., PotentialNN(key, hidden=..., depth=...)).
        aux_like:  A structure with the same keys/shapes/dtypes as what was saved
                   (e.g., {"masses": jnp.zeros(2), "step": jnp.array(0)}).
    Returns:
        (model, aux) if aux_like is provided, else just model.
    """
    ckpt_dir = Path(ckpt_dir)

    if model_name.startswith("three_body_problem_hnn_pot"):
        model_name_key = "three_body_problem_hnn_pot"
    elif model_name.startswith("three_body_problem_hnn_tot"):
        model_name_key = "three_body_problem_hnn_tot"
    elif model_name.startswith("three_body_problem_node_phys"):
        model_name_key = "three_body_problem_node_phys"
    elif model_name.startswith("three_body_problem_node_vanilla"):
        model_name_key = "three_body_problem_node_vanilla"
    else:
        model_name_key = model_name

    model_path = ckpt_dir / f"{model_name_key}_model.eqx"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = eqx.tree_deserialise_leaves(model_path, model_like)

    if aux_like is not None:
        aux_path = ckpt_dir / f"{model_name_key}_aux.eqx"
        if not aux_path.exists():
            raise FileNotFoundError(aux_path)
        aux = eqx.tree_deserialise_leaves(aux_path, aux_like)
        return model, aux

    return model