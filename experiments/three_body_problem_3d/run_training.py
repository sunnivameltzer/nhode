import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
from pathlib import Path
import json
import yaml
from datetime import datetime
import numpy as np

# Local imports
from models import HamiltonianNN, NeuralODE
from vector_fields import (
    vector_field_three_body_plummer,
    vector_field_hnn_potenergy,
    vector_field_hnn_totenergy,
    vector_field_node_phys,
    vector_field_node_vanilla
)
from utilities import Params, integrate_vector_field, to_canonical, split_train_val, save_model
from train import train_partially_observed


class Config:
    """Configuration object to hold training parameters."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def zero_total_momentum(v, masses):
    """
    Adjust velocities to ensure zero center-of-mass momentum.

    Parameters
    ----------
    v : array [..., 3, 3]
        Velocities for 3 masses in 3D.
    masses : array [3]
        Mass values for each particle.

    Returns
    -------
    v_adjusted : array [..., 3, 3]
        Velocities with zero total momentum (same shape as input).
    """
    M = jnp.sum(masses)
    V_cm = jnp.tensordot(masses, v, axes=(0, -2)) / M  # (..., 3)
    return v - V_cm[..., None, :]


def sample_ic_batch(key: jax.random.PRNGKey, system_params: Params, N: int):
    """
    Sample initial conditions for the 3D three-body system.

    Returns
    -------
    ic_batch : array [N, 18]
        Initial conditions in format:
        [x0,y0,z0, x1,y1,z1, x2,y2,z2,
         vx0,vy0,vz0, vx1,vy1,vz1, vx2,vy2,vz2]
    """
    masses = jnp.array([system_params.m0, system_params.m1, system_params.m2])
    
    key0, key1, key2, key3, keyz = jax.random.split(key, 5)

    # Mass 0 near top
    x0 = jax.random.uniform(key0, (N,), minval=0.3, maxval=0.7)
    y0 = jax.random.uniform(key0, (N,), minval=1.0, maxval=1.4)

    # Mass 1 bottom-left region
    x1 = jax.random.uniform(key1, (N,), minval=0.1, maxval=0.4)
    y1 = jax.random.uniform(key1, (N,), minval=0.1, maxval=0.5)

    # Mass 2 bottom-right region
    x2 = jax.random.uniform(key2, (N,), minval=0.6, maxval=0.9)
    y2 = jax.random.uniform(key2, (N,), minval=0.1, maxval=0.5)

    # z-coordinates: small perturbations around 0 (near a plane)
    z0 = jax.random.uniform(keyz, (N,), minval=-0.1, maxval=0.1)
    z1 = jax.random.uniform(keyz, (N,), minval=-0.1, maxval=0.1)
    z2 = jax.random.uniform(keyz, (N,), minval=-0.1, maxval=0.1)

    # Stack positions: (N, 9)
    r = jnp.stack([x0, y0, z0,
                   x1, y1, z1,
                   x2, y2, z2], axis=-1)
    
    # Recenter center of mass to origin
    r_reshaped = r.reshape(N, 3, 3)               # (N, n_bodies, dim)
    total_mass = jnp.sum(masses)
    r_cm = (masses[None, :, None] * r_reshaped).sum(axis=1, keepdims=True) / total_mass
    r_reshaped = r_reshaped - r_cm                # shift COM to origin
    r = r_reshaped.reshape(N, 9)

    # Zero total momentum
    v = jax.random.uniform(key3, (N, 9), minval=-0.3, maxval=0.3)
    v_reshaped = v.reshape(N, 3, 3)
    v_zero = jax.vmap(zero_total_momentum, in_axes=(0, None))(v_reshaped, masses)
    v = v_zero.reshape(N, 9)

    # Return concatenated state: (N, 18)
    return jnp.concatenate([r, v], axis=-1)


def generate_training_data(config, system_params):

    key = jax.random.PRNGKey(config.seed)
    ts = jnp.linspace(0.0, config.trajectory_length, config.trajectory_steps)
    y0s = sample_ic_batch(key, system_params, config.n_trajectories)
    eps = config.eps
    
    solve_batch = integrate_vector_field(ts=ts, term=vector_field_three_body_plummer, eps=eps)
    ys_all = solve_batch(model=None, params=system_params, y0s=y0s)
    
    ys_can = to_canonical(ys_all, system_params.m0, system_params.m1, system_params.m2)
    
    key, subkey = jax.random.split(key)
    train_all, val_all = split_train_val(subkey, ys_can, config.train_fraction)
    
    print(f"Training samples: {train_all.shape[0]}")
    print(f"Validation samples: {val_all.shape[0]}")
    
    return train_all, val_all, ts, key



def initialize_model(config, key):
    
    if config.model == "hnn_pot":
        model = HamiltonianNN(
            key=key,
            input_dim=9,  
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            relative_distances=config.relative_distances
        )
        vector_field = vector_field_hnn_potenergy
        print(f"HNN Potential Energy Model")
        print(f"  Input dim: 9 (positions only)")
        print(f"  Hidden dim: {config.hidden_dim}, Depth: {config.depth}")
        print(f"  Assumes known masses for kinetic energy")
        print(f"  Relative distances: {config.relative_distances}")
        
    elif config.model == "hnn_tot":
        model = HamiltonianNN(
            key=key,
            input_dim=18,  # Full state: (q, p)
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            relative_distances=config.relative_distances
        )
        vector_field = vector_field_hnn_totenergy
        print(f"HNN Total Energy Model")
        print(f"  Input dim: 18 (full state)")
        print(f"  Mass-agnostic approach")
        print(f"  Relative distances: {config.relative_distances}")
        
    elif config.model == "node_phys":
        model = NeuralODE(
            key=key,
            input_dim=18,   # Full state input
            output_dim=9,   # Momentum derivatives output
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            relative_distances=config.relative_distances
        )
        vector_field = vector_field_node_phys
        print(f"Neural ODE Model")
        print(f"  Input dim: 18, Output dim: 9")
        print(f"  Relative distances: {config.relative_distances}")

    elif config.model == "node_vanilla":
        model = NeuralODE(
            key=key,
            input_dim=18,   # Full state input
            output_dim=18,   # Full state derivatives output
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            relative_distances=config.relative_distances
        )
        vector_field = vector_field_node_vanilla
        print(f"Neural ODE Vanilla Model")
        print(f"  Input dim: 18, Output dim: 18")
        print(f"  Relative distances: {config.relative_distances}")
        
    else:
        raise ValueError(f"Unknown model: {config.model}")
    
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))}")
    
    return model, vector_field


def save_losses(losses: dict, dir_name: str, model_name: str):
    """
    Saves losses to: dir_name / f"{model_name}_losses.npz"

    losses: e.g. {"train": [...], "val": [...], "lr": [...] (optional)}
    Uses jax.numpy for array creation; uses numpy for file I/O.
    """
    out_dir = Path(dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_losses.npz"

    # lists -> jax arrays
    losses_jax = {k: jnp.asarray(v) for k, v in losses.items()}

    # device -> host numpy arrays
    losses_host = {k: np.asarray(jax.device_get(arr)) for k, arr in losses_jax.items()}

    np.savez_compressed(path, **losses_host)
    return


def train_model(config, model, vector_field, system_params, train_all, val_all, ts, key):
 
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Prepare learning rate schedule if requested
    lr_schedule = None
    if config.lr_decay is not False:
        lr_schedule = {
            'type': config.lr_decay,
            'decay_rate': config.lr_decay_rate,
            'decay_epochs': config.lr_decay_epochs
        }
        
    model_params, model_static = eqx.partition(model, eqx.is_array)
    
    # Create ODE integrator for the chosen vector field
    run_model = integrate_vector_field(ts, term=vector_field)
    
    model_params, model_static, losses = train_partially_observed(
        batch_size=config.batch_size,
        epochs=config.epochs,
        key=key,
        run_hnn=run_model,
        system_params=system_params,
        model_params=model_params,
        model_static=model_static,
        train_all=train_all,
        val_all=val_all,
        learning_rate=config.lr,
        lr_schedule=lr_schedule
    )
    
    save_losses(losses, dir_name=f"{config.savedir}/seed_{config.seed}", model_name=config.model)

    # Recombine parameters and structure
    model_final = eqx.combine(model_params, model_static)
    return model_final


def save_trained_model(config, model, system_params):
    
    ckpt_dir = Path(f"{config.savedir}/seed_{config.seed}")
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(config, 'model_name') and config.model_name:
        model_name = config.model_name
    else:
        coord_type = "rel_dist" if config.relative_distances else "abs_pos"
        lr_suffix = "_lr_decay" if config.lr_decay else ""
        model_name = f"{config.model}_three_body_{coord_type}_{config.epochs}ep{lr_suffix}"
    
    # Save model
    save_model(
        model_name=model_name,
        ckpt_dir=ckpt_dir,
        model=model,
        aux={"system_params": system_params}
    )
    
    # Save metadata
    metadata = {
        "model_type": config.model,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "lr_decay": config.lr_decay,
        "lr_decay_rate": config.lr_decay_rate if config.lr_decay else None,
        "lr_decay_epochs": config.lr_decay_epochs if config.lr_decay else None,
        "hidden_dim": config.hidden_dim,
        "depth": config.depth,
        "relative_distances": config.relative_distances,
        "system_params": {
            "m0": float(system_params.m0),
            "m1": float(system_params.m1),
            "m2": float(system_params.m2),
            "G": float(system_params.G),
        },
        "eps": config.eps,
        "n_trajectories": config.n_trajectories,
        "trajectory_length": config.trajectory_length,
        "trajectory_steps": config.trajectory_steps,
        "train_fraction": config.train_fraction,
        "timestamp": timestamp,
        "seed": config.seed,
    }
    
    metadata_path = ckpt_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {ckpt_dir / model_name}")
    print(f"Metadata saved to: {metadata_path}")


def load_config(config_path):

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def main(config_path=None):
    """
    Main training pipeline orchestrator.
    
    Executes the complete training workflow:
    1. Load configuration from YAML
    2. Initialize system parameters
    3. Generate training data
    4. Initialize model
    5. Train model
    6. Save model and metadata
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. If None, reads from command line arguments.
        
    Raises
    ------
    SystemExit
        If no config file is provided and script is run from command line
    ValueError
        If config contains invalid model type or parameters
    """
    # Load config from file or command line
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python run_training.py <config_file.yaml>")
            sys.exit(1)
        config_path = sys.argv[1]
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    print("\n" + "="*70)
    print("THREE BODY PROBLEM TRAINING")
    print("="*70)
    print(f"Model: {config.model}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Seed: {config.seed}")
    print(f"Epsilon (softening): {config.eps}")
    
    # Initialize physical system parameters
    system_params = Params(
        m0=config.mass[0], m1=config.mass[1], m2=config.mass[2], G=config.G
    )
    
    # Generate synthetic training data
    train_all, val_all, ts, key = generate_training_data(config, system_params)
    
    # Initialize neural network model
    key, subkey = jax.random.split(key)
    model, vector_field = initialize_model(config, subkey)
    
    # Train the model
    key, subkey = jax.random.split(key)
    model_final = train_model(
        config, model, vector_field, system_params,
        train_all, val_all, ts, subkey
    )
    
    # Save trained model and metadata
    save_trained_model(config, model_final, system_params)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
