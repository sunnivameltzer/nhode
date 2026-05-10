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
from models import HamiltonianNN, NeuralODE, ICEncoder
from vector_fields import (
    vector_field_double_mass_spring,
    vector_field_hnn_potenergy,
    vector_field_hnn_totenergy,
    vector_field_node_phys,
    vector_field_node_vanilla
)
from utilities import Params, integrate_vector_field, to_canonical, split_train_val, save_model
from train import train_partially_observed_learning_ic


class Config:
    """Configuration object to hold training parameters."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def zero_total_momentum(v, masses):
    """
    Adjust velocities to ensure zero center-of-mass momentum.

    This enforces conservation of momentum by subtracting the
    center-of-mass velocity from each particle.

    Parameters
    ----------
    v : array [..., 3, 3]
        Velocities for 3 masses in 3D. For example, shape (N, 3, 3)
        with axes = (batch, body, xyz).
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

    k1, k2, k3, k4 = jax.random.split(key, 4)
    x0 = jax.random.uniform(k1, (N,), minval=0.2, maxval=0.8)
    x1 = jax.random.uniform(k3, (N,), minval=0.9, maxval=1.4)
    v0 = jax.random.uniform(k2, (N,), minval=-0.7, maxval=0.7)
    v1 = jax.random.uniform(k4, (N,), minval=-0.7, maxval=0.7)

    return jnp.stack([x0, x1, v0, v1], axis=-1)


def generate_training_data(config, system_params):

    key = jax.random.PRNGKey(config.seed)
    ts = jnp.linspace(0.0, config.trajectory_length, config.trajectory_steps)
    y0s = sample_ic_batch(key, system_params, config.n_trajectories)
    
    solve_batch = integrate_vector_field(ts=ts, term=vector_field_double_mass_spring)
    ys_all = solve_batch(model=None, params=system_params, y0s=y0s)
    
    ys_can = to_canonical(ys_all, system_params.m0, system_params.m1)
    
    key, subkey = jax.random.split(key)
    train_all, val_all = split_train_val(subkey, ys_can, config.train_fraction)
    
    print(f"Training samples: {train_all.shape[0]}")
    print(f"Validation samples: {val_all.shape[0]}")
    
    return train_all, val_all, ts, key



def initialize_model(config, key):

    if config.model == "hnn_pot":
        model = HamiltonianNN(
            key=key,
            input_dim=2,  
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        vector_field = vector_field_hnn_potenergy
        print(f"HNN Potential Energy Model")
        print(f"  Input dim: 2 (positions only)")
        print(f"  Hidden dim: {config.hidden_dim}, Depth: {config.depth}")
        print(f"  Assumes known masses for kinetic energy")
        
    elif config.model == "hnn_tot":
        model = HamiltonianNN(
            key=key,
            input_dim=4,  # Full state: (q, p)
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        vector_field = vector_field_hnn_totenergy
        print(f"HNN Total Energy Model")
        print(f"  Input dim: 4 (full state)")
        print(f"  Mass-agnostic approach")
        
    elif config.model == "node_phys":
        model = NeuralODE(
            key=key,
            input_dim=4,   # Full state input
            output_dim=2,   # Momentum derivatives output (dp0/dt, dp1/dt)
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        vector_field = vector_field_node_phys
        print(f"Neural ODE Model")
        print(f"  Input dim: 4, Output dim: 2")
    
    elif config.model == "node_vanilla":
        model = NeuralODE(
            key=key,
            input_dim=4,   # Full state input
            output_dim=4,   # Full state derivatives output (dx0/dt, dx1/dt, dp0/dt, dp1/dt)
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        vector_field = vector_field_node_vanilla
        print(f"Vanilla Neural ODE Model")
        print(f"  Input dim: 4, Output dim: 4")
        
    else:
        raise ValueError(f"Unknown model: {config.model}")
    
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))}")
    
    return model, vector_field

def initialize_encoder(config, key):

    encoder = ICEncoder(
        key=key,
        conditioning_steps=config.conditioning_steps,
        hidden_dim=config.encoder_hidden_dim,
        depth=config.encoder_depth,
    )
    
    print(f"IC Encoder Model")
    print(f"  Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(encoder, eqx.is_array)))}")
    
    return encoder


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


def train_model(config, 
                model, 
                encoder, 
                vector_field, 
                system_params, 
                train_all, 
                val_all, 
                ts, 
                key):
 
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
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_array)
    
    # Create ODE integrator for the chosen vector field
    run_model = integrate_vector_field(ts, term=vector_field)
    
    model_params, model_static, enc_params, enc_static, losses = train_partially_observed_learning_ic(
        batch_size=config.batch_size,
        epochs=config.epochs,
        key=key,
        run_hnn=run_model,
        system_params=system_params,
        model_params=model_params,
        model_static=model_static,
        enc_params=encoder_params,
        enc_static=encoder_static,
        train_all=train_all,
        val_all=val_all,
        learning_rate_hnn=config.lr_hnn,
        learning_rate_enc=config.lr_enc,
        lr_schedule=lr_schedule,
        conditioning_steps=config.conditioning_steps
        #penalty_positivity_weight=config.penalty_positivity_weight,   # regularization weight for inferred (x1_0, p1_0) positivity
    )
    
    save_losses(losses, dir_name=f"{config.savedir}/seed_{config.seed}", model_name=config.model)

    # Recombine parameters and structure
    model_final = eqx.combine(model_params, model_static)
    encoder_final = eqx.combine(enc_params, enc_static)

    return model_final, encoder_final


def save_trained_model(config, model, encoder, system_params):
    
    ckpt_dir = Path(f"{config.savedir}/seed_{config.seed}")
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(config, 'model_name') and config.model_name:
        model_name = config.model_name
        encoder_name = f"{model_name}_encoder"
    else:
        lr_suffix = "_lr_decay" if config.lr_decay else ""
        model_name = f"{config.model}_double_mass_spring_{config.epochs}ep{lr_suffix}"
        encoder_name = f"{model_name}_encoder"
    
    # Save model
    save_model(
        model_name=model_name,
        ckpt_dir=ckpt_dir,
        model=model,
        aux={"system_params": system_params}
    )

    # Save encoder
    save_model(
        model_name=encoder_name,
        ckpt_dir=ckpt_dir,
        model=encoder,
        aux={"system_params": system_params}
    )
    
    # Save metadata
    metadata = {
        "model_type": config.model,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate_hnn": config.lr_hnn,
        "learning_rate_enc": config.lr_enc,
        "lr_decay": config.lr_decay,
        "lr_decay_rate": config.lr_decay_rate if config.lr_decay else None,
        "lr_decay_epochs": config.lr_decay_epochs if config.lr_decay else None,
        "hidden_dim": config.hidden_dim,
        "depth": config.depth,
        "conditioning_steps": config.conditioning_steps,
        "encoder_hidden_dim": config.encoder_hidden_dim,
        "encoder_depth": config.encoder_depth,
        "system_params": {
            "m0": float(system_params.m0),
            "m1": float(system_params.m1),
            "k0": float(system_params.k0),
            "k1": float(system_params.k1),
            "L0": float(system_params.L0),
            "L1": float(system_params.L1)
        },
        "n_trajectories": config.n_trajectories,
        "trajectory_length": config.trajectory_length,
        "trajectory_steps": config.trajectory_steps,
        "train_fraction": config.train_fraction,
        "penalty_positivity_weight": config.penalty_positivity_weight,
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
    print("TWO LINEAR MASS-SPRING TRAINING, LEARNING INITIAL CONDITIONS")
    print("="*70)
    print(f"Model: {config.model}")
    print(f"Encoder conditioning steps: {config.conditioning_steps}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate HNN: {config.lr_hnn}")
    print(f"Learning rate Encoder: {config.lr_enc}")
    print(f"Seed: {config.seed}")
    
    # Initialize physical system parameters
    system_params = Params(
        m0=config.masses[0], m1=config.masses[1], k0=config.spring_constants[0], k1=config.spring_constants[1], L0=config.rest_lengths[0], L1=config.rest_lengths[1]
    )
    
    # Generate synthetic training data
    train_all, val_all, ts, key = generate_training_data(config, system_params)
    
    # Initialize neural network model
    key, subkey = jax.random.split(key)
    model, vector_field = initialize_model(config, subkey)

    key, subkey = jax.random.split(key)
    encoder = initialize_encoder(config, subkey)
    
    # Train the model
    key, subkey = jax.random.split(key)
    model_final, encoder_final = train_model(
        config, model, encoder, vector_field, system_params,
        train_all, val_all, ts, subkey
    )
    
    # Save trained model and metadata
    save_trained_model(
        config=config, 
        model=model_final, 
        encoder=encoder_final, 
        system_params=system_params)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
