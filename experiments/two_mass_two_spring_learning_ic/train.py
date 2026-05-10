import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import optax

from utilities import make_minibatches


def mse(a, b): 
    return jnp.mean((a - b) ** 2)

def create_optimizer(learning_rate, lr_schedule, n_train_samples, batch_size, epochs):

    if lr_schedule is None:
        return optax.adam(learning_rate=learning_rate), None
    
    schedule_type = lr_schedule.get('type', 'step_decay')
    decay_rate = lr_schedule.get('decay_rate', 0.5)
    decay_epochs = lr_schedule.get('decay_epochs', 200)
    
    steps_per_epoch = n_train_samples // batch_size
    decay_steps = decay_epochs * steps_per_epoch
    
    if schedule_type == 'exponential_decay':
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    elif schedule_type == 'step_decay':
        schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate,
            boundaries_and_scales={
                i * decay_steps: decay_rate 
                for i in range(1, epochs // decay_epochs + 1)
            }
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return optax.adam(learning_rate=schedule), schedule

def train_partially_observed_learning_ic(
        batch_size, 
        epochs, 
        key, 
        run_hnn, 
        system_params, 
        model_params, 
        model_static,
        enc_params, 
        enc_static,
        train_all, 
        val_all, 
        learning_rate_hnn=1e-3,
        learning_rate_enc=5e-3,
        lr_schedule=None,
        conditioning_steps=5, # number of observed steps to infer initial conditions from
        print_every=25
        #penalty_positivity_weight=0.0              # mild regularization on inferred (x1_0, p1_0)
        ):

    OBSERVED_INDICES = jnp.array([0, 2])  # Only observe x0 and p0 → indices 0 and 2 
    Q_LATENT_INDICES = jnp.array([1])     # Latent (unobserved) positions q1

    T = train_all.shape[1]
    if conditioning_steps < 1 or conditioning_steps > T:
        raise ValueError(f"conditioning_steps must be in [1, T], got conditioning_steps={conditioning_steps}, T={T}")
    
    val_initial_conditions = val_all[:, 0]

    optimizer_hnn, schedule_hnn = create_optimizer(
        learning_rate=learning_rate_hnn,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state_hnn = optimizer_hnn.init(model_params)
    
    optimizer_enc, schedule_enc = create_optimizer(
        learning_rate=learning_rate_enc,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state_enc = optimizer_enc.init(enc_params)

    # Train both HNN and encoder parameters together
    trainable = (model_params, enc_params)

    @eqx.filter_jit
    def predict_trajectories(model_params, model_static, initial_conditions):
        """
        Predict full state trajectories from initial conditions.
        """
        model = eqx.combine(model_params, model_static)
        return run_hnn(model, system_params, initial_conditions)

    @eqx.filter_value_and_grad
    def compute_loss(trainable, model_static, enc_static, batch_trajectories):
        model_params, enc_params = trainable

        encoder = eqx.combine(enc_params, enc_static)
        target_oberved = batch_trajectories[..., OBSERVED_INDICES]   # (B,T,2)

        # prefix for encoder: [B, K, 2] -> [B, 2K]
        conditioning_context = target_oberved[:, :conditioning_steps, :].reshape(target_oberved.shape[0], -1)

        # infer hidden initial (x1_0, p1_0): [B, 2]
        predicted_initial_conditions = jax.vmap(encoder)(conditioning_context)
        observed_initial_conditions = target_oberved[:, 0, :]

        # assemble full canonical y0 = [x0_0, x1_0, p0_0, p1_0]
        full_initial_condition = jnp.stack([observed_initial_conditions[:, 0], predicted_initial_conditions[:, 0], observed_initial_conditions[:, 1], predicted_initial_conditions[:, 1]], axis=-1)

        predicted_trajectories = predict_trajectories(model_params, model_static, full_initial_condition)  # [B, T, 4]
        predicted_observed = predicted_trajectories[..., OBSERVED_INDICES]        # (B,T,2) -> [x0, p0]
        
        data_loss = mse(predicted_observed, target_oberved)

        # Regularization to encourage positivity of inferred q1 trajectory (softplus penalty on negative values)
        # def neg_penalty_softplus_traj(q_latent, tau=0.01, p=2):
        #     v = jnn.softplus(-q_latent / tau)
        #     return jnp.mean(v**p)

        #q_latent = predicted_trajectories[..., Q_LATENT_INDICES]   # [B, T, 1]
        #penalty = neg_penalty_softplus_traj(q_latent, tau=0.01, p=2)

        loss = data_loss #+ penalty_positivity_weight * penalty

        return loss
    
    @eqx.filter_jit
    def training_step(trainable, model_static, enc_static, optimizer_state_hnn, optimizer_state_enc, batch_targets_full):
        loss, grads = compute_loss(trainable, model_static, enc_static, batch_targets_full)
        grads_hnn, grads_enc = grads
        model_params, enc_params = trainable

        updates_hnn, optimizer_state_hnn = optimizer_hnn.update(
            eqx.filter(grads_hnn, eqx.is_array),
            optimizer_state_hnn,
            params=eqx.filter(model_params, eqx.is_array),
        )
        model_params = eqx.apply_updates(model_params, updates_hnn)

        updates_enc, optimizer_state_enc = optimizer_enc.update(
            eqx.filter(grads_enc, eqx.is_array),
            optimizer_state_enc,
            params=eqx.filter(enc_params, eqx.is_array),
        )
        enc_params = eqx.apply_updates(enc_params, updates_enc)

        trainable = (model_params, enc_params)

        return trainable, optimizer_state_hnn, optimizer_state_enc, loss

    # --- Loss logging ---
    train_losses = []
    val_losses = []

    # Train HNN (partial-observation)
    key, subkey = jax.random.split(key)

    for epoch in range(1, epochs+1):
        key, subkey = jax.random.split(key)

        batch_losses = []

        for batch in make_minibatches(subkey, train_all, batch_size):   # full states as targets (we only use obs dims in loss)
            trainable, optimizer_state_hnn, optimizer_state_enc, train_loss = training_step(
                trainable, model_static, enc_static, optimizer_state_hnn, optimizer_state_enc, batch
            )
            batch_losses.append(train_loss)

        # Mean epoch train loss (convert JAX scalars -> Python float)
        train_loss_epoch = float(jnp.mean(jnp.stack(batch_losses)))
        train_losses.append(train_loss_epoch)

        model_params, enc_params = trainable
        # validation (same partial pipeline: encoder infers hidden IC)
        encoder = eqx.combine(enc_params, enc_static)
        val_batch = val_all[:batch_size]  

        obs_val = val_batch[..., OBSERVED_INDICES]  # [B, T, 2]
        prefix_flat = obs_val[:, :conditioning_steps, :].reshape(obs_val.shape[0], -1)
        z0_val = jax.vmap(encoder)(prefix_flat)
        y0_obs_val = obs_val[:, 0, :]
        y0_val_full = jnp.stack([y0_obs_val[:, 0], z0_val[:, 0], y0_obs_val[:, 1], z0_val[:, 1]], axis=-1)


        val_predictions = predict_trajectories(model_params, model_static, y0_val_full)
        val_loss = mse(val_predictions[..., OBSERVED_INDICES], obs_val)

        val_loss_epoch = float(val_loss)
        val_losses.append(val_loss_epoch)

        if epoch % 25 == 0 or epoch == 1:
            if schedule_hnn is not None:
                current_step = epoch * (len(train_all) // batch_size)
                current_lr = schedule_hnn(current_step)
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val MSE: {float(val_loss):.6f} | "
                    f"lr HNN: {float(current_lr):.2e}" 
                )
                # print(f"    Soft k (avg): {jnp.mean(k_soft):.2f} / {train_all.shape[1]}")
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val MSE: {float(val_loss):.6f}"
                )
            #print(f"[HNN]  epoch {epoch:3d} | train {float(train_loss):.6f} | val {float(val_loss):.6f}")

    losses = {
        "train": train_losses,
        "val": val_losses,
    }

    return model_params, model_static, enc_params, enc_static, losses