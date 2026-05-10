import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from utilities import make_minibatches


def mse(predictions, targets):
    """Mean squared error loss function."""
    return jnp.mean((predictions - targets) ** 2)


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


def prefix_truncation_loss(predicted, target, alpha=10.0, lam=0.1):
    """
    Chaos-aware prefix truncation loss.

    predicted, target: shape (B, T, d)
    """
    e_t = jnp.mean((predicted - target) ** 2, axis=-1)  # (B, T)

    B, T = e_t.shape
    ks = jnp.arange(1, T + 1)              # (T,)
    cumsums = jnp.cumsum(e_t, axis=1)      # (B, T)
    m_k = cumsums / ks[None, :]            # (B, T)

    penalty = lam * (T - ks) / T           # (T,)
    ell_k = m_k + penalty[None, :]         # (B, T)

    loss = - (1 / alpha) * jnp.log(jnp.mean(jnp.exp(-alpha * ell_k), axis=1))
    
    # jax.debug.print("k value for minimum ell: {val}", val=jnp.argmin(ell_k, axis=1).mean())
    # jax.debug.print("k value for maximum ell: {val}", val=jnp.argmax(ell_k, axis=1).mean())
    # jax.debug.print("min ell_k: {val}", val=jnp.min(ell_k, axis=1).mean())
    # jax.debug.print("max ell_k: {val}", val=jnp.max(ell_k, axis=1).mean())
    # #jax.debug.print("minimum m_k: {val}", val=jnp.argmin(m_k[:, 1:], axis=1).mean())
    
    return jnp.mean(loss)


# ---------------------------------------------------------------------------------------
# Training function for a partially observed system
# ---------------------------------------------------------------------------------------

def train_partially_observed(
    batch_size, 
    epochs, 
    key, 
    run_hnn, 
    system_params, 
    model_params, 
    model_static, 
    train_all, 
    val_all, 
    learning_rate=1e-3,
    lr_schedule=None,
    print_every=25
):
    # Observed state indices for triangular system (masses 0 and 2)
    # State layout: [qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2, px0,py0,pz0,px1,py1,pz1,px2,py2,pz2]
    # Observe: [qx0,qy0,qz0,px0,py0,pz0, qx2,qy2,qz2,px2,py2,pz2] = indices [0,1,2,9,10,11,6,7,8,15,16,17]
    OBSERVED_INDICES = jnp.array([0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17])
    
    val_initial_conditions = val_all[:, 0]
    
    optimizer, schedule = create_optimizer(
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state = optimizer.init(model_params)
    
    @eqx.filter_jit
    def predict_trajectories(model_params, model_static, initial_conditions):
        """
        Predict full state trajectories from initial conditions.
        """
        model = eqx.combine(model_params, model_static)
        return run_hnn(model, system_params, initial_conditions)
    
    @eqx.filter_value_and_grad
    def compute_loss(model_params, model_static, batch_trajectories):
        """
        Compute MSE loss on observed state components only.
        """
        initial_conditions = batch_trajectories[:, 0]
        predicted_trajectories = predict_trajectories(
            model_params, model_static, initial_conditions
        )

        predicted_observed = predicted_trajectories[..., OBSERVED_INDICES]
        target_observed = batch_trajectories[..., OBSERVED_INDICES]
        
        return mse(predicted_observed, target_observed)
    
    @eqx.filter_jit
    def training_step(model_params, model_static, optimizer_state, batch_trajectories):
        """Perform one gradient update step."""
        loss, gradients = compute_loss(model_params, model_static, batch_trajectories)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )
        model_params = eqx.apply_updates(model_params, updates)
        return model_params, optimizer_state, loss
    
    # Training loop
    if lr_schedule:
        print(f"LR schedule: {lr_schedule['type']}, decay={lr_schedule['decay_rate']} every {lr_schedule['decay_epochs']} epochs")
    
    # --- Loss logging ---
    train_losses = []
    val_losses = []
    lrs = []  # will stay empty if schedule is None

    for epoch in range(1, epochs + 1):
        # Shuffle training data for this epoch
        key, subkey = jax.random.split(key)
        
        batch_losses = []
        # Train on all batches
        for batch in make_minibatches(subkey, train_all, batch_size):
            model_params, optimizer_state, train_loss = training_step(
                model_params, model_static, optimizer_state, batch
            )
            batch_losses.append(train_loss)
        
        # Mean epoch train loss (convert JAX scalars -> Python float)
        train_loss_epoch = float(jnp.mean(jnp.stack(batch_losses)))
        train_losses.append(train_loss_epoch)

        # Compute validation loss (on a subset for speed)
        val_predictions = predict_trajectories(
            model_params, model_static, val_initial_conditions[:batch_size]
        )
        val_loss = mse(
            val_predictions[..., OBSERVED_INDICES],
            val_all[:batch_size, ..., OBSERVED_INDICES]
        )

        val_loss_epoch = float(val_loss)
        val_losses.append(val_loss_epoch)
        
        # --- track LR  ---
        current_lr = None
        if schedule is not None:
            current_step = epoch * (len(train_all) // batch_size)
            current_lr = float(schedule(current_step))
            lrs.append(current_lr)

        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            if schedule is not None:
                current_step = epoch * (len(train_all) // batch_size)
                current_lr = schedule(current_step)
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f} | "
                    f"LR: {float(current_lr):.2e}"
                )
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f}"
                )
    
    losses = {
        "train": train_losses,
        "val": val_losses,
    }
    if schedule is not None:
        losses["lr"] = lrs

    print(f"\nTraining complete!")
    return model_params, model_static, losses


# ---------------------------------------------------------------------------------------
# Almost same training function as above, but with the chaos-aware prefix truncation loss
# ---------------------------------------------------------------------------------------

def train_partially_observed_chaos_aware(
    batch_size, 
    epochs, 
    key, 
    run_hnn, 
    system_params, 
    model_params, 
    model_static, 
    train_all, 
    val_all, 
    learning_rate=1e-3,
    lr_schedule=None,
    print_every=25,
    alpha=10.0,      # softmin sharpness
    lam=0.1,         # truncation penalty
):
    # Observed state indices 
    OBSERVED_INDICES = jnp.array([0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17])
    
    val_initial_conditions = val_all[:, 0]
    
    optimizer, schedule = create_optimizer(
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state = optimizer.init(model_params)
    
    @eqx.filter_jit
    def predict_trajectories(model_params, model_static, initial_conditions):
        """
        Predict full state trajectories from initial conditions.
        """
        model = eqx.combine(model_params, model_static)
        return run_hnn(model, system_params, initial_conditions)
    
    @eqx.filter_value_and_grad
    def compute_loss(model_params, model_static, batch_trajectories):
        """
        Chaos-aware loss on observed state components only.
        """
        initial_conditions = batch_trajectories[:, 0]
        predicted_trajectories = predict_trajectories(
            model_params, model_static, initial_conditions
        )

        # (B, T, d_obs)
        predicted_observed = predicted_trajectories[..., OBSERVED_INDICES]
        target_observed = batch_trajectories[..., OBSERVED_INDICES]
        
        # Use prefix truncation instead of plain MSE:
        loss = prefix_truncation_loss(
            predicted_observed,
            target_observed,
            alpha=alpha,
            lam=lam,
        )
        return loss
    
    @eqx.filter_jit
    def training_step(model_params, model_static, optimizer_state, batch_trajectories):
        """Perform one gradient update step."""
        loss, gradients = compute_loss(model_params, model_static, batch_trajectories)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )
        model_params = eqx.apply_updates(model_params, updates)

        # added this for debugging: print gradient stats
        # grad_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree_leaves(gradients)))

        # jax.debug.print("Global grad norm: {x}", x=grad_norm)
        #------------------------------------------

        return model_params, optimizer_state, loss
    
    # Training loop
    if lr_schedule:
        print(f"LR schedule: {lr_schedule['type']}, decay={lr_schedule['decay_rate']} every {lr_schedule['decay_epochs']} epochs")
    
    for epoch in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        
        for batch in make_minibatches(subkey, train_all, batch_size):
            model_params, optimizer_state, train_loss = training_step(
                model_params, model_static, optimizer_state, batch
            )
        
        # Validation: you can keep plain MSE here to monitor rollout quality
        val_predictions = predict_trajectories(
            model_params, model_static, val_initial_conditions[:batch_size]
        )
        val_loss = mse(
            val_predictions[..., OBSERVED_INDICES],
            val_all[:batch_size, ..., OBSERVED_INDICES]
        )
        
        if epoch % print_every == 0 or epoch == 1:
            if schedule is not None:
                current_step = epoch * (len(train_all) // batch_size)
                current_lr = schedule(current_step)
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val MSE: {float(val_loss):.6f} | "
                    f"LR: {float(current_lr):.2e}" 
                )
                # print(f"    Soft k (avg): {jnp.mean(k_soft):.2f} / {train_all.shape[1]}")
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val MSE: {float(val_loss):.6f}"
                )
    
    print(f"\nTraining complete!")
    return model_params, model_static



# ---------------------------------------------------------------------------------------
# Training directly on the true hamiltonian instead of unrolling the trajectories
# ---------------------------------------------------------------------------------------

def train_true_potential(
    batch_size, 
    epochs, 
    key, 
    system_params, 
    model_params, 
    model_static, 
    train_all,          # [N_train, T, state_dim]
    val_all,            # [N_val,   T, state_dim]
    true_potential_fn,  # function: (y, system_params) -> scalar potential
    learning_rate=1e-3,
    lr_schedule=None,
    print_every=25
):
    
    # Shapes
    n_train, T, state_dim = train_all.shape
    n_val   = val_all.shape[0]

    val_initial_conditions = val_all[:, 0]
    
    optimizer, schedule = create_optimizer(
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state = optimizer.init(model_params)

    @eqx.filter_value_and_grad
    def compute_loss(model_params, model_static, batch_trajectories):
        """
        Compute MSE loss.
        """
        B, T, state_dim = batch_trajectories.shape

        # Flatten trajectories into a big batch of states [B*T, state_dim]
        flattened_data = batch_trajectories.reshape(-1, state_dim)

        model = eqx.combine(model_params, model_static)
        pred_potential = jax.vmap(model)(flattened_data)
        pred_potential = jnp.squeeze(pred_potential)       # ensure shape is [B*T]

        true_potential = jax.vmap(true_potential_fn, in_axes=(0, None))(
            flattened_data, system_params
        )
        return mse(pred_potential, true_potential)
    
    @eqx.filter_jit
    def training_step(model_params, model_static, optimizer_state, batch_trajectories):
        """Perform one gradient update step."""
        loss, gradients = compute_loss(model_params, model_static, batch_trajectories)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )
        model_params = eqx.apply_updates(model_params, updates)
        return model_params, optimizer_state, loss
    
    @eqx.filter_jit
    def compute_loss_nograd(model_params, model_static, batch_trajectories):
        # Similar to compute_loss, but without grad
        B, T, state_dim = batch_trajectories.shape
        flattened_data = batch_trajectories.reshape(-1, state_dim)
        model = eqx.combine(model_params, model_static)
        pred_potential = jax.vmap(model)(flattened_data)
        pred_potential = jnp.squeeze(pred_potential)
        true_potential = jax.vmap(true_potential_fn, in_axes=(0, None))(
            flattened_data, system_params
        )
        return mse(pred_potential, true_potential)

    # Training loop
    if lr_schedule:
        print(f"LR schedule: {lr_schedule['type']}, decay={lr_schedule['decay_rate']} every {lr_schedule['decay_epochs']} epochs")
    
    for epoch in range(1, epochs + 1):
        # Shuffle training data for this epoch
        key, subkey = jax.random.split(key)
        
        # Train on all batches
        for batch in make_minibatches(subkey, train_all, batch_size):
            model_params, optimizer_state, train_loss = training_step(
                model_params, model_static, optimizer_state, batch
            )
        
        val_loss = compute_loss_nograd(model_params, model_static, val_all)
        
        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            if schedule is not None:
                current_step = epoch * (len(train_all) // batch_size)
                current_lr = schedule(current_step)
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f} | "
                    f"LR: {float(current_lr):.2e}"
                )
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f}"
                )
    
    print(f"\nTraining complete!")
    return model_params, model_static


# ---------------------------------------------------------------------------------------
# Fully observed training function
# ---------------------------------------------------------------------------------------

def train_fully_observed(
    batch_size, 
    epochs, 
    key, 
    run_hnn, 
    system_params, 
    model_params, 
    model_static, 
    train_all, 
    val_all, 
    learning_rate=1e-3,
    lr_schedule=None,
    print_every=25
):
    # Extract initial conditions for validation
    val_initial_conditions = val_all[:, 0]
    
    # Create optimizer with optional learning rate schedule
    optimizer, schedule = create_optimizer(
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        n_train_samples=len(train_all),
        batch_size=batch_size,
        epochs=epochs
    )
    optimizer_state = optimizer.init(model_params)
    
    @eqx.filter_jit
    def predict_trajectories(model_params, model_static, initial_conditions):
        """
        Predict full state trajectories from initial conditions.
        """
        model = eqx.combine(model_params, model_static)
        return run_hnn(model, system_params, initial_conditions)
    
    @eqx.filter_value_and_grad
    def compute_loss(model_params, model_static, batch_trajectories):
        """
        Compute MSE loss.
        """
        # Extract initial conditions and predict full trajectories
        initial_conditions = batch_trajectories[:, 0]
        predicted_trajectories = predict_trajectories(
            model_params, model_static, initial_conditions
        )
        
        # Compute loss on all state components
        return mse(predicted_trajectories, batch_trajectories)
    
    @eqx.filter_jit
    def training_step(model_params, model_static, optimizer_state, batch_trajectories):
        """Perform one gradient update step."""
        loss, gradients = compute_loss(model_params, model_static, batch_trajectories)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )
        model_params = eqx.apply_updates(model_params, updates)
        return model_params, optimizer_state, loss
    
    # Training loop
    if lr_schedule:
        print(f"LR schedule: {lr_schedule['type']}, decay={lr_schedule['decay_rate']} every {lr_schedule['decay_epochs']} epochs")
    
    for epoch in range(1, epochs + 1):
        # Shuffle training data for this epoch
        key, subkey = jax.random.split(key)
        
        # Train on all batches
        for batch in make_minibatches(subkey, train_all, batch_size):
            model_params, optimizer_state, train_loss = training_step(
                model_params, model_static, optimizer_state, batch
            )
        
        # Compute validation loss (on a subset for speed)
        val_predictions = predict_trajectories(
            model_params, model_static, val_initial_conditions[:batch_size]
        )
        val_loss = mse(val_predictions, val_all[:batch_size])
        
        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            if schedule is not None:
                current_step = epoch * (len(train_all) // batch_size)
                current_lr = schedule(current_step)
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f} | "
                    f"LR: {float(current_lr):.2e}"
                )
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {float(train_loss):.6f} | "
                    f"Val Loss: {float(val_loss):.6f}"
                )
    
    print(f"\nTraining complete!")
    return model_params, model_static