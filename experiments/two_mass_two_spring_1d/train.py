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

    OBSERVED_INDICES = jnp.array([0, 2])

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

    if lr_schedule:
        print(f"LR schedule: {lr_schedule['type']}, decay={lr_schedule['decay_rate']} every {lr_schedule['decay_epochs']} epochs")
    
     # --- Loss logging ---
    train_losses = []
    val_losses = []
    lrs = []  # will stay empty if schedule is None

    # Train HNN (partial-observation)
    key, subkey = jax.random.split(key)

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
    
    return model_params, model_static, losses