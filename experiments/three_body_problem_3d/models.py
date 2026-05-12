import equinox as eqx
import jax.numpy as jnp
import jax

class HamiltonianNN(eqx.Module):
    net: eqx.nn.MLP
    relative_distances: bool

    def __init__(self, key, input_dim, hidden_dim=64, depth=3, relative_distances=True):
        self.relative_distances = relative_distances

        if relative_distances:
            if input_dim == 9:
                in_size = 3  
            elif input_dim == 18:
                in_size = 12  
            else:
                raise ValueError(f"input_dim must be 9 (q-only) or 18 (full state), got {input_dim}")
        else:
            in_size = input_dim

        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=1,
            width_size=hidden_dim, depth=depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x, 
            key=key,
        )

    def _preprocess_positions(self, x):
        """
        Convert absolute coordinates to relative distances.
        """
        input_dim = x.shape[-1]
        #eps = 1e-12  # For numerical stability
        
        if input_dim == 9:
            q = x
            has_momentum = False

        elif input_dim == 18:
            q = x[..., :9] 
            p = x[..., 9:]  
            has_momentum = True

        else:
            raise ValueError(f"Expected input dimension 9 or 18, got {input_dim}")

        original_shape = q.shape[:-1]  
        q_reshaped = q.reshape(*original_shape, 3, 3)
        
        r01_vec = q_reshaped[..., 0, :] - q_reshaped[..., 1, :]
        r12_vec = q_reshaped[..., 1, :] - q_reshaped[..., 2, :] 
        r02_vec = q_reshaped[..., 0, :] - q_reshaped[..., 2, :]
        
        r01 = jnp.sqrt(jnp.sum(r01_vec**2, axis=-1, keepdims=True)) 
        r12 = jnp.sqrt(jnp.sum(r12_vec**2, axis=-1, keepdims=True)) 
        r02 = jnp.sqrt(jnp.sum(r02_vec**2, axis=-1, keepdims=True)) 
        
        distances = jnp.concatenate([r01, r12, r02], axis=-1)

        if has_momentum:
            return jnp.concatenate([distances, p], axis=-1)  # (..., 3 + 9 = 12)
        else:
            return distances         # (..., 3)
        
    def __call__(self, x):           # only q data or both (q, p)
        """
        Forward pass through HamiltonianNN.
        
        Parameters
        ----------
        x : array
            Input state [q] or [q, p]

        Returns
        -------
        energy : float
            Predicted energy (scalar)
        """
        if self.relative_distances:
            x = self._preprocess_positions(x)

        return self.net(x)[0]        # scalar
    

class NeuralODE(eqx.Module):
    net: eqx.nn.MLP
    relative_distances: bool

    def __init__(self, key, input_dim, output_dim, hidden_dim=64, depth=3, relative_distances=True):
        self.relative_distances = relative_distances
        if relative_distances:
            if input_dim == 9:
                in_size = 3  # 3 distances (r01, r12, r02)
            elif input_dim == 18:
                in_size = 12  # 3 distances + 9 momenta
            else:
                raise ValueError(f"input_dim must be 9 (q-only) or 18 (full state), got {input_dim}")
        else:
            in_size = input_dim
            
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=output_dim,
            width_size=hidden_dim, depth=depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x, 
            key=key,
        )

    def _preprocess_positions(self, y):
        """
        Convert absolute coordinates to relative distances.
        """
        input_dim = y.shape[-1]
        eps = 1e-12  # For numerical stability
        
        if input_dim == 9:
            q = y
            has_momentum = False
        elif input_dim == 18:
            q = y[..., :9]
            p = y[..., 9:]
            has_momentum = True
        else:
            raise ValueError(f"Expected input dimension 9 or 18, got {input_dim}")
        
        # Reshape to (..., 3, 3) for computing pairwise distances
        original_shape = q.shape[:-1]
        q_reshaped = q.reshape(*original_shape, 3, 3)
        
        # Compute pairwise distances with numerical stability
        r01_vec = q_reshaped[..., 0, :] - q_reshaped[..., 1, :]
        r12_vec = q_reshaped[..., 1, :] - q_reshaped[..., 2, :] 
        r02_vec = q_reshaped[..., 0, :] - q_reshaped[..., 2, :]
        
        r01 = jnp.sqrt(jnp.sum(r01_vec**2, axis=-1, keepdims=True) + eps)
        r12 = jnp.sqrt(jnp.sum(r12_vec**2, axis=-1, keepdims=True) + eps)
        r02 = jnp.sqrt(jnp.sum(r02_vec**2, axis=-1, keepdims=True) + eps)
        
        # Stack distances: (..., 3)
        distances = jnp.concatenate([r01, r12, r02], axis=-1)
        
        if has_momentum:
            return jnp.concatenate([distances, p], axis=-1)  # (..., 3 + 9 = 12)
        else:
            return distances  # (..., 3)

    def __call__(self, y):           
        """
        Forward pass through NeuralODE.
        
        Parameters
        ----------
        y : array
            Input state [q, p]
            
        Returns
        -------
        dpdt : array
            Time derivatives of momenta
        """
        if self.relative_distances:
            y = self._preprocess_positions(y)

        return self.net(y)           
