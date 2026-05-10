import equinox as eqx
import jax.numpy as jnp
import jax

class HamiltonianNN(eqx.Module):
    net: eqx.nn.MLP
    relative_distances: bool

    def __init__(self, key, input_dim, hidden_dim=64, depth=3, relative_distances=True):
        self.relative_distances = relative_distances
        if relative_distances:
            if input_dim == 6:
                in_size = 3  
            elif input_dim == 12:
                in_size = 9  
            else:
                raise ValueError(f"input_dim must be 6 (q-only) or 12 (full state), got {input_dim}")
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
       
        input_dim = x.shape[-1]
        
        if input_dim == 6:
            q = x
            has_momentum = False

        elif input_dim == 12:
            q = x[..., :6] 
            p = x[..., 6:]  
            has_momentum = True

        else:
            raise ValueError(f"Expected input dimension 6 or 12, got {input_dim}")

        original_shape = q.shape[:-1]  
        q_reshaped = q.reshape(*original_shape, 3, 2)
        
        r01 = jnp.linalg.norm(q_reshaped[..., 0, :] - q_reshaped[..., 1, :], axis=-1, keepdims=True)
        r12 = jnp.linalg.norm(q_reshaped[..., 1, :] - q_reshaped[..., 2, :], axis=-1, keepdims=True)
        r02 = jnp.linalg.norm(q_reshaped[..., 0, :] - q_reshaped[..., 2, :], axis=-1, keepdims=True)
        
        # Stack distances: (..., 3)
        distances = jnp.concatenate([r01, r12, r02], axis=-1)

        if has_momentum:
            return jnp.concatenate([distances, p], axis=-1)  # (..., 9)
        else:
            return distances  # (..., 3)
        
    def __call__(self, x):           # only q data or both (q, p)

        if self.relative_distances:
            x = self._preprocess_positions(x)

        return self.net(x)[0]        # scalar
    

class NeuralODE(eqx.Module):
    net: eqx.nn.MLP
    relative_distances: bool

    def __init__(self, key, input_dim, output_dim, hidden_dim=64, depth=3, relative_distances=True):
        self.relative_distances = relative_distances
        if relative_distances:
            if input_dim == 6:
                in_size = 3  # 3 distances (r01, r12, r02)
            elif input_dim == 12:
                in_size = 9  # 3 distances + 6 momenta
            else:
                raise ValueError(f"input_dim must be 6 (q-only) or 12 (full state), got {input_dim}")
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

        input_dim = y.shape[-1]
        
        if input_dim == 6:
            q = y
            has_momentum = False
        elif input_dim == 12:
            q = y[..., :6]
            p = y[..., 6:]
            has_momentum = True
        else:
            raise ValueError(f"Expected input dimension 6 or 12, got {input_dim}")
        
        # Reshape to (..., 3, 2) for computing pairwise distances
        original_shape = q.shape[:-1]
        q_reshaped = q.reshape(*original_shape, 3, 2)
        
        # Compute pairwise distances
        r01 = jnp.linalg.norm(q_reshaped[..., 0, :] - q_reshaped[..., 1, :], axis=-1, keepdims=True)
        r12 = jnp.linalg.norm(q_reshaped[..., 1, :] - q_reshaped[..., 2, :], axis=-1, keepdims=True)
        r02 = jnp.linalg.norm(q_reshaped[..., 0, :] - q_reshaped[..., 2, :], axis=-1, keepdims=True)
        
        # Stack distances: (..., 3)
        distances = jnp.concatenate([r01, r12, r02], axis=-1)
        
        if has_momentum:
            return jnp.concatenate([distances, p], axis=-1)  # (..., 9)
        else:
            return distances  # (..., 3)

    def __call__(self, y):           

        if self.relative_distances:
            y = self._preprocess_positions(y)

        return self.net(y)           
