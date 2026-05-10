import jax
import jax.numpy as jnp
import equinox as eqx


class ICEncoder(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, key, conditioning_steps: int, hidden_dim=64, depth=2):
        # input: K steps of (x0,p0) => 2K features
        self.net = eqx.nn.MLP(
            in_size=2*conditioning_steps,
            out_size=2,  # (x1_0, p1_0)
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x,
            key=key
        )

    def __call__(self, obs_prefix_flat):
        return self.net(obs_prefix_flat)  # [2]
    

class ICEncoderLSTM(eqx.Module):
    lstm: eqx.nn.LSTMCell
    head: eqx.nn.MLP
    hidden_dim: int

    def __init__(self, key, hidden_dim: int = 64, head_depth: int = 1):
        """
        Sequence encoder:
          input at each time step: [x0(t), p0(t)] (2 dims)
          output: [x1_0, p1_0] (2 dims)
        """
        k1, k2 = jax.random.split(key, 2)
        self.hidden_dim = hidden_dim
        self.lstm = eqx.nn.LSTMCell(input_size=2, hidden_size=hidden_dim, key=k1)
        self.head = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=2,
            width_size=hidden_dim,
            depth=head_depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x,
            key=k2,
        )

    def __call__(self, obs_prefix):
        """
        obs_prefix: array shape (K, 2) = (x0, p0) over K time steps.
        Returns: array shape (2,) = (x1_0, p1_0)
        """
        K, D = obs_prefix.shape
        if D != 2:
            raise ValueError(f"Expected obs_prefix shape (K,2), got {obs_prefix.shape}")

        # initial LSTM state
        h = jnp.zeros((self.hidden_dim,))
        c = jnp.zeros((self.hidden_dim,))

        # scan over time
        def step(carry, x_t):
            h, c = carry
            h, c = self.lstm(x_t, (h, c))
            return (h, c), h

        (h, c), _ = jax.lax.scan(step, (h, c), obs_prefix)

        # map final hidden state to (x1_0, p1_0)
        return self.head(h)


    
class HamiltonianNN(eqx.Module):
    net: eqx.nn.MLP
    def __init__(self, key, input_dim, hidden_dim=64, depth=3):
        self.net = eqx.nn.MLP(
            in_size=input_dim, out_size=1,
            width_size=hidden_dim, depth=depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x, 
            key=key,
        )
    def __call__(self, x):           # only q data or both (q, p)
        return self.net(x)[0]        # scalar
    

class NeuralODE(eqx.Module):
    net: eqx.nn.MLP
    def __init__(self, key, input_dim, output_dim, hidden_dim=64, depth=3):
        self.net = eqx.nn.MLP(
            in_size=input_dim, out_size=output_dim,
            width_size=hidden_dim, depth=depth,
            activation=jax.nn.tanh,
            final_activation=lambda x: x, 
            key=key,
        )
    def __call__(self, y):           # q shape (2,) OR (q, p) shape (4,)
        return self.net(y)           # shape (1,) OR (2,)
