import equinox as eqx
import jax

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
