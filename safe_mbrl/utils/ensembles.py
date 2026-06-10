from typing import Sequence, Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from flax import nnx

from safe_mbrl.utils.mlp import MLP

EPS = 1e-6

def gaussian_log_likelihood(x: jax.Array, mu: jax.Array, sig: jax.Array):
    log_sig = jnp.log(sig + EPS)
    log_l = -0.5 * (2 * log_sig + jnp.log(2 * jnp.pi)
                    + jnp.square((x - mu) / (sig + EPS)))
    log_l = jnp.sum(log_l, axis=-1)
    return log_l

class EnsembleModel(nnx.Module):
    def __init__(self,
                input_dim: int = 1,
                features: Sequence[int]=(128, 128),
                num_ensembles: int = 5,
                output_dim: int = 1, 
                activation: str = "silu",
                key: jax.Array = jax.random.key(0)):
        
        self.input_dim = input_dim
        self.features = features
        self.num_ensembles = num_ensembles
        self.output_dim = output_dim
        self.set_activation(activation)
    
        @nnx.vmap(in_axes=0, out_axes=0)
        def create_ensemble(key: jax.Array):
            return MLP(
            input_dim=input_dim,
            features=features,
            output_dim=output_dim,
            activation=self._activation,
            rngs=nnx.Rngs(key)
        )
            
        keys = jax.random.split(key, self.num_ensembles)
        self.net = create_ensemble(keys)    
            
    def set_activation(self, activation):
        if activation == "relu":
            self._activation = nnx.relu
        elif activation == "leaky_relu":
            self._activation = nnx.leaky_relu
        elif activation == "sigmoid":
            self._activation = nnx.sigmoid
        elif activation == "silu":
            self._activation = nnx.silu
        else:
            raise ValueError("Unknown activation function")

    def _predict(self, x: jax.Array) -> jax.Array:
        """Predict next mean and aleatoric std."""
        forward = lambda model, x: model(x)
        predictions = nnx.vmap(forward, in_axes=(0, None), out_axes=0)(self.net, x)
        mu = jnp.mean(predictions, axis=0)
        epistemic = jnp.std(predictions, axis=0)
        return jnp.concatenate([mu, epistemic], axis=-1)
    
    def __call__(self, x:jax.Array):
         return self._predict(x)
     
class ProbabilisticEnsembleModel(nnx.Module):
    def __init__(
            self,
            input_dim: int =1,
            features: Sequence[int] = (128, 128),
            num_ensembles: int = 10,
            output_dim: int = 1,
            activation: str = "silu",
            key: jax.Array = jax.random.key(0),
            sig_min: float = 1e-3,
            sig_max: float = 1e3,
            deterministic: bool = False
    ):
        self.output_dim = output_dim
        self.num_ensembles = num_ensembles
        self.deterministic = deterministic
        self.set_activation(activation)
        
        

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_ensemble(key: jax.Array):
            return MLP(
            input_dim=input_dim,
            features=features,
            output_dim=2 * output_dim,
            activation=self._activation,
            rngs=nnx.Rngs(key)
        )
        
        keys = jax.random.split(key, self.num_ensembles)
        self.net = create_ensemble(keys)
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.num_ps = 10

        
    def set_activation(self, activation):
        if activation == "relu":
            self._activation = nnx.relu
        elif activation == "leaky_relu":
            self._activation = nnx.leaky_relu
        elif activation == "sigmoid":
            self._activation = nnx.sigmoid
        elif activation == "silu":
            self._activation = nnx.silu
        else:
            raise ValueError("Unknown activation function")

    def _predict(self, x: jax.Array) -> jax.Array:
        """Predict next mean and aleatoric std."""
        forward = lambda model, x: model(x)
        predictions = nnx.vmap(forward, in_axes=(0, None), out_axes=0)(self.net, x)
        mu, sig = jnp.split(predictions, 2, axis=-1)
        sig = nnx.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        eps = jnp.ones_like(sig) * self.sig_min
        sig = (1 - self.deterministic) * sig + self.deterministic * eps
        predictions = jnp.concatenate([mu, sig], axis=-1)
        return predictions
    

    def _loss_grad_fn(self, x: jax.Array, y: jax.Array):
        loss, grads = nnx.value_and_grad(lambda model, x, y: model._likelihood_loss(x,y))(self, x, y)
        return loss, grads
    
    def _likelihood_loss(self, x, y):
        likelihood = jax.vmap(gaussian_log_likelihood, in_axes=(None, 0, 0), out_axes=0)
        predictions = self.__call__(x)
        mu, sig = jnp.split(predictions, 2, axis=-1)
        logl = likelihood(y, mu, sig)
        return -logl.mean()

    
    def calculate_calibration_score(self, xs, ys, ps, alpha, output_dim):
        assert alpha.shape == (output_dim,)

        def calculate_score(x, y):
            predictions = self.__call__(x)
            mean, std = jnp.split(predictions, 2, axis=-1)
            mu = jnp.mean(mean, axis=0)
            eps_std = jnp.std(mean, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
            cdfs = jax.vmap(norm.cdf)(y, mu, eps_std * alpha + al_uncertainty)
            def check_cdf(cdf):
                assert cdf.shape == ()
                return cdf <= ps

            return jax.vmap(check_cdf, out_axes=1)(cdfs)

        cdfs = jax.vmap(calculate_score)(xs, ys)
        return jnp.mean(cdfs, axis=0)
    
    def calculate_calibration_error(self, xs, ys, ps, alpha, output_dim):
        ps_hat = self.calculate_calibration_score(xs, ys, ps, alpha, output_dim)
        ps = jnp.repeat(ps[..., jnp.newaxis], repeats=output_dim, axis=1)
        return jnp.mean((ps - ps_hat) ** 2, axis=0)

    def calculate_calibration_alpha(self, xs, ys):
        ps = jnp.linspace(0, 1, self.num_ps + 1)[1:]
        test_alpha = jnp.flip(jnp.linspace(0, 1, 100)[1:])
        test_alphas = jnp.repeat(test_alpha[..., jnp.newaxis], repeats=self.output_dim, axis=1)

        errors = nnx.vmap(
            lambda x, y, p, a: self.calculate_calibration_error(x, y, p, a, self.output_dim),
            in_axes=(None, None, None, 0)
        )(xs, ys, ps, test_alphas)
        
        indices = jnp.argmin(errors, axis=0)
        best_alpha = test_alpha[indices]
        assert best_alpha.shape == (self.output_dim,)
        return best_alpha, jnp.diag(errors[indices])

    def __call__(self, x:jax.Array):
         return self._predict(x)