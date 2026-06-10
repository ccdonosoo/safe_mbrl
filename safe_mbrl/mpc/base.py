from typing import Optional, Union
import jax.numpy as jnp

class Optimizer(object):
    def __init__(self,
                 action_dim=(1, ),
                 upper_bound: Optional[Union[float, jnp.ndarray]] = 1.0,
                 lower_bound: Optional[Union[float, jnp.ndarray]] = None,
                 *args,
                 **kwargs):

        self.action_dim = action_dim
        
        if upper_bound is None:
            self.upper_bound = jnp.ones(self.action_dim)*jnp.inf
            
        elif isinstance(upper_bound, float):
            self.upper_bound = jnp.ones(self.action_dim)*upper_bound
        
        if lower_bound is None:
            self.lower_bound = - upper_bound
        elif isinstance(lower_bound, float):
            self.lower_bound = jnp.ones(self.action_dim)*lower_bound

    def optimize(self, func, rng=None):
        pass

    def clip_action(self, action):
        return jnp.clip(action, a_min=self.lower_bound, a_max=self.upper_bound)