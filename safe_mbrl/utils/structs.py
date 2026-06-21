import jax
import jax.numpy as jnp

import flax.struct as struct

from typing import Any

SAMPLING_TYPES = {'mean':0, 'normal':1, 'TS1':2, 'TSInf':3, 'DS':4}

@struct.dataclass
class Dataset:
    input: jax.Array
    target: jax.Array

    def __len__(self):
        return self.target.shape[0]

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return jax.tree_util.tree_map(lambda *x: jax.numpy.concatenate(x, axis=axis), self, *others)

    def shift_order(self, key: jax.Array=jax.random.key(0)):
        idx = jax.random.permutation(key, jax.numpy.arange(len(self)))
        return Dataset(self.input[idx], self.target[idx])

    def slice(self, beg: int, end: int) -> Any:
        return jax.tree_util.tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return jax.tree_util.tree_map(lambda x: jax.numpy.take(x, i, axis=axis, mode='wrap'), self)
    
    
@struct.dataclass
class RobotState:
    
    q_buffer: jax.Array
    qd_buffer: jax.Array
    act_buffer: jax.Array
    q_dim: int = struct.field(pytree_node=False)

    def ravel(self) -> jax.Array:
        return jnp.concatenate([
            self.q_buffer,
            self.qd_buffer,
            self.act_buffer
        ])

    def current_state(self):
        return jnp.concatenate([
            self.q_buffer[-self.q_dim:],
            self.qd_buffer[-self.q_dim:],
            self.act_buffer[-self.q_dim:]
        ])

    def take(self, i, axis=0) -> Any:
        return jax.tree_util.tree_map(
            lambda x: jax.numpy.take(x, i, axis=axis, mode='wrap'),
            self
        )
    
    def get_q(self)->jax.Array:
        return self.q_buffer[-self.q_dim:] 
    
    def get_qd(self)->jax.Array:
        return self.qd_buffer[-self.q_dim:] 

    @classmethod
    def create(cls,
               current_q: jax.Array,
               buffer_size: int = 10,
               q_dim: int = 4):
        q_buffer = jnp.tile(current_q, buffer_size)
        qd_buffer = jnp.zeros(q_dim * buffer_size)
        act_buffer = jnp.zeros(q_dim * buffer_size)
        return RobotState(
            q_buffer=q_buffer,
            qd_buffer=qd_buffer,
            act_buffer=act_buffer,
            q_dim=q_dim
        )
        
