import jax
import jax.numpy as jnp
from typing import Dict, Optional, Sequence, Union

from safe_mbrl.utils.ensembles import ProbabilisticEnsembleModel, EnsembleModel

from safe_mbrl.utils.structs import RobotState

_MASK_KEYS = ("q", "qd", "act")




class RobotEnsemble(object):

    def __init__(self,
                 joint_dim: int = 4,
                 buffer_dim: int = 10,
                 model_type: str = "PE",
                 mode: str = "dv", # "v, dv"
                 activation: str = "silu",
                 key: jax.Array = jax.random.key(0),
                 features: Sequence[int] = (128, 128, 128),
                 num_ensembles: int = 5,
                 dt: float = 0.04,
                 mask: Optional[Dict[str, Sequence[int]]] = None
                 ):

        self.joint_dim = joint_dim
        self.buffer_dim = buffer_dim
        self.model_type = model_type
        self.mode = mode
        self._dt = dt
        self.mask = mask
        self._input_idx = (None if mask is None
                           else _build_input_index(mask, joint_dim, buffer_dim))

        if self.mode not in ("v", "dv"):
            raise NotImplementedError(f"Just dv or v prediction, not {self.mode}")

        self.build_model(
            input_dim = (joint_dim*buffer_dim*3 if mask is None # Joint position, velocity, and action buffer
                         else int(self._input_idx.size)),
            features = features,
            num_ensembles = num_ensembles,
            output_dim = joint_dim,             # We predict v or dv
            activation = activation,
            key = key
        )

    def featurize(self, robot_state: RobotState) -> jax.Array:
        x = robot_state.ravel()
        return x if self._input_idx is None else x[self._input_idx]
    
    def build_model(self, **kwargs):
        if self.model_type == "PE":
            self.model = ProbabilisticEnsembleModel(**kwargs)
        elif self.model_type == "DE":
            self.model = EnsembleModel(**kwargs)
        else:
            raise NotImplementedError(f"Unknown model_type: {self.model_type!r}")
        
    


    def step(self, robot_state: RobotState) -> RobotState: # We roll everything, but the action is padded wirh zero
        prediction = self.model(self.featurize(robot_state))
        mu , _ = jnp.split(prediction, 2, axis=-1)
        
        if self.model_type == "PE":
            mu = mu.mean(axis=0)

        q = robot_state.q_buffer[-self.joint_dim:]
        
        if self.mode == "dv":
            qd = robot_state.qd_buffer[-self.joint_dim:]
            qd_next = qd + mu
            
        elif self.mode == "v":
            qd_next = mu
            
        q_next = q + qd_next * self._dt
        
        # Update buffers
        q_buffer = jnp.roll(robot_state.q_buffer, -self.joint_dim).at[-self.joint_dim:].set(q_next)        
        qd_buffer = jnp.roll(robot_state.qd_buffer, -self.joint_dim).at[-self.joint_dim:].set(qd_next)        
        act_buffer = jnp.roll(robot_state.act_buffer, -self.joint_dim).at[-self.joint_dim:].set(jnp.zeros(self.joint_dim)) 
        
        robot_state = robot_state.replace(q_buffer=q_buffer,
                                        qd_buffer=qd_buffer,
                                        act_buffer=act_buffer)
        
        return robot_state
            
            
def _build_input_index(mask: Dict[str, Sequence[int]], joint_dim: int, buffer_dim: int) -> jax.Array:
    # mask: {"q"|"qd"|"act": newest steps kept per joint}, 0 drops the signal,
    # missing key keeps the full buffer. Eager only (data-dependent size).
    unknown = set(mask) - set(_MASK_KEYS)
    if unknown:
        raise ValueError(f"Unknown mask keys {unknown}; expected a subset of {_MASK_KEYS}")
    full = [buffer_dim] * joint_dim
    rows = [list(mask.get(k, full)) for k in _MASK_KEYS]
    for key, row in zip(_MASK_KEYS, rows):
        if len(row) != joint_dim:
            raise ValueError(f"mask[{key!r}] must have {joint_dim} entries, got {len(row)}")
        if not all(0 <= lag <= buffer_dim for lag in row):
            raise ValueError(f"mask[{key!r}] lags must be in [0, {buffer_dim}], got {row}")
    if not all(lag >= 1 for lag in rows[2]):
        raise ValueError("mask['act'] lags must be >= 1")

    lags = jnp.asarray(rows, dtype=jnp.int32)
    t = jnp.arange(buffer_dim)[None, :, None]
    keep = t >= buffer_dim - lags[:, None, :]
    return jnp.flatnonzero(keep)
    

