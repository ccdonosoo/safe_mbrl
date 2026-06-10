import jax
import jax.numpy as jnp

from flax import struct
from brax import base

from typing import Optional, Dict, Any, Union, Mapping, Tuple
import abc


Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


@struct.dataclass
class State(base.Base):
    """Environment state for training and inference."""
    pipeline_state: Optional[base.State]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)



class Env(abc.ABC):
    """Interface for driving training and inference."""

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    @abc.abstractmethod
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""

    @property
    def unwrapped(self) -> 'Env':
        return self