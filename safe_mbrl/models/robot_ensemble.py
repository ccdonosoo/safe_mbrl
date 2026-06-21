import jax
import jax.numpy as jnp
from typing import Sequence, Union

from safe_mbrl.utils.ensembles import ProbabilisticEnsembleModel, EnsembleModel

from safe_mbrl.utils.structs import RobotState


class RobotEnsemble(object):
    
    def __init__(self,
                 joint_dim: int = 4,
                 buffer_dim: int = 10, 
                 model_type: str = "PE",
                 mode: str = "dv", # "v, dv"
                 activation: str = "silu",
                 key: jax.Array = jax.random.key(0),
                 features: Sequence[int] = (128, 128),
                 num_ensembles: int = 5,
                 dt: float = 0.04
                 ):
        
        self.joint_dim = joint_dim
        self.buffer_dim = buffer_dim
        self.model_type = model_type
        self.mode = mode
        self._dt = dt
        
        if self.mode not in ("v", "dv"):
            raise NotImplementedError(f"Just dv or v prediction, not {self.mode}")
        
        self.build_model(
            input_dim = joint_dim*buffer_dim*3, # Joint position, velocity, and action buffer
            features = features,
            num_ensembles = num_ensembles,
            output_dim = joint_dim,             # We predict v or dv 
            activation = activation,
            key = key
        )
    
    def build_model(self, **kwargs):
        if self.model_type == "PE":
            self.model = ProbabilisticEnsembleModel(**kwargs)
        elif self.model_type == "DE":
            self.model = EnsembleModel(**kwargs)
        else:
            raise NotImplementedError(f"Unknown model_type: {self.model_type!r}")
        
    

    def fk(self, robot_state_or_q: Union[jax.Array, RobotState]):
        if isinstance(robot_state_or_q, RobotState):
            q = robot_state_or_q.get_q()
        elif isinstance(robot_state_or_q, jax.Array):
            q = robot_state_or_q
        else:
            raise NotImplementedError("query the joint position as a jax array or feed with the robot_state")
        #TODO
        pass
    
    def step(self, robot_state: RobotState) -> RobotState: # We roll everything, but the action is padded wirh zero
        prediction = self.model(robot_state.ravel())
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
            
            
    

