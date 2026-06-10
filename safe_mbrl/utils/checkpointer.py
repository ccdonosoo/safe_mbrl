import json
import orbax.checkpoint as orbax
from flax import nnx
import jax
import os
import shutil
from typing import TypeVar, Type

T = TypeVar("T") 

def save_model(model:nnx.Module, config_kwargs:dict, path:str, name: str):
        """
        Save the parameters of the network and all the configuration associated with it:
        """

        full_path = os.path.join(path, name)

        if not os.path.exists(path):
            assert f"Directory '{path}' does not exist. Please create it before saving the checkpoint."

        if os.path.exists(full_path):
            shutil.rmtree(full_path)

        os.makedirs(full_path, exist_ok=True)

        with open(os.path.join(full_path, 'config.json'), 'w') as f:
            json.dump(config_kwargs, f, indent=4)

        graphdef, rng_state, state = nnx.split(model, nnx.RngState, ...)

        checkpointer = orbax.PyTreeCheckpointer()
        checkpointer.save(os.path.join(full_path, 'state'), state)



def load_model(model_class:Type[T], path: str, name: str) -> nnx.Module:
    """
    Load the model and its configuration from the specified path.
    """

    filename = os.path.join(path, name, 'config.json')

    try:
        with open(filename, 'r') as f:
            config_dict = json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{filename}' not found in path: {path}.")

    model = model_class(**config_dict)
    graphdef, rng_state, state = nnx.split(model, nnx.RngState, ...)

    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree.map(
        lambda _: orbax.ArrayRestoreArgs(sharding=sharding),
        state,
    )
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(
        os.path.join(path, name, 'state'),
        item=state,
        restore_args=restore_args,
    )

    nnx.update(model, state)

    return model