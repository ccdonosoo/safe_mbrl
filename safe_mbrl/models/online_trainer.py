import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Sequence

from safe_mbrl.utils.type_aliases import RobotState, Dataset


# Important update, assymetric joint pos and joint vel, embedding. 
# Check the entirity of this code, manually.
# Also check the mppi code with the online trainer
# Check the MPC algorithms , fulll, to see if there is no inconsitency.

class OnlineTrainer:

    def __init__(self,
                 model,                              # RobotEnsemble
                 optimizer,                          # optax.GradientTransformation
                 train_dataset: Sequence[Dataset],
                 val_dataset: Sequence[Dataset],
                 batch_size: int = 256,
                 horizon: int = 10,
                 horizon_val: int = 50,
                 gamma: float = 0.95,
                 nb_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 logger=None):
        self.model = model
        self._tx = optimizer
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._batch_size = batch_size
        self._horizon = horizon
        self._horizon_val = horizon_val
        self._gamma = gamma
        self._nb_epochs = nb_epochs
        self._early_stopping_patience = early_stopping_patience
        self._logger = logger
        self.epoch = 0

    def log_training(self, loss_train: float, loss_val: float):
        if self._logger is None:
            return
        self._logger.scalar_summary("train/train_loss", loss_train, self.epoch)
        self._logger.scalar_summary("train/val_loss", loss_val, self.epoch)

    def train_model_bptt(self, seed: int = 0, verbose: bool = True):
        key = jax.random.key(seed)
        jd, bd = self.model.joint_dim, self.model.buffer_dim
        mode, dt, gamma = self.model.mode, self.model._dt, self._gamma
        H, Hv = self._horizon, self._horizon_val

        # nnx transforms keep the ensemble as a tracked graph node (graph/state are
        # split internally by nnx.jit), so a fixed batch size compiles once.
        ens = self.model.model
        optimizer = nnx.Optimizer(ens, self._tx, wrt=nnx.Param)

        def mean_rollout(loss_single, ens, states, actions, horizon):
            f = lambda e, s, a: loss_single(e, s, a, jd, mode, dt, gamma, horizon)
            return jnp.mean(nnx.vmap(f, in_axes=(None, 0, 0))(ens, states, actions))

        @nnx.jit
        def train_step(ens, optimizer, states, actions):
            loss, grads = nnx.value_and_grad(
                lambda e: mean_rollout(rollout_loss, e, states, actions, H))(ens)
            optimizer.update(ens, grads)
            return loss

        best_mse, best_state, patience = float("inf"), nnx.state(ens, nnx.Param), 0
        n_batches = max(1, sum(len(d) for d in self._train_dataset) // self._batch_size)

        for epoch in range(self.epoch, self.epoch + self._nb_epochs):
            self.epoch = epoch
            train_losses = []
            for _ in range(n_batches):
                key, states, actions = sample_rollout_datasets(self._train_dataset, H, self._batch_size, key, jd, bd)
                train_losses.append(np.asarray(train_step(ens, optimizer, states, actions)))

            _, vs, va = sample_rollout_datasets(self._val_dataset, Hv, 100, jax.random.key(0), jd, bd)
            val_nll = float(mean_rollout(rollout_loss, ens, vs, va, Hv))
            val_mse = float(mean_rollout(rollout_loss_mse, ens, vs, va, Hv))
            train_loss = float(np.mean(train_losses))

            if verbose:
                print(f"Epoch {epoch}: train_nll {train_loss:.6e}, val_nll {val_nll:.6e}, val_mse {val_mse:.6e}")
            self.log_training(train_loss, val_nll)

            if val_mse < best_mse:
                best_mse, best_state, patience = val_mse, nnx.state(ens, nnx.Param), 0
            else:
                patience += 1
                if patience > self._early_stopping_patience:
                    break

        # Restore the best params into the live model in place.
        nnx.update(ens, best_state)
        return best_mse



def _bptt_step(ens, state, action, joint_dim, mode, dt):
    """One differentiable model step: predict -> integrate -> roll buffers, with
    the real `action` injected into the newest action slot."""
    mu, _ = jnp.split(ens(state.ravel()), 2, axis=-1)
    if mu.ndim > 1:                              # PE: (num_ensembles, joint_dim)
        mu = jnp.mean(mu, axis=0)

    qd_next = state.get_qd() + mu if mode == "dv" else mu
    q_next = state.get_q() + qd_next * dt

    jd = joint_dim
    q_buffer = jnp.roll(state.q_buffer, -jd).at[-jd:].set(q_next)
    qd_buffer = jnp.roll(state.qd_buffer, -jd).at[-jd:].set(qd_next)
    act_buffer = jnp.roll(state.act_buffer, -jd).at[-jd:].set(action)
    return state.replace(q_buffer=q_buffer, qd_buffer=qd_buffer, act_buffer=act_buffer)


def rollout_loss(ens, state_rollout, action_sequence, joint_dim, mode, dt, gamma, horizon):
    """Discounted open-loop NLL over the horizon (requires a PE ensemble)."""
    graphdef, estate = nnx.split(ens)          # thread params through the loop carry
    state0 = state_rollout.take(0)

    def body(i, carry):
        estate, state, loss = carry
        ens = nnx.merge(graphdef, estate)
        qd_target = state_rollout.qd_buffer[i + 1, -joint_dim:]
        y = qd_target - state.get_qd() if mode == "dv" else qd_target
        loss = loss + ens._likelihood_loss(state.ravel(), y) * gamma ** i
        state = _bptt_step(ens, state, action_sequence[i + 1], joint_dim, mode, dt)
        return (estate, state, loss)

    _, _, loss = jax.lax.fori_loop(0, horizon, body, (estate, state0, jnp.zeros(())))
    return loss / horizon


def rollout_loss_mse(ens, state_rollout, action_sequence, joint_dim, mode, dt, gamma, horizon):
    """Discounted open-loop MSE on joint position (model-agnostic eval metric)."""
    graphdef, estate = nnx.split(ens)
    state0 = state_rollout.take(0)

    def body(i, carry):
        estate, state, loss = carry
        ens = nnx.merge(graphdef, estate)
        state = _bptt_step(ens, state, action_sequence[i + 1], joint_dim, mode, dt)
        q_target = state_rollout.q_buffer[i + 1, -joint_dim:]
        loss = loss + jnp.mean((state.get_q() - q_target) ** 2) * gamma ** i
        return (estate, state, loss)

    _, _, loss = jax.lax.fori_loop(0, horizon, body, (estate, state0, jnp.zeros(())))
    return loss / horizon


def _input_to_state(x, joint_dim, buffer_dim):
    n = joint_dim * buffer_dim
    return RobotState(q_buffer=x[:n], qd_buffer=x[n:2 * n], act_buffer=x[2 * n:], q_dim=joint_dim)


def sample_rollout_datasets(datasets: Sequence, horizon, num_rollouts, key, joint_dim, buffer_dim):
    """Sample `num_rollouts` contiguous (horizon+1)-length windows, length-weighted
    across datasets and never crossing a dataset boundary -> fixed-shape batch."""
    H = horizon + 1
    lengths = jnp.array([len(d) for d in datasets], dtype=jnp.int32)
    pmf = lengths / jnp.sum(lengths)
    offsets = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(lengths)[:-1]])

    key, rng = jax.random.split(key)
    didx = jax.random.choice(rng, jnp.arange(lengths.shape[0]), shape=(num_rollouts,), p=pmf)
    keys = jax.random.split(rng, num_rollouts)
    local = jax.vmap(lambda d, k: jax.random.randint(k, (), 0, lengths[d] - H))(didx, keys)
    starts = offsets[didx] + local

    data = Dataset.concatenate(*datasets)

    def sample(start):
        inp = jax.lax.dynamic_slice_in_dim(data.input, start, H, axis=0)   # (H, 3*jd*bd)
        state = jax.vmap(lambda x: _input_to_state(x, joint_dim, buffer_dim))(inp)
        return state, state.act_buffer[:, -joint_dim:]                     # (H, jd)

    states, actions = jax.vmap(sample)(starts)
    return key, states, actions

