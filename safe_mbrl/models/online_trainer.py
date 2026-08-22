import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Sequence

from safe_mbrl.utils.structs import RobotState, Dataset


#TODO Robustness training: random multiplicative masks sampled per batch.


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

    def train_model_bptt(self, seed: int = 0, verbose: bool = True, val_samples: int = 100,
                         train_concat: Dataset = None):
        key = jax.random.key(seed)
        rng, key = jax.random.split(key)
        jd, bd = self.model.joint_dim, self.model.buffer_dim
        mode, dt, gamma = self.model.mode, self.model._dt, self._gamma
        H, Hv = self._horizon, self._horizon_val
        idx = getattr(self.model, "_input_idx", None)

        ens = self.model.model
        optimizer = nnx.Optimizer(ens, self._tx, wrt=nnx.Param)

        def mean_rollout(loss_single, ens, states, actions, horizon):
            f = lambda e, s, a: loss_single(e, s, a, jd, mode, dt, gamma, horizon, input_idx=idx)
            return jnp.mean(nnx.vmap(f, in_axes=(None, 0, 0))(ens, states, actions))

        @nnx.jit
        def train_step(ens, optimizer, states, actions):
            loss, grads = nnx.value_and_grad(
                lambda e: mean_rollout(rollout_loss, e, states, actions, H))(ens)
            optimizer.update(ens, grads)
            return loss

        # No val data -> train on everything, keep the LAST epoch, return its train NLL.
        use_val = self._val_dataset is not None and len(self._val_dataset) > 0
        best_mse, best_state, patience = float("inf"), None, 0
        n_batches = max(1, sum(len(d) for d in self._train_dataset) // self._batch_size)

        train_loss = float("nan")
        for epoch in range(self.epoch, self.epoch + self._nb_epochs):
            self.epoch = epoch
            train_losses = []
            for _ in range(n_batches):
                key, states, actions = sample_rollout_datasets(self._train_dataset, H, self._batch_size, key, jd, bd,
                                                               concat=train_concat)
                train_losses.append(np.asarray(train_step(ens, optimizer, states, actions)))
            train_loss = float(np.mean(train_losses))

            if not use_val:
                if verbose:
                    print(f"Epoch {epoch}: train_nll {train_loss:.6e}")
                self.log_training(train_loss, float("nan"))
                continue

            rng, vs, va = sample_rollout_datasets(self._val_dataset, Hv, val_samples, rng, jd, bd)
            val_nll = float(mean_rollout(rollout_loss, ens, vs, va, Hv))
            val_mse = float(mean_rollout(rollout_loss_mse, ens, vs, va, Hv))

            if verbose:
                print(f"Epoch {epoch}: train_nll {train_loss:.6e}, val_nll {val_nll:.6e}, val_mse {val_mse:.6e}")
            self.log_training(train_loss, val_nll)

            if val_mse < best_mse:
                best_mse, best_state, patience = val_mse, nnx.state(ens, nnx.Param), 0
            else:
                patience += 1
                if patience > self._early_stopping_patience:
                    break

        if not use_val:
            return train_loss
        nnx.update(ens, best_state)   # restore best-epoch params
        return best_mse

    def _make_fused_step(self):
        """One nnx.jit'ed call fusing batch sampling + gradient step. Sampling key
        stream and update math are identical to sample_rollout_datasets + train_step
        in train_model_bptt; only the dispatch granularity changes."""
        jd, bd = self.model.joint_dim, self.model.buffer_dim
        mode, dt, gamma = self.model.mode, self.model._dt, self._gamma
        H, BS = self._horizon, self._batch_size
        idx = getattr(self.model, "_input_idx", None)
        H1 = H + 1

        @nnx.jit
        def fused(ens, optimizer, key, data_input, lengths, offsets, pmf):
            key, rng = jax.random.split(key)
            didx = jax.random.choice(rng, jnp.arange(lengths.shape[0]), shape=(BS,), p=pmf)
            keys = jax.random.split(rng, BS)
            local = jax.vmap(lambda d, k: jax.random.randint(k, (), 0, lengths[d] - H1))(didx, keys)
            starts = offsets[didx] + local

            def sample(start):
                inp = jax.lax.dynamic_slice_in_dim(data_input, start, H1, axis=0)
                state = jax.vmap(lambda x: _input_to_state(x, jd, bd))(inp)
                return state, state.act_buffer[:, -jd:]

            states, actions = jax.vmap(sample)(starts)

            def mean_rollout(e):
                f = lambda e_, s, a: rollout_loss(e_, s, a, jd, mode, dt, gamma, H, input_idx=idx)
                return jnp.mean(nnx.vmap(f, in_axes=(None, 0, 0))(e, states, actions))

            def mean_rollout_mse(e):
                f = lambda e_, s, a: rollout_loss_mse(e_, s, a, jd, mode, dt, gamma, H, input_idx=idx)
                return jnp.mean(nnx.vmap(f, in_axes=(None, 0, 0))(e, states, actions))

            def one_step_vel_mse(e):
                # one-step velocity prediction MSE [rad^2/s^2]: the quantity the
                # online_learning_control baseline logs as "Model Err"
                def single(e_, s, a):
                    s0 = s.take(0)
                    mu, _ = jnp.split(e_(_featurize(s0, idx)), 2, axis=-1)
                    if mu.ndim > 1:                      # PE: (num_ensembles, jd)
                        mu = jnp.mean(mu, axis=0)
                    pred_qd = s0.get_qd() + mu if mode == "dv" else mu
                    return jnp.mean((pred_qd - s.qd_buffer[1, -jd:]) ** 2)
                return jnp.mean(nnx.vmap(single, in_axes=(None, 0, 0))(e, states, actions))

            loss, grads = nnx.value_and_grad(mean_rollout)(ens)
            mse = mean_rollout_mse(ens)                  # pre-update params, same batch
            mse_vel = one_step_vel_mse(ens)
            optimizer.update(ens, grads)
            return key, loss, mse, mse_vel

        return fused

    def train_model_bptt_jit(self, seed: int = 0, verbose: bool = True, train_concat: Dataset = None):
        """Fused-jit variant of train_model_bptt for the no-validation path: same
        sampling distribution, key stream, and optimizer behavior (fresh optimizer
        state per call), with sampling + gradient step compiled into one dispatch.

        Returns {"nll", "mse", "mse_vel"}: the Gaussian NLL that is optimized, the
        discounted H-step joint-position MSE, and the one-step velocity MSE, all
        averaged over the last epoch's batches (identical windows, pre-update params).
        The extra metrics are diagnostics only - they do not affect the update."""
        key = jax.random.key(seed)
        rng, key = jax.random.split(key)

        ens = self.model.model
        optimizer = nnx.Optimizer(ens, self._tx, wrt=nnx.Param)
        if getattr(self, "_fused_step", None) is None:
            self._fused_step = self._make_fused_step()

        lengths_np = np.array([len(d) for d in self._train_dataset], dtype=np.int64)
        lengths = jnp.asarray(lengths_np, dtype=jnp.int32)
        pmf = lengths / jnp.sum(lengths)
        offsets = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(lengths)[:-1]])
        data = Dataset.concatenate(*self._train_dataset) if train_concat is None else train_concat

        n_batches = max(1, int(lengths_np.sum()) // self._batch_size)
        train_loss = float("nan")
        for epoch in range(self.epoch, self.epoch + self._nb_epochs):
            self.epoch = epoch
            losses, mses, mses_vel = [], [], []
            for _ in range(n_batches):
                key, loss, mse, mse_vel = self._fused_step(
                    ens, optimizer, key, data.input, lengths, offsets, pmf)
                losses.append(loss); mses.append(mse); mses_vel.append(mse_vel)
            train_loss = float(np.mean(np.asarray(jnp.stack(losses))))
            train_mse = float(np.mean(np.asarray(jnp.stack(mses))))
            train_mse_vel = float(np.mean(np.asarray(jnp.stack(mses_vel))))
            if verbose:
                print(f"Epoch {epoch}: train_nll {train_loss:.6e}, "
                      f"train_mse {train_mse:.6e}, train_mse_vel {train_mse_vel:.6e}")
            self.log_training(train_loss, float("nan"))
        return {"nll": train_loss, "mse": train_mse, "mse_vel": train_mse_vel}



def _featurize(state, input_idx):
    x = state.ravel()
    return x if input_idx is None else x[input_idx]


def _model_step(ens, state, action, joint_dim, mode, dt, input_idx=None):
    """One differentiable model step: predict -> integrate -> roll buffers, with
    the real `action` injected into the newest action slot."""
    mu, _ = jnp.split(ens(_featurize(state, input_idx)), 2, axis=-1)
    if mu.ndim > 1:                              # PE: (num_ensembles, joint_dim)
        mu = jnp.mean(mu, axis=0)

    qd_next = state.get_qd() + mu if mode == "dv" else mu
    q_next = state.get_q() + qd_next * dt

    jd = joint_dim
    q_buffer = jnp.roll(state.q_buffer, -jd).at[-jd:].set(q_next)
    qd_buffer = jnp.roll(state.qd_buffer, -jd).at[-jd:].set(qd_next)
    act_buffer = jnp.roll(state.act_buffer, -jd).at[-jd:].set(action)
    return state.replace(q_buffer=q_buffer, qd_buffer=qd_buffer, act_buffer=act_buffer)


def rollout_loss(ens, state_rollout, action_sequence, joint_dim, mode, dt, gamma, horizon, scale_q_loss:float=0.0, input_idx=None):
    """Discounted open-loop NLL over the horizon (requires a PE ensemble)."""
    graphdef, estate = nnx.split(ens)          # thread params through the loop carry
    state0 = state_rollout.take(0)

    def body(i, carry):
        estate, state, loss = carry
        ens = nnx.merge(graphdef, estate)
        qd_target = state_rollout.qd_buffer[i + 1, -joint_dim:]
        y = qd_target - state.get_qd() if mode == "dv" else qd_target
        loss = loss + ens._likelihood_loss(_featurize(state, input_idx), y) * gamma ** i
        state = _model_step(ens, state, action_sequence[i + 1], joint_dim, mode, dt, input_idx)
        q_err = state.get_q() - state_rollout.q_buffer[i + 1, -joint_dim:]
        loss = loss + scale_q_loss * jnp.mean(jnp.square(q_err))
        return (estate, state, loss)
 
    _, _, loss = jax.lax.fori_loop(0, horizon, body, (estate, state0, jnp.zeros(())))
    return loss / horizon


def rollout_loss_mse(ens, state_rollout, action_sequence, joint_dim, mode, dt, gamma, horizon, input_idx=None):
    """Discounted open-loop MSE on joint position (model-agnostic eval metric)."""
    graphdef, estate = nnx.split(ens)
    state0 = state_rollout.take(0)

    def body(i, carry):
        estate, state, loss = carry
        ens = nnx.merge(graphdef, estate)
        state = _model_step(ens, state, action_sequence[i + 1], joint_dim, mode, dt, input_idx)
        q_target = state_rollout.q_buffer[i + 1, -joint_dim:]
        loss = loss + jnp.mean((state.get_q() - q_target) ** 2) * gamma ** i
        return (estate, state, loss)

    _, _, loss = jax.lax.fori_loop(0, horizon, body, (estate, state0, jnp.zeros(())))
    return loss / horizon


def _input_to_state(x, joint_dim, buffer_dim):
    n = joint_dim * buffer_dim
    return RobotState(q_buffer=x[:n], qd_buffer=x[n:2 * n], act_buffer=x[2 * n:], q_dim=joint_dim)


def sample_rollout_datasets(datasets: Sequence, horizon, num_rollouts, key, joint_dim, buffer_dim,
                            concat: Dataset = None):
    """Sample `num_rollouts` contiguous (horizon+1)-length windows, length-weighted
    across datasets and never crossing a dataset boundary -> fixed-shape batch.
    `concat` may pass a precomputed Dataset.concatenate(*datasets) so callers can
    hoist the concatenation out of a per-batch loop (identical sampling either way)."""
    H = horizon + 1
    lengths = jnp.array([len(d) for d in datasets], dtype=jnp.int32)
    pmf = lengths / jnp.sum(lengths)
    offsets = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(lengths)[:-1]])

    key, rng = jax.random.split(key)
    didx = jax.random.choice(rng, jnp.arange(lengths.shape[0]), shape=(num_rollouts,), p=pmf)
    keys = jax.random.split(rng, num_rollouts)
    local = jax.vmap(lambda d, k: jax.random.randint(k, (), 0, lengths[d] - H))(didx, keys)
    starts = offsets[didx] + local

    data = Dataset.concatenate(*datasets) if concat is None else concat

    def sample(start):
        inp = jax.lax.dynamic_slice_in_dim(data.input, start, H, axis=0)   # (H, 3*jd*bd)
        state = jax.vmap(lambda x: _input_to_state(x, joint_dim, buffer_dim))(inp)
        return state, state.act_buffer[:, -joint_dim:]                     # (H, jd)

    states, actions = jax.vmap(sample)(starts)
    return key, states, actions

