"""
Model Predictive Path Integral (MPPI) optimizer for MPC.

Sampling-based controller that uses importance-weighted averaging of trajectories
instead of elite selection (as in CEM/ICEM). The soft weighting produces smoother
action sequences and better exploration of the action space.

Reference:
    Williams, G., Aldrich, A., Theodorou, E. A. (2017).
    "Information Theoretic MPC for Model-Based Reinforcement Learning."
    IEEE International Conference on Robotics and Automation (ICRA).
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional, NamedTuple
from functools import partial
from flax import nnx
from jax.numpy.fft import irfft, rfftfreq


class MPPIHyperparams(NamedTuple):
    num_samples: int = 500
    temperature: float = 1.0
    init_std: float = 0.5
    num_steps: int = 1
    colored_noise: bool = False
    noise_exponent: float = 2.0
    momentum: float = 0.0


class MPPI(nnx.Module):
    """
    MPPI optimizer for Koopman MPC.

    Matches the ICEM interface: optimize(func, rng, initial_actions, clip_limits)
    returns (best_action_seq, best_value).

    Instead of selecting elite samples and fitting a Gaussian (CEM/ICEM),
    MPPI computes a weighted average over ALL samples using softmax weights
    derived from trajectory costs. The temperature parameter controls
    exploration vs exploitation.

    Parameters
    ----------
    env : Environment
        Environment with action_size attribute.
    horizon : int
        Planning horizon.
    nb_samples : int
        Number of action sequences sampled per iteration.
    temperature : float
        Softmax temperature (lambda). Lower = greedier (closer to best sample).
        Higher = more uniform weighting (more exploration).
    init_std : float
        Initial standard deviation for noise sampling.
    nb_steps : int
        Number of MPPI refinement iterations per call.
    colored_noise : bool
        If True, use (1/f)^beta colored noise for temporally smooth perturbations.
        If False (default), use standard Gaussian white noise (classical MPPI).
    noise_exponent : float
        Power-law exponent for colored noise (only used if colored_noise=True).
        2 = brown noise (smooth), 1 = pink noise.
    momentum : float
        Blending factor for mean update: new_mean = momentum * old_mean + (1 - momentum) * weighted_mean.
    action_repeat : int
        Number of times each action is repeated in the environment.
    """

    def __init__(self,
                 env,
                 horizon: int,
                 nb_samples: int = 500,
                 temperature: float = 1.0,
                 init_std: float = 0.5,
                 nb_steps: int = 1,
                 colored_noise: bool = False,
                 noise_exponent: float = 2.0,
                 momentum: float = 0.0,
                 action_repeat: int = 1,
                 **kwargs):

        self.env = env
        self.horizon = horizon
        self.nb_samples = nb_samples
        self.opt_dim = (horizon, env.action_size)
        self.action_repeat = action_repeat

        self.opt_params = MPPIHyperparams(
            num_samples=nb_samples,
            temperature=temperature,
            init_std=init_std,
            num_steps=nb_steps,
            colored_noise=colored_noise,
            noise_exponent=noise_exponent,
            momentum=momentum,
        )

        policy = lambda env_state, init_actions, rng: optimize_action_sequence(
            env,
            self.optimize,
            rollout_env,
            horizon,
            env_state,
            rng,
            initial_actions=init_actions,
            action_repeat=action_repeat,
            clip_limits=(-1., 1.))

        self.policy = policy

    def optimize(self, func: Callable, rng: Optional[jax.Array] = None,
                 initial_actions: Optional[jax.Array] = None,
                 clip_limits: Optional[tuple] = (-1., 1.)):
        """
        Maximize func using MPPI with importance-weighted averaging.

        Parameters
        ----------
        func : Callable
            Objective function mapping action sequence (H, action_dim) -> scalar reward.
        rng : jax.Array
            Random key.
        initial_actions : jax.Array, optional
            Warm-start mean action sequence, shape (H, action_dim).
        clip_limits : tuple
            (min, max) for action clipping.

        Returns
        -------
        best_action_seq : jax.Array
            Weighted-average action sequence, shape (H, action_dim).
        best_value : float
            Reward of the best individual sample.
        """
        if initial_actions is None:
            mean = jnp.zeros(self.opt_dim)
        else:
            mean = initial_actions

        rng, optimizer_key = jax.random.split(rng, 2)

        def step(carry, _):
            key, mu = carry
            opt_size = self.opt_dim[0] * self.opt_dim[1]

            # Sample perturbations
            key, noise_key = jax.random.split(key)

            # Standard MPPI: white Gaussian noise
            # Optional: colored noise for temporally smooth perturbations
            noise = jax.lax.cond(
                self.opt_params.colored_noise,
                lambda k: _colored_noise_batch(self.opt_params.noise_exponent,
                                               opt_size,
                                               self.opt_params.num_samples, k),
                lambda k: jax.random.normal(k, shape=(self.opt_params.num_samples, opt_size)),
                noise_key,
            )

            # Perturb mean with scaled noise
            action_samples = mu.ravel() + noise * self.opt_params.init_std
            action_samples = jnp.clip(action_samples, clip_limits[0], clip_limits[1])
            action_samples = action_samples.reshape((-1,) + self.opt_dim)

            # Evaluate all samples
            values = jax.vmap(func)(action_samples)

            # Compute importance weights via softmax
            weights = _softmax_weights(values, self.opt_params.temperature)

            # Weighted average of action sequences
            weighted_mean = jnp.sum(
                weights[:, None, None] * action_samples, axis=0
            )
            weighted_mean = jnp.clip(weighted_mean, clip_limits[0], clip_limits[1])

            # Apply momentum
            new_mean = self.opt_params.momentum * mu + (1.0 - self.opt_params.momentum) * weighted_mean

            best_value = jnp.max(values)
            best_idx = jnp.argmax(values)
            best_seq = action_samples[best_idx]

            carry = (key, new_mean)
            outs = (best_value, best_seq, new_mean)
            return carry, outs

        carry = (optimizer_key, mean)
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.opt_params.num_steps)

        # Return the weighted mean (smoother) as the action sequence,
        # and the best individual reward as the value
        final_mean = outs[2][-1]
        best_value = outs[0][-1]
        return final_mean, best_value


def _softmax_weights(values, temperature):
    """Compute normalized importance weights from rewards using softmax.

    Shifts values by max for numerical stability before applying softmax.
    """
    shifted = (values - jnp.max(values)) / temperature
    exp_vals = jnp.exp(shifted)
    return exp_vals / jnp.sum(exp_vals)


def _colored_noise_batch(exponent, size, num_samples, rng):
    """Generate a batch of (1/f)^beta colored noise samples.

    Uses the Timmer & Koenig (1995) algorithm, same as icem.py.
    """
    keys = jax.random.split(rng, num_samples)
    return jax.vmap(lambda k: _powerlaw_noise_single(exponent, size, k))(keys)


@partial(jax.jit, static_argnums=(0, 1))
def _powerlaw_noise_single(exponent: float, size: int, rng: jax.Array) -> jax.Array:
    """Generate a single (1/f)^beta colored noise signal, unit variance."""
    f = rfftfreq(size)
    fmin = 1.0 / size
    s_scale = jnp.maximum(f, fmin)
    s_scale = s_scale ** (-exponent / 2.0)

    w = s_scale[1:]
    w = w.at[-1].set(w[-1] * (1 + (size % 2)) / 2.0)
    sigma = 2 * jnp.sqrt(jnp.sum(w ** 2)) / size

    key_r, key_i = jax.random.split(rng, 2)
    sr = jax.random.normal(key=key_r, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_i, shape=s_scale.shape) * s_scale

    si = si.at[-1].set(0)
    sr = sr.at[-1].set(sr[-1] * jnp.sqrt(2))
    si = si.at[0].set(0)
    sr = sr.at[0].set(sr[0] * jnp.sqrt(2))

    s = sr + 1j * si
    y = irfft(s, n=size, axis=-1) / jnp.maximum(sigma, 1e-8)
    return y


def rollout_env(env, env_state, action_seq, horizon, action_repeat):
    """Rollout environment with action sequence. Identical to icem.py rollout."""
    from rl_algorithms_lib.utils.replay_buffer_jax import Transition

    def step(carry, ins):
        env_state = carry[0]
        acs = ins[-1]
        acs = jnp.array(acs, dtype=jnp.float32)

        def repeat_step(carry, ins):
            env_state = carry
            env_state = env.step(env_state, acs)
            return env_state, env_state.reward

        env_state, reward = jax.lax.scan(repeat_step, init=(env_state), length=action_repeat)

        env_state = env.step(env_state, acs)
        carry = [env_state]
        outs = [env_state, reward.mean(), acs]
        return carry, outs

    ins = []
    ins.append(action_seq)
    carry = [env_state]
    _, outs = jax.lax.scan(step, carry, ins, length=horizon)

    next_state = jax.vmap(lambda x: x.ravel())(outs[0].obs)
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(env_state.obs.ravel())
    state = state.at[1:, ...].set(next_state[:-1, ...])
    rewards = outs[1].reshape(-1, 1)
    acs = outs[-1]

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(state),
        action=flatten(acs),
        reward=rewards,
        next_obs=flatten(next_state),
        done=flatten(jnp.zeros_like(rewards)),
    )
    return transitions


def optimize_action_sequence(env,
                             optimize_fn,
                             sample_rollout,
                             horizon,
                             env_state,
                             optimizer_key,
                             action_repeat=1,
                             initial_actions=None,
                             clip_limits=(-1., 1.)):
    """Wrapper matching icem.py's optimize_action_sequence_gaussian."""

    eval_func = lambda seq: sample_rollout(
        env,
        env_state,
        seq,
        horizon,
        action_repeat)

    def sum_rewards(seq):
        transition = eval_func(seq)
        return transition.reward.mean()

    action_seq, reward = optimize_fn(
        func=sum_rewards,
        rng=optimizer_key,
        initial_actions=initial_actions,
        clip_limits=clip_limits,
    )
    return action_seq, reward
