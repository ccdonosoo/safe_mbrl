import jax


import jax.numpy as jnp
from functools import partial

from jax.numpy.fft import irfft, rfftfreq

from typing import NamedTuple

from typing import Callable, Optional

from flax import nnx


@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(exponent: float, size: int, rng: jax.Array, fmin: float = 0) -> jax.Array:
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    s_scale = f
    ix = jnp.sum(s_scale < fmin)  # Index of the cutoff

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, start_indices=(idx,), slice_sizes=(1,))
        y = jnp.ones_like(x) * x_idx
        indexes = jnp.arange(0, x.shape[0], step=1)
        first_idx = indexes < idx
        z = (1 - first_idx) * x + first_idx * y
        return z

    def no_cutoff(x, idx):
        return x

    s_scale = jax.lax.cond(
        jnp.logical_and(ix < len(s_scale), ix),
        cutoff,
        no_cutoff,
        s_scale,
        ix
    )
    s_scale = s_scale ** (-exponent / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.)  # correct f = +-0.5
    sigma = 2 * jnp.sqrt(jnp.sum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(jnp.newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    key_sr, key_si, rng = jax.random.split(rng, 3)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * jnp.sqrt(2))  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * jnp.sqrt(2))  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


class ICEMHyperparams(NamedTuple):
    """
    maxiter: maximum iterations.
    grad_norm_threshold: tolerance for stopping optimization.
    make_psd: whether to zero negative eigenvalues after quadratization.
    psd_delta: The delta value to make the problem PSD. Specifically, it will
        ensure that d^2c/dx^2 and d^2c/du^2, i.e. the hessian of cost function
        with respect to state and control are always positive definite.
    alpha_0: initial line search value.
    alpha_min: minimum line search value.
    """
    num_samples: int = 500
    num_elites: int = 50
    init_std: float = 0.5
    alpha: float = 0.0
    num_steps: int = 1
    exponent: float = 0.0
    elite_set_fraction: float = 0.3


class ICEM(nnx.Module):

    def __init__(self,
                env,
                horizon: int,
                nb_samples: int = 500,
                nb_elites: int = 50,
                init_std: float = 0.5,
                alpha: float = 0.0,
                nb_steps: int = 1,
                exponent: float = 0.0,
                elite_set_fraction: float = 0.3,
                action_repeat: int = 1,
                **kwargs):
        
        self.env = env

        self.horizon = horizon
        self.nb_samples = nb_samples
        self.nb_elites = nb_elites
        self.init_std = init_std
        self.alpha = alpha
        self.nb_steps = nb_steps

        self.opt_params = ICEMHyperparams(num_samples=nb_samples,
                                          num_elites=nb_elites,
                                          alpha=alpha,
                                          num_steps=nb_steps,
                                          init_std=init_std,
                                          exponent=exponent,
                                          elite_set_fraction=elite_set_fraction)

        self.opt_dim = (horizon, env.action_size)
        



        policy = lambda env_state, init_actions, rng: optimize_action_sequence_gaussian(
                                                env,
                                                self.optimize,
                                                rollout_env,
                                                horizon,
                                                env_state,
                                                rng,
                                                initial_actions=init_actions,
                                                action_repeat=action_repeat,
                                                clip_limits=(-1., 1.)) #TODO
        
        self.policy = policy

        self.action_repeat = action_repeat
        


    

    def optimize(self, func: Callable, rng:Optional[int]=None,
                    initial_actions:Optional[jax.Array] = None, 
                    clip_limits:Optional[jax.Array] = (-1.,1.)):
        
        best_value = -jnp.inf
        if initial_actions is None:
            mean = jnp.zeros(self.opt_dim)
        else:
            assert initial_actions.shape == self.opt_dim
            mean = initial_actions
        std = jnp.ones(self.opt_dim) * self.opt_params.init_std
        

        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.key(self.seed)
        
        rng, optimizer_key = jax.random.split(rng, 2)
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        num_prev_elites_per_iter = max(int(self.opt_params.elite_set_fraction * self.opt_params.num_elites), 1)

        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4] 
            prev_elites = carry[5]
            mu = mu.reshape(-1, 1).squeeze()
            sig = sig.reshape(-1, 1).squeeze()
            sampling_rng = jax.random.split(key=key, num=self.opt_params.num_samples + 1)
            key = sampling_rng[0]
            sampling_rng = sampling_rng[1:]
            opt_size = self.opt_dim[0] * self.opt_dim[1]
            colored_samples = jax.vmap(
                lambda rng: powerlaw_psd_gaussian(exponent=self.opt_params.exponent, size=opt_size, rng=rng))(
                sampling_rng)
            
            action_samples = mu + colored_samples * sig
            action_samples = jnp.clip(action_samples, clip_limits[0], clip_limits[1])
            action_samples = action_samples.reshape((-1,) + self.opt_dim)
            action_samples = jnp.concatenate([action_samples, prev_elites], axis=0)
            values = jax.vmap(func)(action_samples)
            best_elite_idx = jnp.argsort(values, axis=0).squeeze()[-self.opt_params.num_elites:]
            elites = action_samples[best_elite_idx]
            elite_values = values[best_elite_idx]
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)
            mean = mu.reshape(self.opt_dim) * self.opt_params.alpha + (1 - self.opt_params.alpha) * elite_mean
            var = jnp.square(sig.reshape(self.opt_dim)) * self.opt_params.alpha + (
                        1 - self.opt_params.alpha) * elite_var
            std = jnp.sqrt(var)
            best_elite = elite_values[-1].squeeze()
            bests = jax.lax.cond(best_val <= best_elite,
                                 get_best_action,
                                 get_curr_best_action,
                                 best_val,
                                 best_seq,
                                 elite_values,
                                 elites)
            best_val = bests[0]
            best_seq = bests[-1]
            outs = [best_val, best_seq]
            elite_set = jnp.atleast_2d(elites[-num_prev_elites_per_iter:]).reshape((-1,) + self.opt_dim)
            carry = [key, mean, std, best_val, best_seq, elite_set]
            return carry, outs


        std = jnp.ones(self.opt_dim) * self.opt_params.init_std
        best_sequence = mean
        if optimizer_key is None:
            optimizer_key = jax.random.key(self.seed)
        prev_elites = jnp.zeros((num_prev_elites_per_iter,) + self.opt_dim)
        carry = [optimizer_key, mean, std, best_value, best_sequence, prev_elites]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.opt_params.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]


def rollout_env(env, env_state, action_seq, horizon, action_repeat):

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
        outs = [env_state,  reward.mean(), acs]
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
    

def optimize_action_sequence_gaussian(env,
                            optimize_fn,
                            sample_rollout,
                            horizon,
                            env_state,
                            optimizer_key,
                            action_repeat=1,
                            initial_actions=None,
                            clip_limits=(-1., 1.)):


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

