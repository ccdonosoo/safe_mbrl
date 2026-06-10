import jax
import jax.numpy as jnp

from rl_algorithms_lib.utils.replay_buffer_jax import Transition
from rl_algorithms_lib.algorithms.mpc.base import Optimizer

from typing import Callable, Optional

class CEM:

    def __init__(self,
                 env,
                 method: str,
                 horizon: int,
                 nb_samples: int,
                 nb_elites: int,
                 alpha: float,
                 nb_steps: int,
                 **kwargs):
        
        self.env = env
        self.method = method
        self.horizon = horizon
        self.nb_samples = nb_samples
        self.nb_elites = nb_elites
        self.alpha = alpha
        self.nb_steps = nb_steps

        self.action_dim = (horizon, env.action_size)
        

        optimize_fn = self.set_optimizer_fn(**kwargs)

        policy = lambda env_state, rng: optimize_action_sequence(
                                            env,
                                            optimize_fn,
                                            rollout_env,
                                            horizon,
                                            env_state,
                                            rng)
        
        if method in ["Gaussian", "TanhGaussian", "TanhGaussian-TC"]:
            policy = lambda env_state, mean, rng: optimize_action_sequence_gaussian(
                                                    env,
                                                    optimize_fn,
                                                    rollout_env,
                                                    horizon,
                                                    env_state,
                                                    rng,
                                                    mean)
        else:
            policy = lambda env_state, rng: optimize_action_sequence(
                                            env,
                                            optimize_fn,
                                            rollout_env,
                                            horizon,
                                            env_state,
                                            rng)
        
        self.policy = policy


    def set_optimizer_fn(self, **kwargs):

        if self.method == "Gaussian":

            cem = CrossEntropyOptimizer(
                        num_samples=self.nb_samples,
                        num_elites=self.nb_elites,
                        num_steps=self.nb_steps,
                        alpha=self.alpha,
                        action_dim=self.action_dim,
                        **kwargs)
            

        elif self.method == "Categorical":

            cem = CrossEntropyOptimizerDiscrete(
                num_samples=self.nb_samples,
                num_elites=self.nb_elites,
                num_steps=self.nb_steps,
                alpha= self.alpha,
                action_dim=self.action_dim,
                **kwargs
            )
        
        elif self.method == "TanhGaussian":

            cem = CrossEntropyOptimizerTanh(
                num_samples=self.nb_samples,
                num_elites=self.nb_elites,
                num_steps=self.nb_steps,
                alpha= self.alpha,
                action_dim=self.action_dim,
                **kwargs
            )

        elif self.method == "TanhGaussian-TC":

            cem = CrossEntropyOptimizerTanhTC(
                num_samples=self.nb_samples,
                num_elites=self.nb_elites,
                num_steps=self.nb_steps,
                alpha= self.alpha,
                action_dim=self.action_dim,
                **kwargs
            )
        else:

            assert False, "Method not implemented, enter a valid method"

        return cem.optimize



    
def rollout_env(env, env_state, action_seq, horizon):

    def step(carry, ins):
        env_state = carry[0]
        acs = ins[-1]
        acs = jnp.array(acs, dtype=jnp.float32)
        env_state = env.step(env_state, acs)
        carry = [env_state]
        outs = [env_state,  env_state.reward, acs]
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
                            mean):


    eval_func = lambda seq: sample_rollout(
        env,
        env_state,
        seq,
        horizon)

    def sum_rewards(seq):
        transition = eval_func(seq)
        return transition.reward.mean()

    action_seq, reward = optimize_fn(
        func=sum_rewards,
        rng=optimizer_key,
        mean=mean
    )
    return action_seq, reward

def optimize_action_sequence(env,
                            optimize_fn,
                            sample_rollout,
                            horizon,
                            env_state,
                            optimizer_key):


    eval_func = lambda seq: sample_rollout(
        env,
        env_state,
        seq,
        horizon)

    def sum_rewards(seq):
        transition = eval_func(seq)
        return transition.reward.mean()

    action_seq, reward = optimize_fn(
        func=sum_rewards,
        rng=optimizer_key
    )
    return action_seq, reward


        


class CrossEntropyOptimizer(Optimizer):
    "The state and horizon dim are represented with tuples"
    def __init__(self, 
                 num_samples,
                 num_elites,
                 num_steps: int=10,
                 seed: int = 0,
                 init_std: float = 5.0,
                 alpha: float = 0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_steps = num_steps
        self.alpha = alpha
        self.seed = seed
        assert 0 <= alpha < 1, "alpha must be between [0, 1]"
        self.init_std = init_std

    def step(self, func, mean, std, key):
        mean = mean.reshape(-1, 1).squeeze()
        std = std.reshape(-1, 1).squeeze()
        samples = mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean))*std,
            shape=(self.num_samples,) 
        ) 
        samples = samples.reshape((self.num_samples,) + self.action_dim)
        values = jax.vmap(func)(samples)
        best_elite_idx = jax.numpy.argsort(values, axis=0).squeeze()[-self.num_elites:]

        elites = samples[best_elite_idx]
        elites_values = values[best_elite_idx]
        return elites, elites_values
    
    def optimize(self, func: Callable, rng:Optional[int]=None,
                  mean:Optional[jax.Array] = None):
        
        best_value = -jnp.inf
        if mean is None:
            mean = jnp.zeros(self.action_dim)
        else:
            assert mean.shape == self.action_dim

        std = jnp.ones(self.action_dim) * self.init_std
        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.key(self.seed)
        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, mu, sig, sample_key)
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)
            mean = mu * self.alpha + (1 - self.alpha) * elite_mean
            var = jnp.square(sig) * self.alpha + (1 - self.alpha) * elite_var
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
            carry = [key, mean, std, best_val, best_seq]

            return carry, outs
        carry = [rng, mean, std, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]
    

class CrossEntropyOptimizerTanh(Optimizer):
    "The state and horizon dim are represented with tuples"
    def __init__(self, 
                 num_samples,
                 num_elites,
                 num_steps: int=10,
                 seed: int = 0,
                 init_std: float = 5.0,
                 alpha: float = 0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_steps = num_steps
        self.alpha = alpha
        self.seed = seed
        assert 0 <= alpha < 1, "alpha must be between [0, 1]"
        self.init_std = init_std

    def step(self, func, mean, std, key):
        mean = mean.reshape(-1, 1).squeeze()
        std = std.reshape(-1, 1).squeeze()
        samples = jnp.tanh(mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean))*std,
            shape=(self.num_samples,) 
        ) )
        samples = samples.reshape((self.num_samples,) + self.action_dim)
        values = jax.vmap(func)(samples)
        best_elite_idx = jax.numpy.argsort(values, axis=0).squeeze()[-self.num_elites:]

        elites = samples[best_elite_idx]
        elites_values = values[best_elite_idx]
        return elites, elites_values
    
    def optimize(self, func: Callable, rng:Optional[int]=None,
                  mean:Optional[jax.Array] = None):
        
        best_value = -jnp.inf
        if mean is None:
            mean = jnp.zeros(self.action_dim)
        else:
            assert mean.shape == self.action_dim

        std = jnp.ones(self.action_dim) * self.init_std
        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)
        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, mu, sig, sample_key)
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)
            mean = mu * self.alpha + (1 - self.alpha) * elite_mean
            var = jnp.square(sig) * self.alpha + (1 - self.alpha) * elite_var
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
            carry = [key, mean, std, best_val, best_seq]

            return carry, outs
        carry = [rng, mean, std, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]


class CrossEntropyOptimizerTanhTC(Optimizer):
    "The state and horizon dim are represented with tuples"
    def __init__(self, 
                 num_samples,
                 num_elites,
                 num_steps: int=10,
                 seed: int = 0,
                 init_std: float = 5.0,
                 alpha: float = 0.1,
                 window_length: int =10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_steps = num_steps
        self.alpha = alpha
        self.seed = seed
        self.window_length = window_length
        assert 0 <= alpha < 1, "alpha must be between [0, 1]"
        self.init_std = init_std

    def step(self, func, mean, std, key):
        mean = mean.reshape(-1, 1).squeeze()
        std = std.reshape(-1, 1).squeeze()
        samples = jnp.tanh(mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean))*std,
            shape=(self.num_samples,) 
        ) )

        window = jnp.ones(self.window_length) / self.window_length

        samples = samples.reshape((self.num_samples,) + self.action_dim)

        sampling_fn = jax.vmap(jax.vmap(lambda s: jnp.convolve(s, window, mode='same'), \
                                         in_axes=1, out_axes=1), in_axes=0, out_axes=0)
        
        samples = sampling_fn(samples)
        values = jax.vmap(func)(samples)
        best_elite_idx = jax.numpy.argsort(values, axis=0).squeeze()[-self.num_elites:]
        
        elites = samples[best_elite_idx]
        elites_values = values[best_elite_idx]
        return elites, elites_values
    
    def optimize(self, func: Callable, rng:Optional[int]=None,
                  mean:Optional[jax.Array] = None):
        
        best_value = -jnp.inf
        if mean is None:
            mean = jnp.zeros(self.action_dim)
        else:
            assert mean.shape == self.action_dim

        std = jnp.ones(self.action_dim) * self.init_std
        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)
        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, mu, sig, sample_key)
            elite_mean = jnp.mean(jnp.arctanh(elites), axis=0)
            elite_var = jnp.var(jnp.arctanh(elites), axis=0)
            mean = mu * self.alpha + (1 - self.alpha) * elite_mean
            var = jnp.square(sig) * self.alpha + (1 - self.alpha) * elite_var
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
            carry = [key, mean, std, best_val, best_seq]

            return carry, outs
        carry = [rng, mean, std, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]


class CrossEntropyOptimizerDiscrete(Optimizer):
    "The state and horizon dim are represented with tuples"
    def __init__(self, 
                 num_samples,
                 num_elites,
                 num_steps: int=10,
                 seed: int = 0,
                 num_categories: int = 3,
                 index_start: int = -1,
                 alpha: float = 0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_steps = num_steps
        self.alpha = alpha
        self.seed = seed
        self.index_start = index_start
        assert 0 <= alpha < 1, "alpha must be between [0, 1]"
        self.num_categories = num_categories

    def step(self, func, theta, key):
        #Reshape distribution parameters TODO
        theta = theta.reshape(self.num_categories, -1, 1).squeeze()
        key, rng = jax.random.split(key, 2)
        subkeys = jax.random.split(rng, theta.shape[1])

        samples = jax.vmap(lambda key, theta: \
                  jax.random.choice(key, jnp.arange(self.num_categories)+ \
                  self.index_start, (self.num_samples,), p=theta), in_axes=(0, 1),\
                  out_axes=1)(subkeys, theta)
        
        samples = samples.reshape((self.num_samples,) + self.action_dim) 
        values = jax.vmap(func)(samples)
        best_elite_idx = jax.numpy.argsort(values, axis=0).squeeze()[-self.num_elites:]
        elites = samples[best_elite_idx]
        elites_values = values[best_elite_idx]
        return elites, elites_values
    
    def optimize(self, func: Callable, rng:Optional[int]=None):
        
        best_value = -jnp.inf

        theta = jnp.ones((self.num_categories,) + self.action_dim)/self.num_categories

        best_sequence = jnp.zeros(self.action_dim, dtype=jnp.int32)
        get_best_action = lambda best_val, best_seq, val, seq: [val[-1].squeeze(), seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.key(self.seed)
        def step(carry, ins):
            key = carry[0]
            theta=carry[1]
            best_val = carry[2]
            best_seq = carry[3]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, theta, sample_key)
            elite_theta = jax.vmap(jax.vmap(lambda t, x: (t==x).sum()/self.num_samples,
                         in_axes=(1, None)), in_axes=(None, 0)
                         )(elites.reshape(self.num_elites, -1, 1).squeeze(), 
                           jnp.arange(self.num_categories))
            elite_theta = elite_theta.reshape((self.num_categories,) + self.action_dim)
            theta = theta * self.alpha + (1 - self.alpha) * elite_theta
            # END HERE
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
            carry = [key, theta, best_val, best_seq]

            return carry, outs
        carry = [rng, theta, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]
