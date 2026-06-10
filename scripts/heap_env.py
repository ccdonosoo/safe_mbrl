from safe_mbrl.envs.heap_env.heap_example_env import HeapEnv

import numpy as np
from types import SimpleNamespace

cfg = SimpleNamespace(
    task="heap-eetracking",
    obs="state",
    n_envs=1,
    step_penalty_coef=0,
    action_penalty_coef=1,
    accel_penalty_coef=0,
    accel_sign_penalty_coef=0,
    run_baseline=False,
)
step_penalty_coef: 0 #10
action_penalty_coef: 1 #0.1
accel_penalty_coef: 0 #0.5
accel_sign_penalty_coef: 0 #0.1
env = HeapEnv(n_envs=1,
            use_act_net = True,
            n_history_steps= 10,
            n_ref_steps = 15,
            t_step = 0.04,
            t_traj = 6.0,
            cfg=cfg)

obs, info = env.reset(seed=1)
env.render()  # seed the render buffers with the initial state

# Full-episode rollout: the env's plot() assumes ref_traj_steps (151) points,
# so run until terminated and let render() record each step.
terminated = False
while not terminated:
    obs, rwd, terminated, truncated, info = env.step(np.random.randn(1, 4))
    env.render(done=terminated)  # flushes the episode into the histories on done

# Use the env's built-in 2D plotter -> writes <file_name>.png into save_dir.
env.render(mode="plot", save_dir=".", file_name="rollout_env")
print("saved rollout_env.png")