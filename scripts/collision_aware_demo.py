"""Drive the collision-aware env with a scripted valve command and watch the raycast.

    python scripts/collision_aware_demo.py [--steps 400] [--sampling mean|member|normal] [--headless]
"""
import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from safe_mbrl.envs.collision_aware import CollisionAwareEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--sampling", choices=["mean", "member", "normal"], default="mean")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = CollisionAwareEnv(sampling=args.sampling)
    env.scatter_rocks(6, area=((2.5, 6.5), (-3.0, 3.0)), radius=(0.35, 0.9), seed=7)
    obs = env.reset()
    print(f"dt {env.dt} s | {len(env.rocks)} rocks | camera {env.camera.width}x{env.camera.height} | "
          f"rock tops {[round(r.top, 2) for r in env.rocks.rocks]} m")

    for k in range(args.steps):
        t = k * env.dt
        action = np.array([0.3 * np.sin(0.3 * t), -0.4 * np.sin(0.5 * t), 0.4 * np.sin(0.4 * t),
                           0.0, 0.3 * np.sin(0.6 * t)], np.float32)
        obs = env.step(action)
        print(obs)
        if not args.headless and not env.render():
            break
        if k % 25 == 0:
            hits = np.bincount(obs.cloud.hit_id[obs.valid == 1], minlength=len(env.rocks) + 1)
            print(f"t={obs.time:5.2f}s q={np.round(obs.q, 2)} tip={np.round(obs.tip, 2)} "
                  f"clearance={np.round(obs.clearance, 2)} floor={obs.floor_clearance:.2f} "
                  f"hits floor/rocks={hits.tolist()} collided={obs.collided}")
