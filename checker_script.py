# ----- DIAGNOSTICS (paste after creating `envs`) -----

import gymnasium as gym, sys



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        render = "rgb_array" if (capture_video and idx == 0) else None
        env = gym.make(env_id, render_mode=render)
        #env = gym.wrappers.TimeLimit(env, max_episode_steps = 1000))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        obs, info = env.reset(seed=seed + idx)  # gymnasium reset signature
        return env
    return thunk


envs = gym.vector.SyncVectorEnv(
        [make_env("Pendulum-v1", 42, 0, False, "Fuck")]
    )


print("Gymnasium version:", getattr(gym, "__version__", "unknown"))
print("Env ID:", "Pendulum-v1")
try:
    spec = gym.spec("Pendulum-v1")
    print("Spec max_episode_steps:", getattr(spec, "max_episode_steps", None))
except Exception as e:
    print("spec() failed:", e)

# See the actual underlying single env and whether TimeLimit wrapped it
try:
    base = envs.envs[0]
    print("Underlying env type:", type(base).__name__)
    # If TimeLimit exists, it will have these attrs
    print("Has TimeLimit attrs:",
          hasattr(base, "max_episode_steps"),
          getattr(base, "max_episode_steps", None))
except Exception as e:
    print("Introspect envs.envs[0] failed:", e)

# Quick “does truncation ever happen?” probe with random actions
# (run a short loop separate from your training to verify)
def _probe_vector_done(envs, max_checks=1500):
    import numpy as np
    o, info = envs.reset(seed=42)
    for t in range(1, max_checks + 1):
        a = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)], dtype=np.float32)
        o, r, term, trunc, info = envs.step(a)
        if bool(term.any() or trunc.any()):
            print(f"[PROBE] done at step {t} | term.any={bool(term.any())} trunc.any={bool(trunc.any())}")
            return True
    print("[PROBE] never saw done within", max_checks, "steps")
    return False

_probe_vector_done(envs, max_checks=2000)
# ------------------------------------------------------

