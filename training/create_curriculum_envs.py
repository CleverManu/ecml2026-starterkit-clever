import os
from pathlib import Path
from flatland.envs.persistence import RailEnvPersister
from training.env_factory import make_training_env

SAMPLING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling")
DEFAULT_SCENARIO = os.path.join(SAMPLING_DIR, "level_0_scenario_1.pkl")

OUT_DIR = Path("training/curriculum")

scenes = ['scene_1', 'scene_4', 'scene_5']
n_agents = [1, 10, 25]
line_lengths = [2, 3, 4]

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_order = 0
    for line_length in line_lengths:
        for n_agent in n_agents:
            for scene in scenes:
                env = make_training_env(
                    scenario_path=DEFAULT_SCENARIO,
                    line_length=line_length,
                    scene=scene,
                    obs_version="v1",
                    n_agents_range=(n_agent, n_agent),
                )
                env.reset()
                n_order_str = f"{n_order:02d}"
                out = OUT_DIR / f"{n_order_str}_{scene}_ll-{line_length}_a-{n_agent}.pkl"
                RailEnvPersister.save(env, str(out))
                n_order += 1
                print(f"saved {out}")