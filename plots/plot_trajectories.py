"""Plot trajectories from a trained SafeFQL (or SafeFQL_Base) agent on PointRobot."""
import os, sys, re, json
sys.path.append(".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ml_collections import ConfigDict
from env.point_robot import PointRobot
from jaxrl5.agents import SafeFQL_Base, SafeFQL


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def collect_trajectory(agent, env, init_state=None):
    """Roll out one episode, return list of (x, y) and metadata."""
    obs = env.reset(state=init_state)
    positions = [env.state[:2].copy()]
    total_reward, total_cost = 0.0, 0.0
    for _ in range(env._max_episode_steps):
        action, agent = agent.eval_actions(obs)
        obs, reward, done, info = env.step(action)
        positions.append(env.state[:2].copy())
        total_reward += reward
        total_cost += info["violation"]
        if done:
            break
    return np.array(positions), total_reward, total_cost, agent


def main():
    model_location = sys.argv[1] if len(sys.argv) > 1 else \
        "results/PointRobot/SafeFQL_Base_FQL_feasibility_hj_N16_minqc_2026-02-17_s779_733"

    # Load config
    with open(os.path.join(model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = PointRobot(id=0, seed=0)

    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeFQL": SafeFQL}[model_cls_name]
    config_dict.pop("cost_scale", None)

    agent = model_cls.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )

    # Find latest checkpoint
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    numbers = {}
    for f in pickle_files:
        match = re.search(r"\d+", f)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, f)
    max_path = numbers[max(numbers.keys())]
    print(f"Loading checkpoint: {max_path}")
    agent = agent.load(max_path)

    # ------- Collect trajectories from different starting states -------
    init_states = [
        np.array([-1.8,  0.0, 2.0, np.pi / 4], dtype=np.float32),   # default feasible
        np.array([-2.7, -2.7, 2.0, np.pi / 2], dtype=np.float32),   # bottom-left
        np.array([ 0.0, -2.0, 1.5, np.pi / 3], dtype=np.float32),   # mid-bottom
        np.array([-2.0,  2.0, 1.0, 0.0],       dtype=np.float32),   # top-left
        np.array([ 1.0, -1.0, 2.0, np.pi / 2], dtype=np.float32),   # near hazard
        np.array([-1.0, -1.5, 1.5, np.pi / 4], dtype=np.float32),
        np.array([ 0.5,  0.0, 2.0, np.pi / 4], dtype=np.float32),
        np.array([-2.5,  1.0, 1.0, -np.pi / 6], dtype=np.float32),
    ]

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw environment (hazards + goal)
    ax = env.plot_task(ax)

    print(f"Rolling out {len(init_states)} trajectories ...")
    for idx, s0 in enumerate(init_states):
        traj, ret, cost, agent = collect_trajectory(agent, env, init_state=s0)
        color = cmap(idx % 10)
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.5, alpha=0.85)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=7, zorder=5)   # start
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color=color, markersize=9, mew=2.5, zorder=5)  # end
        print(f"  Traj {idx+1}: steps={len(traj)-1:3d}  return={ret:.3f}  cost={cost:.1f}")

    # Mark goal explicitly
    goal = Circle(env.goal_position, env.goal_size, fill=True, alpha=0.6,
                  color=(0.35, 0.66, 0.35), label="Goal")
    ax.add_patch(goal)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{model_cls_name} – PointRobot Trajectories")

    out_dir = os.path.join(model_location, "imgs")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "trajectories.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
