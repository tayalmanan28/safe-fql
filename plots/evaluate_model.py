"""Evaluate a trained SafeFQL_Base / SafeFQL model."""
import os, sys, re, json, argparse
sys.path.append(".")
import gymnasium as gym
from ml_collections import ConfigDict
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.agents import SafeFQL_Base, SafeFQL
from jaxrl5.agents import NormalizedSafeFQL
from jaxrl5.agents import SafeIFQL
from jaxrl5.evaluation import evaluate, evaluate_pr, evaluate_br
from jaxrl5.wrappers import wrap_gym


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SafeFQL_Base/SafeFQL model")
    parser.add_argument("model_location", type=str,
                        help="Path to the results directory containing config.json and model*.pickle")
    parser.add_argument("--N", type=int, default=None,
                        help="Override the number of action candidates at evaluation (default: use config value)")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint number to load (default: latest)")
    args = parser.parse_args()

    model_location = args.model_location

    # Load config
    with open(os.path.join(model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env_name = cfg.get("env_name", "PointRobot")

    if env_name == "PointRobot":
        env = PointRobot(id=0, seed=0)
        env_max_steps = env._max_episode_steps
    elif env_name == "BoatRobot":
        env = BoatRobot(id=0, seed=0)
        env_max_steps = env._max_episode_steps
    else:
        # DSRL / safety-gymnasium / bullet-safety-gym envs
        env = gym.make(env_name)
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=cfg["agent_kwargs"]["cost_limit"])

    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    # Backward compatibility with older checkpoints whose config.json still
    # uses the legacy class names.
    LEGACY_ALIASES = {
        "FISOR": "SafeFQL_Base",
        "FISOR_FQL": "SafeFQL",
        "FISOR_IFQL": "SafeIFQL",
    }
    model_cls_name = LEGACY_ALIASES.get(model_cls_name, model_cls_name)
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeFQL": SafeFQL, "NormalizedSafeFQL": NormalizedSafeFQL, "SafeIFQL": SafeIFQL}[model_cls_name]

    # Override N if requested
    if args.N is not None:
        print(f"Overriding N: {config_dict.get('N')} -> {args.N}")
        config_dict["N"] = args.N

    # Remove keys not accepted by create()
    config_dict.pop("cost_scale", None)

    config_dict["env_max_steps"] = env_max_steps

    agent = model_cls.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )

    # Find checkpoint
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    numbers = {}
    for f in pickle_files:
        match = re.search(r"\d+", f)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, f)

    if args.checkpoint is not None:
        if args.checkpoint not in numbers:
            raise ValueError(f"Checkpoint {args.checkpoint} not found. Available: {sorted(numbers.keys())}")
        ckpt_path = numbers[args.checkpoint]
    else:
        ckpt_path = numbers[max(numbers.keys())]

    print(f"Loading checkpoint: {ckpt_path}")
    agent = agent.load(ckpt_path)

    # Evaluate
    n_val = config_dict.get("N", "default")
    print(f"Evaluating {model_cls_name} on {env_name} for {args.num_episodes} episodes (N={n_val})...")
    if env_name == "PointRobot":
        eval_info = evaluate_pr(agent, env, args.num_episodes)
    elif env_name == "BoatRobot":
        eval_info = evaluate_br(agent, env, args.num_episodes)
    else:
        eval_info = evaluate(agent, env, args.num_episodes)

    print("\n=== Evaluation Results ===")
    for k, v in eval_info.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
