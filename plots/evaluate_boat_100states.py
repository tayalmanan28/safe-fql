"""Evaluate boat agent on 100 initial states from Traj_points.csv"""
import argparse
import json
import os
import sys

sys.path.append(".")

import numpy as np
import jax
from env.boat_robot import BoatRobot
from jaxrl5.agents import SafeFQL_Base, SafeFQL, SafeIFQL, NormalizedSafeFQL


def find_checkpoint(model_location):
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    if not pickle_files:
        raise ValueError(f"No checkpoints found in: {model_location}")
    
    numbers = {}
    for filename in pickle_files:
        import re
        match = re.search(r"\d+", filename)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, filename)
    
    return numbers[max(numbers.keys())]


def build_init_states(env, csv_path):
    """Load initial states from Traj_points.csv"""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    
    if data.size == 0:
        raise ValueError(f"CSV has no rows: {csv_path}")
    
    names = list(data.dtype.names or [])
    lower_to_orig = {n.lower(): n for n in names}
    
    if "x" in lower_to_orig and "y" in lower_to_orig:
        x_col = lower_to_orig["x"]
        y_col = lower_to_orig["y"]
        init_states = np.stack([data[x_col], data[y_col]], axis=-1).astype(np.float32)
    else:
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw[None, :]
        init_states = raw[:, :2].astype(np.float32)
    
    # Filter out-of-bounds states
    low, high = env.observation_space.low, env.observation_space.high
    valid = np.logical_and(init_states >= low, init_states <= high).all(axis=1)
    if not np.all(valid):
        dropped = int((~valid).sum())
        print(f"Warning: dropping {dropped} out-of-bounds initial states from CSV")
        init_states = init_states[valid]
    
    return init_states


def evaluate_on_initial_states(agent, env, init_states, max_steps=400):
    """Evaluate agent on specific initial states"""
    episode_rets = []
    episode_costs = []
    episode_lens = []
    
    for init_state in init_states:
        obs = env.reset(state=init_state)
        episode_ret = 0.0
        episode_cost = 0.0
        episode_len = 0
        
        for _ in range(max_steps):
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_cost += cost
            episode_len += 1
            if done:
                break
        
        episode_rets.append(episode_ret)
        episode_costs.append(episode_cost)
        episode_lens.append(episode_len)
    
    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
        "num_episodes": len(init_states)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_location", type=str)
    parser.add_argument("--csv_path", type=str, default="Traj_points.csv")
    parser.add_argument("--checkpoint", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config_file = os.path.join(args.model_location, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    
    # Create environment
    env = BoatRobot(id=0, seed=0)
    
    # Create agent with config
    config_dict = dict(config['agent_kwargs'])
    model_cls = config_dict.pop("model_cls")
    
    # Handle legacy model class names
    legacy_mapping = {
        "FISOR_FQL": "SafeFQL",
        "FISOR": "SafeFQL_Base",
    }
    if model_cls in legacy_mapping:
        model_cls = legacy_mapping[model_cls]
    
    agent = globals()[model_cls].create(
        config['seed'], env.observation_space, env.action_space, **config_dict
    )
    
    # Load checkpoint
    checkpoint_path = find_checkpoint(args.model_location)
    print(f"Loading checkpoint: {checkpoint_path}")
    agent = agent.load(checkpoint_path)
    
    # Load initial states
    init_states = build_init_states(env, args.csv_path)
    print(f"Loaded {len(init_states)} initial states from {args.csv_path}")
    
    # Evaluate
    results = evaluate_on_initial_states(agent, env, init_states)
    
    print(f"\nResults on {len(init_states)} initial states:")
    print(f"  Return: {results['return']:.2f}")
    print(f"  Cost: {results['cost']:.2f}")
    print(f"  Episode Length: {results['episode_len']:.1f}")


if __name__ == "__main__":
    main()
