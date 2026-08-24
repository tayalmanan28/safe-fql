# Safe Flow Q-Learning: Offline Safe Reinforcement Learning with Reachability-Based Flow Policies
Reinforcement Learning Conference (RLC), 2026

[**[Arxiv]**](https://arxiv.org/abs/2603.15136)

Mumuksh Tayal\*, Manan Tayal\*, Ravi Prakash

\*Equal contribution

SafeFQL tackles offline safe reinforcement learning by combining a Hamilton-Jacobi reachability-inspired safety value function with an efficient one-step flow policy. Safety constraints are learned through a self-consistent Bellman recursion, policies are trained via behavioral cloning, and the resulting policy is distilled for immediate deployment without rejection sampling. Compared to diffusion-style safe generative baselines, SafeFQL achieves substantially lower inference latency while matching or exceeding their performance, reducing constraint violations across navigation and robotics tasks.

# Methods
SafeFQL decouples the safety-constrained offline RL problem into:

- Offline learning of a reachability-based safety value function via self-consistent Bellman recursion;
- Optimal advantage learning;
- Policy extraction via a one-step flow policy, trained with behavioral cloning and distilled for fast, rejection-sampling-free deployment.

## Installation
``` Bash
conda create -n SafeFQL python=3.9
conda activate SafeFQL
cd safefql
pip install -r requirements.txt
```

## Main results
Run
``` Bash
bash scripts/run.sh
```
which trains SafeFQL_Base on ``OfflineCarButton1Gymnasium-v0`` (``env_id`` 0). See [scripts/run.sh](scripts/run.sh) for the full command and [env/env_list.py](env/env_list.py) for the full list of environments to swap in.

## Feasible Region Visualization
We need to download the necessary offline dataset for the ``Point Robot`` and ``Boat Robot`` environments from our [Hugging Face dataset](https://huggingface.co/datasets/tayalmanan/SafeBoatData). Training the SafeFQL_Base agent in the ``Point Robot`` environment
``` Bash
python launcher/examples/train_offline.py --env_id 29 --config configs/train_config.py:safefql_base
```
Then visualize the feasible region by running [launcher/viz/viz_map.py](launcher/viz/viz_map.py) (or [launcher/viz/viz_boat_vc.py](launcher/viz/viz_boat_vc.py) for the ``Boat Robot`` environment).

## Scripts
All training helper scripts (sweeps, benchmarks, and the default run command) live in [scripts/](scripts/) and must be run from the repository root, since they reference paths such as ``launcher/``, ``configs/``, and ``logs/`` relative to the current working directory:
``` Bash
bash scripts/train_bullet_envs.sh             # train all Bullet-Safety-Gym envs
bash scripts/train_bullet_envs_final.sh       # final per-env safety_weight settings
bash scripts/sweep_safety_weight.sh           # safety_weight sweep
bash scripts/sweep_aggressive.sh              # aggressive safety_weight sweep
bash scripts/sweep_remaining.sh               # remaining envs sweep
bash scripts/sweep_alpha_ant.sh               # alpha sweep for Ant
bash scripts/sweep_alpha_halfcheetah.sh       # alpha sweep for HalfCheetah
bash scripts/sweep_alpha_remaining_mujoco.sh  # alpha sweep for remaining MuJoCo envs
bash scripts/benchmark_training_time.sh       # training-time benchmark on BoatRobot
```

## Bibtex

If you find our code and paper helpful, please cite our paper as:
```
@inproceedings{tayal2026safefql,
title={Safe Flow Q-Learning: Offline Safe Reinforcement Learning with Reachability-Based Flow Policies},
author={Tayal, Mumuksh and Tayal, Manan and Prakash, Ravi},
booktitle={Reinforcement Learning Conference (RLC)},
year={2026},
url={https://arxiv.org/abs/2603.15136}
}
```

## Acknowledgements

This codebase builds on [FISOR](https://github.com/ZhengYinan-AIR/FISOR), and parts of its code are adapted from [IDQL](https://github.com/philippe-eecs/IDQL) and [DRPO](https://github.com/ManUtdMoon/Distributional-Reachability-Policy-Optimization).
