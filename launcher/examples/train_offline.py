import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import SafeFQL_Base, SafeFQL, SafeIFQL, NormalizedSafeFQL
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr, evaluate_br
import json


# ---------------------------------------------------------------------------
# Per-environment SafeFQL safety hyperparameters
# ---------------------------------------------------------------------------
# safety_weight: coefficient on the unsafe-branch penalty in the one-step
#                actor loss (`safety_weight * max(qc, 0)`).
# These were tuned per-env via sweeps; envs not listed fall back to defaults.
SAFEFQL_PER_ENV = {
    # DSRL velocity envs (binarized HJ cost, scale=25, ep_len=1000)
    'OfflineHalfCheetahVelocityGymnasium-v1': dict(safety_weight=10.0),
    'OfflineWalker2dVelocityGymnasium-v1':    dict(safety_weight=1000.0),
    'OfflineHopperVelocityGymnasium-v1':      dict(safety_weight=2.0),
    'OfflineSwimmerVelocityGymnasium-v1':     dict(safety_weight=0.1),
    'OfflineAntVelocityGymnasium-v1':         dict(safety_weight=10000.0),
    # Point/Boat (continuous HJ-margin cost, scale=1, short horizon)
    'PointRobot':                             dict(safety_weight=100000.0),
    'BoatRobot':                              dict(safety_weight=100000.0),
}

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 31, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_float('safety_weight', -1.0, 'Override safety_weight (-1 = use config default)')
flags.DEFINE_float('alpha', -1.0, 'Override alpha (-1 = use config default)')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
    elif details['env_name'] == 'BoatRobot':
        assert details['dataset_kwargs']['br_data'] is not None, "No data for Boat Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['br_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )


    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        
        if i % details['log_interval'] == 0:
            print({f"train/{k}": v for k, v in info.items()})

        # if i % details['eval_interval'] == 0 and i > 0:
        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            elif details['env_name'] == 'BoatRobot':
                eval_info = evaluate_br(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            print({f"eval/{k}": v for k, v in eval_info.items()})


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']

    if FLAGS.experiment_name == '':
        ak = parameters['agent_kwargs']
        if 'sampling_method' in ak:
            # SafeFQL_Base-style name
            parameters['experiment_name'] = ak['sampling_method'] + '_' \
                                    + ak['actor_objective'] + '_' \
                                    + ak['critic_type'] + '_N' \
                                    + str(ak['N']) + '_' \
                                    + ak['extract_method']
        else:
            # SafeFQL-style name
            parameters['experiment_name'] = ak['model_cls'] + '_' \
                                    + ak['actor_objective'] + '_' \
                                    + ak['critic_type'] + '_N' \
                                    + str(ak['N']) + '_' \
                                    + ak['extract_method']
    else:
        parameters['experiment_name'] = FLAGS.experiment_name
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_s' + str(parameters['seed']) + '_' + str(random.randint(0,1000))

    if parameters['env_name'] == 'PointRobot' or parameters['env_name'] == 'BoatRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    # Optional override of max_steps via env var, e.g. for longer DSRL trainings
    _max_steps_override = os.environ.get('SAFEFQL_MAX_STEPS')
    if _max_steps_override:
        parameters['max_steps'] = int(_max_steps_override)
        print(f"[override] max_steps -> {parameters['max_steps']}")

    # Apply per-environment SafeFQL safety hyperparameters (safety_weight, safety_delta).
    # Only relevant for the SafeFQL agent (its config has model_cls == 'SafeFQL').
    if parameters['agent_kwargs'].get('model_cls') == 'SafeFQL':
        env_overrides = SAFEFQL_PER_ENV.get(parameters['env_name'])
        if env_overrides is not None:
            for k, v in env_overrides.items():
                parameters['agent_kwargs'][k] = v
            print(f"[safefql] per-env override for {parameters['env_name']}: {env_overrides}")
        else:
            print(f"[safefql] no per-env override for {parameters['env_name']}; using defaults from agent.create()")

    # Apply CLI safety_weight override if provided
    if FLAGS.safety_weight >= 0:
        parameters['agent_kwargs']['safety_weight'] = FLAGS.safety_weight
        print(f"[override] safety_weight -> {FLAGS.safety_weight}")

    # Apply CLI alpha override if provided
    if FLAGS.alpha >= 0:
        parameters['agent_kwargs']['alpha'] = FLAGS.alpha
        print(f"[override] alpha -> {FLAGS.alpha}")

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
