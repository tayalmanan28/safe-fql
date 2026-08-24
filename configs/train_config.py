from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='SafeFQL',
        seed=-1,
        max_steps=100001,
        eval_episodes=1,
        batch_size=2048, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=25000,
        eval_interval=25000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
        br_data='data/boat-1M_mix.hdf5', # The location of boat_robot data
    )

    possible_structures = {
        "safefql_base": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="SafeFQL_Base",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="feasibility", 
                    sampling_method="ddpm", 
                    extract_method="minqc", 
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safefql": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="SafeFQL",
                    cost_limit=10,
                    # Actor (FQL flow-based)
                    actor_lr=3e-4,
                    actor_hidden_dims=(512, 512, 512, 512),
                    actor_layer_norm=False,
                    alpha=0.0, #10.0,           # distillation coefficient
                    flow_steps=10,        # Euler steps for BC flow
                    normalize_q_loss=True, #True,False
                    # Critics (SafeFQL_Base-style)
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    critic_hyperparam=0.9,
                    cost_critic_hyperparam=0.9,
                    critic_objective='expectile',
                    critic_type="hj",     # [hj, qc]
                    value_layer_norm=False,
                    # Cost / safety
                    cost_temperature=5,
                    reward_temperature=3,
                    cost_ub=150,
                    actor_objective="feasibility",
                    # Evaluation
                    N=1,
                    extract_method="safe_maxq",  # [minqc, maxq, safe_maxq]
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safeifql": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="SafeIFQL",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    actor_hidden_dims=(256, 256, 256),
                    actor_layer_norm=False,
                    actor_weight_decay=None,
                    actor_tau=0.001,
                    decay_steps=int(3e6),
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,               # Euler ODE integration steps
                    N=16,              # rejection sampling candidates
                    clip_sampler=True,
                    critic_objective='expectile',
                    critic_hyperparam=0.9,
                    cost_critic_hyperparam=0.9,
                    critic_type="hj",  # [hj, qc]
                    cost_ub=150,
                    actor_objective="feasibility",
                    extract_method="safe_maxq",  # [minqc, maxq, safe_maxq]
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "normalized_safefql": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="NormalizedSafeFQL",
                    cost_limit=10,
                    # Actor (FQL flow-based)
                    actor_lr=3e-4,
                    actor_hidden_dims=(512, 512, 512, 512),
                    actor_layer_norm=False,
                    alpha=0.0,
                    flow_steps=10,
                    normalize_q_loss=False,
                    # Critics
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    critic_hyperparam=0.9,
                    cost_critic_hyperparam=0.9,
                    critic_objective='expectile',
                    critic_type="hj",
                    value_layer_norm=False,
                    # Cost / safety
                    cost_temperature=2,
                    reward_temperature=5,
                    cost_ub=150,
                    actor_objective="feasibility",
                    # Safety weight (per-env tuning required)
                    safety_weight=0.1,
                    # Evaluation
                    N=1,
                    extract_method="safe_maxq",
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]