"""SafeIFQL: SafeFQL_Base with Flow Matching instead of DDPM.

This agent combines:
  - SafeFQL_Base's critic learning: V(s), Q(s,a) for reward; V_c(s), Q_c(s,a) for cost
  - SafeFQL_Base's AWR feasibility-weighted actor training
  - Flow Matching (Conditional Flow Matching / Rectified Flow) policy instead of DDPM

The policy is a velocity field v_θ(x_t, t, s) that maps noise to actions via
Euler ODE integration. Training uses the same advantage-weighted regression
(feasibility weighting) as SafeFQL_Base, but with the flow matching loss:
    L = w(s,a) * ||v_θ(x_t, t, s) - (a - ε)||²
where x_t = (1-t)*ε + t*a is the linear interpolation, and (a - ε) is the
target velocity.

At evaluation, N samples are drawn by integrating the learned velocity field
from noise to actions in `T` Euler steps, then the best action is selected
using the safety/reward critics.
"""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    StateValue,
    get_weight_decay_mask,
)


# =====================================================================
# Helper losses (identical to SafeFQL_Base)
# =====================================================================

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


# =====================================================================
# Jitted Q / Q_c evaluation helpers
# =====================================================================

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))


# =====================================================================
# Flow Vector Field network
# =====================================================================

class FlowVectorField(nn.Module):
    """Vector field v_θ(observations, actions, times) for flow matching.

    Takes (observations, actions, times) and outputs a velocity vector of
    the same dimensionality as `action_dim`.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, actions, times=None, training=False):
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        x = MLP(
            hidden_dims=tuple(list(self.hidden_dims) + [self.action_dim]),
            activations=nn.gelu,
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
        )(inputs, training=training)
        return x


# =====================================================================
# Jitted Euler ODE sampler for flow matching
# =====================================================================

@partial(jax.jit, static_argnames=("flow_fn", "T"))
def _flow_sample(flow_fn, flow_params, observations, noise, T):
    """Euler ODE integration from noise to actions in T steps (fully jitted)."""
    dt = 1.0 / T

    def euler_step(step, actions):
        t = step * dt
        t_batch = jnp.full((actions.shape[0], 1), t)
        velocity = flow_fn({'params': flow_params}, observations, actions, t_batch)
        return actions + velocity * dt

    actions = jax.lax.fori_loop(0, T, euler_step, noise)
    return jnp.clip(actions, -1.0, 1.0)


# =====================================================================
# SafeIFQL Agent
# =====================================================================

class SafeIFQL(Agent):
    """SafeFQL_Base critics + Flow Matching policy (replaces DDPM in SafeFQL_Base)."""

    flow_model: TrainState
    target_flow_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)     # Euler ODE steps for sampling
    N: int                                        # How many samples per observation
    clip_sampler: bool = struct.field(pytree_node=False)
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        N: int = 64,
        clip_sampler: bool = True,
        actor_objective: str = 'feasibility',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        decay_steps: Optional[int] = int(2e6),
        extract_method: str = 'minqc',
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.,
        cost_scale: float = 25.0,  # unused, for config compat
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = (
            jax.random.split(rng, 6)
        )
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount ** env_max_steps) / (
            1 - discount
        ) / env_max_steps

        # ---- Flow model (replaces DDPM) ----
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        flow_def = FlowVectorField(
            hidden_dims=tuple(actor_hidden_dims),
            action_dim=action_dim,
            use_layer_norm=actor_layer_norm,
        )

        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        times = jnp.zeros((1, 1))
        flow_params = flow_def.init(actor_key, observations, actions, times)["params"]

        flow_model = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.adamw(
                learning_rate=actor_lr,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )
        target_flow_model = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- Critics (identical to SafeFQL_Base) ----
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- Safe critics ----
        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)[
            "params"
        ]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- Value networks ----
        value_base_cls = partial(
            MLP,
            hidden_dims=critic_hidden_dims,
            activate_final=True,
            use_layer_norm=value_layer_norm,
        )
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(
            apply_fn=value_def.apply, params=value_params, tx=value_optimiser
        )

        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value = TrainState.create(
            apply_fn=value_def.apply, params=safe_value_params, tx=value_optimiser
        )

        return cls(
            actor=None,
            flow_model=flow_model,
            target_flow_model=target_flow_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            act_dim=action_dim,
            T=T,
            N=N,
            clip_sampler=clip_sampler,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            critic_objective=critic_objective,
            critic_type=critic_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            extract_method=extract_method,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
        )

    # =================================================================
    # Critic updates (identical to SafeFQL_Base)
    # =================================================================

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            if agent.critic_objective == 'expectile':
                value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f'Invalid critic objective: {agent.critic_objective}')
            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)
        return agent, info

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn(
                {"params": safe_value_params}, batch["observations"]
            )
            safe_value_loss = safe_expectile_loss(
                qc - vc, agent.cost_critic_hyperparam
            ).mean()
            return safe_value_loss, {
                "safe_value_loss": safe_value_loss,
                "vc": vc.mean(),
                "vc_max": vc.max(),
                "vc_min": vc.min(),
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        agent = agent.replace(safe_value=safe_value)
        return agent, info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        if agent.critic_type == "hj":
            qc_nonterminal = (1.0 - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(
                batch["costs"], next_vc
            )
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == "qc":
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f"Invalid critic type: {agent.critic_type}")

        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"],
                batch["actions"],
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()
            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean(),
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(
            params=safe_target_critic_params
        )
        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    # =================================================================
    # Actor update — Flow Matching with AWR feasibility weighting
    # =================================================================

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        batch_size = batch['actions'].shape[0]

        # ---- Sample random time t ~ U(0, 1) and noise ε ~ N(0, I) ----
        time = jax.random.uniform(key, (batch_size,))
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(key, (batch_size, agent.act_dim))

        # Linear interpolation: x_t = (1 - t) * ε + t * a
        t_expand = time[:, None]  # (B, 1)
        x_t = (1.0 - t_expand) * noise + t_expand * batch['actions']
        # Target velocity: u_t = a - ε
        target_velocity = batch['actions'] - noise

        time_input = jnp.expand_dims(time, axis=1)  # (B, 1)

        key, rng = jax.random.split(rng, 2)

        # ---- Compute feasibility weights (identical to SafeFQL_Base) ----
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )

        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["observations"]
        )

        if agent.critic_type == "qc":
            qc = qc - agent.qc_thres
            vc = vc - agent.qc_thres

        if agent.actor_objective == "feasibility":
            eps = 0.
            unsafe_condition = jnp.where(vc > 0. - eps, 1, 0)
            safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc <= 0. - eps, 1, 0)

            cost_exp_adv = jnp.exp((vc - qc) * agent.cost_temperature)
            reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)

            unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub)
            safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)

            weights = unsafe_weights + safe_weights
        elif agent.actor_objective == "bc":
            weights = jnp.ones(qc.shape)
        else:
            raise ValueError(f'Invalid actor objective: {agent.actor_objective}')

        # ---- Weighted flow matching loss ----
        def actor_loss_fn(flow_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v_pred = agent.flow_model.apply_fn(
                {'params': flow_params},
                batch['observations'],
                x_t,
                time_input,
                training=True,
            )
            # Weighted MSE: w(s,a) * ||v_θ(x_t, t, s) - (a - ε)||²
            actor_loss = (
                ((v_pred - target_velocity) ** 2).sum(axis=-1) * weights
            ).mean()
            return actor_loss, {'actor_loss': actor_loss, 'weights': weights.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.flow_model.params)
        flow_model = agent.flow_model.apply_gradients(grads=grads)

        agent = agent.replace(flow_model=flow_model)

        target_flow_params = optax.incremental_update(
            flow_model.params, agent.target_flow_model.params, agent.actor_tau
        )
        target_flow_model = agent.target_flow_model.replace(params=target_flow_params)

        new_agent = agent.replace(
            flow_model=flow_model, target_flow_model=target_flow_model, rng=rng
        )
        return new_agent, info

    # =================================================================
    # Evaluation — jitted Euler ODE sampling + critic-based selection
    # =================================================================

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)

        flow_params = self.target_flow_model.params

        # Sample initial noise
        rng, key = jax.random.split(rng, 2)
        noise = jax.random.normal(key, (self.N, self.act_dim))

        # Jitted Euler ODE integration + clipping in one call
        actions = _flow_sample(
            self.flow_model.apply_fn, flow_params,
            observations, noise, self.T,
        )

        if self.N == 1:
            # N=1 fast path: skip critic evaluation entirely
            return np.array(actions.squeeze(0)), self.replace(rng=rng)

        # N > 1: select best action using critics
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(
            self.target_critic.apply_fn, self.target_critic.params,
            observations, actions,
        )
        qcs = compute_safe_q(
            self.safe_target_critic.apply_fn, self.safe_target_critic.params,
            observations, actions,
        )

        if self.critic_type == "qc":
            qcs = qcs - self.qc_thres

        if self.extract_method == 'maxq':
            idx = jnp.argmax(qs)
        elif self.extract_method == 'minqc':
            idx = jnp.argmin(qcs)
        elif self.extract_method == 'safe_maxq':
            safe_mask = qcs < 0
            safe_qs = jnp.where(safe_mask, qs, -jnp.inf)
            idx = jnp.where(jnp.any(safe_mask), jnp.argmax(safe_qs), jnp.argmin(qcs))
        else:
            raise ValueError(f'Invalid extract_method: {self.extract_method}')

        action = actions[idx]
        return np.array(action.squeeze()), self.replace(rng=rng)

    # =================================================================
    # Main update loop
    # =================================================================

    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch['observations'].shape[0] / 2)

        def first_half(x):
            return x[:batch_size]

        def second_half(x):
            return x[batch_size:]

        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)

        def slice(x):
            return x[:256]

        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)
        new_agent, safe_critic_info = new_agent.update_vc(mini_batch)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch)

        return new_agent, {
            **actor_info,
            **critic_info,
            **value_info,
            **safe_critic_info,
            **safe_value_info,
        }

    # =================================================================
    # Save / load
    # =================================================================

    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
