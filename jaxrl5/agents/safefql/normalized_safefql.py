"""NormalizedSafeFQL: SafeFQL with per-branch normalized gated actor loss.

Identical to SafeFQL except the one-step actor loss normalizes the reward
and safety branches independently by the number of samples in each branch:

    q_loss = mean_safe(-q) + safety_weight * mean_unsafe(max(qc, 0))

where mean_safe and mean_unsafe average only over their respective samples.
This eliminates the dependency of the effective gradient ratio on
frac_actor_safe, making a single safety_weight work across environments
without per-env tuning.
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
from flax.training.train_state import TrainState
from flax import struct

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    StateValue,
    get_weight_decay_mask,
)

# Import shared helpers from safefql module
from jaxrl5.agents.safefql.safefql import (
    expectile_loss,
    safe_expectile_loss,
    compute_q,
    compute_safe_q,
    FlowVectorField,
)


class NormalizedSafeFQL(Agent):
    """SafeFQL with per-branch normalized gated actor loss."""

    # ---- Flow-based actor ----
    actor_bc_flow: TrainState
    actor_onestep_flow: TrainState

    # ---- Reward critics ----
    critic: TrainState
    target_critic: TrainState
    value: TrainState

    # ---- Cost / safety critics ----
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState

    # ---- Continuous hyper-parameters ----
    discount: float
    tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    alpha: float
    safety_weight: float  # single universal weight (no per-env tuning needed)
    safety_delta: float   # safety margin; qc < delta treated as safe (default 0)

    # ---- Static (non-pytree) hyper-parameters ----
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    flow_steps: int = struct.field(pytree_node=False)
    normalize_q_loss: bool = struct.field(pytree_node=False)

    # ------------------------------------------------------------------
    # Construction (identical to SafeFQL)
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        # --- actor (FQL) ---
        actor_lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
        actor_layer_norm: bool = False,
        alpha: float = 0.0,
        flow_steps: int = 10,
        normalize_q_loss: bool = False,
        # --- critics ---
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        value_layer_norm: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.9,
        cost_critic_hyperparam: float = 0.9,
        critic_objective: str = "expectile",
        critic_type: str = "hj",
        # --- cost / safety ---
        cost_temperature: float = 5.0,
        reward_temperature: float = 3.0,
        cost_limit: float = 10.0,
        env_max_steps: int = 1000,
        cost_ub: float = 200.0,
        actor_objective: str = "feasibility",
        # --- one-step actor safety penalty ---
        safety_weight: float = 1.0,  # universal default (no per-env tuning)
        safety_delta: float = 0.0,   # safety margin for the gate
        # --- evaluation ---
        N: int = 64,
        extract_method: str = "minqc",
        # --- unused (for config compat) ---
        cost_scale: float = 25.0,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, bc_key, os_key, c_key, v_key, sc_key, sv_key = jax.random.split(rng, 7)

        observations = observation_space.sample()
        actions = action_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount ** env_max_steps) / (
            1 - discount
        ) / env_max_steps

        obs_b = jnp.expand_dims(jnp.array(observations), axis=0)
        act_b = jnp.expand_dims(jnp.array(actions), axis=0)
        time_b = jnp.zeros((1, 1))

        # ---- Flow actors ----
        bc_flow_def = FlowVectorField(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            use_layer_norm=actor_layer_norm,
        )
        bc_flow_params = bc_flow_def.init(bc_key, obs_b, act_b, time_b)["params"]
        actor_bc_flow = TrainState.create(
            apply_fn=bc_flow_def.apply,
            params=bc_flow_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        os_flow_def = FlowVectorField(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            use_layer_norm=actor_layer_norm,
        )
        os_flow_params = os_flow_def.init(os_key, obs_b, act_b)["params"]
        actor_onestep_flow = TrainState.create(
            apply_fn=os_flow_def.apply,
            params=os_flow_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        # ---- Reward critics ----
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(c_key, obs_b, act_b)["params"]
        critic_opt = optax.adam(learning_rate=critic_lr)

        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_opt
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ---- Cost critics ----
        if critic_type == "qc":
            safe_critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            safe_critic_def = Ensemble(safe_critic_cls, num=num_qs)
        else:
            safe_critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = safe_critic_def.init(sc_key, obs_b, act_b)["params"]
        safe_critic = TrainState.create(
            apply_fn=safe_critic_def.apply,
            params=safe_critic_params,
            tx=critic_opt,
        )
        safe_target_critic = TrainState.create(
            apply_fn=safe_critic_def.apply,
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
        value_params = value_def.init(v_key, obs_b)["params"]
        value_opt = optax.adam(learning_rate=value_lr)

        value = TrainState.create(
            apply_fn=value_def.apply, params=value_params, tx=value_opt
        )

        if critic_type == "qc":
            safe_value_def = StateValue(base_cls=value_base_cls)
        else:
            safe_value_def = value_def

        safe_value_params = safe_value_def.init(sv_key, obs_b)["params"]
        safe_value = TrainState.create(
            apply_fn=safe_value_def.apply,
            params=safe_value_params,
            tx=value_opt,
        )

        return cls(
            actor=None,
            rng=rng,
            actor_bc_flow=actor_bc_flow,
            actor_onestep_flow=actor_onestep_flow,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            discount=discount,
            tau=tau,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            alpha=alpha,
            safety_weight=safety_weight,
            safety_delta=safety_delta,
            critic_objective=critic_objective,
            critic_type=critic_type,
            actor_objective=actor_objective,
            extract_method=extract_method,
            act_dim=action_dim,
            N=N,
            flow_steps=flow_steps,
            normalize_q_loss=normalize_q_loss,
        )

    # ==================================================================
    # Critic updates (identical to SafeFQL — delegated via copy)
    # ==================================================================

    def update_v(agent, batch: DatasetDict) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"], batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params):
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            return loss, {"value_loss": loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        return agent.replace(value=value), info

    def update_q(agent, batch: DatasetDict) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params):
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["observations"], batch["actions"],
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        return agent.replace(critic=critic, target_critic=target_critic), info

    def update_vc(agent, batch: DatasetDict) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params):
            vc = agent.safe_value.apply_fn(
                {"params": safe_value_params}, batch["observations"]
            )
            loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()
            return loss, {
                "safe_value_loss": loss, "vc": vc.mean(),
                "vc_max": vc.max(), "vc_min": vc.min(),
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )

        if agent.critic_type == "hj":
            qc_nonterminal = (1.0 - agent.discount) * batch["costs"] + \
                agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == "qc":
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f"Invalid critic type: {agent.critic_type}")

        def safe_critic_loss_fn(safe_critic_params):
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"],
            )
            loss = ((qcs - target_qc) ** 2).mean()
            return loss, {
                "safe_critic_loss": loss, "qc": qcs.mean(),
                "qc_max": qcs.max(), "qc_min": qcs.min(),
                "costs": batch["costs"].mean(),
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        return agent.replace(
            safe_critic=safe_critic, safe_target_critic=safe_target_critic
        ), info

    # ==================================================================
    # FQL-style flow actor updates
    # ==================================================================

    def _compute_feasibility_weights(agent, batch: DatasetDict):
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"], batch["actions"],
        )
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"],
        )
        qc = qcs.max(axis=0)
        vc = agent.safe_value.apply_fn({"params": agent.safe_value.params}, batch["observations"])

        qc_shifted = qc - agent.qc_thres if agent.critic_type == "qc" else qc
        vc_shifted = vc - agent.qc_thres if agent.critic_type == "qc" else vc

        if agent.actor_objective == "feasibility":
            cost_exp_adv = jnp.exp((vc_shifted - qc_shifted) * agent.cost_temperature)
            weights = jnp.clip(cost_exp_adv, 0, agent.cost_ub)
        elif agent.actor_objective == "bc":
            weights = jnp.ones(q.shape)
        else:
            raise ValueError(f"Invalid actor_objective: {agent.actor_objective}")
        return weights, q, v, qc, vc

    def _compute_flow_actions(agent, observations, noises):
        actions = noises
        for i in range(agent.flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / agent.flow_steps)
            vels = agent.actor_bc_flow.apply_fn(
                {"params": agent.actor_bc_flow.params},
                observations, actions, t,
            )
            actions = actions + vels / agent.flow_steps
        return jnp.clip(actions, -1, 1)

    def update_bc_flow(
        agent, batch: DatasetDict
    ) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        rng = agent.rng
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        batch_size = batch["actions"].shape[0]
        action_dim = agent.act_dim

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        def bc_flow_loss_fn(bc_flow_params):
            pred = agent.actor_bc_flow.apply_fn(
                {"params": bc_flow_params},
                batch["observations"], x_t, t,
            )
            per_sample = ((pred - vel) ** 2).sum(axis=-1)
            loss = per_sample.mean()
            return loss, {"bc_flow_loss": loss}

        grads, info = jax.grad(bc_flow_loss_fn, has_aux=True)(agent.actor_bc_flow.params)
        actor_bc_flow = agent.actor_bc_flow.apply_gradients(grads=grads)
        return agent.replace(actor_bc_flow=actor_bc_flow, rng=rng), info

    # ---- One-step flow update (THE KEY DIFFERENCE) ----

    def update_onestep_flow(
        agent, batch: DatasetDict
    ) -> Tuple["NormalizedSafeFQL", Dict[str, float]]:
        """Update one-step flow via distillation + NORMALIZED gated Q-loss.

        The gated loss normalizes each branch by its own sample count:
            q_loss = mean_over_safe_samples(-q)
                   + safety_weight * mean_over_unsafe_samples(max(qc, 0))

        This ensures neither branch dominates regardless of how many samples
        fall on each side of the safety boundary (qc < 0 vs qc >= 0).
        """
        rng = agent.rng
        rng, noise_rng = jax.random.split(rng)

        batch_size = batch["actions"].shape[0]
        noises = jax.random.normal(noise_rng, (batch_size, agent.act_dim))

        target_flow_actions = agent._compute_flow_actions(
            batch["observations"], noises
        )

        def onestep_loss_fn(onestep_params):
            # --- Distillation loss ---
            actor_actions = agent.actor_onestep_flow.apply_fn(
                {"params": onestep_params},
                batch["observations"],
                noises,
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # --- Reward Q-guided loss (normalized gated) ---
            actor_actions_clip = jnp.clip(actor_actions, -1, 1)

            qs = agent.critic.apply_fn(
                {"params": agent.critic.params},
                batch["observations"],
                actor_actions_clip,
            )
            q = qs.min(axis=0)

            qcs = agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params},
                batch["observations"],
                actor_actions_clip,
            )
            qc = qcs.max(axis=0)

            # Gate: safe (qc < delta) vs unsafe (qc >= delta)
            # delta > 0 creates a safety margin around the boundary
            delta = agent.safety_delta
            actor_is_safe = (qc < delta).astype(jnp.float32)  # (B,)
            n_safe = actor_is_safe.sum() + 1e-8
            n_unsafe = (1 - actor_is_safe).sum() + 1e-8

            # Reward branch: average over SAFE samples only
            safe_loss = (actor_is_safe * (-q)).sum() / n_safe

            # Safety branch: average over UNSAFE samples only
            unsafe_loss = ((1 - actor_is_safe) * jnp.maximum(qc - delta, 0)).sum() / n_unsafe

            q_loss = safe_loss + agent.safety_weight * unsafe_loss

            if agent.normalize_q_loss:
                lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
                q_loss = lam * q_loss

            total = agent.alpha * distill_loss + q_loss

            return total, {
                "onestep_total_loss": total,
                "distill_loss": distill_loss,
                "q_loss": q_loss,
                "safe_loss": safe_loss,
                "unsafe_loss": unsafe_loss,
                "actor_q": q.mean(),
                "actor_qc": qc.mean(),
                "frac_actor_safe": actor_is_safe.mean(),
            }

        grads, info = jax.grad(onestep_loss_fn, has_aux=True)(
            agent.actor_onestep_flow.params
        )
        actor_onestep_flow = agent.actor_onestep_flow.apply_gradients(grads=grads)
        return agent.replace(actor_onestep_flow=actor_onestep_flow, rng=rng), info

    # ==================================================================
    # Combined update
    # ==================================================================

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch["observations"].shape[0] / 2)

        def first_half(x):
            return x[:batch_size]
        def second_half(x):
            return x[batch_size:]

        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_bc_flow(first_batch)
        new_agent, bc_flow_info = new_agent.update_bc_flow(second_batch)
        new_agent, _ = new_agent.update_onestep_flow(first_batch)
        new_agent, onestep_info = new_agent.update_onestep_flow(second_batch)

        def slice_mini(x):
            return x[:256]

        mini_batch = jax.tree_util.tree_map(slice_mini, batch)
        new_agent, v_info = new_agent.update_v(mini_batch)
        new_agent, q_info = new_agent.update_q(mini_batch)
        new_agent, vc_info = new_agent.update_vc(mini_batch)
        new_agent, qc_info = new_agent.update_qc(mini_batch)

        return new_agent, {
            **bc_flow_info,
            **onestep_info,
            **v_info,
            **q_info,
            **vc_info,
            **qc_info,
        }

    # ==================================================================
    # Evaluation (identical to SafeFQL — one-shot)
    # ==================================================================

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1

        observations = jax.device_put(observations)
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, (self.act_dim,))

        action = self.actor_onestep_flow.apply_fn(
            {"params": self.actor_onestep_flow.params},
            observations,
            noise,
        )
        action = jnp.clip(action, -1, 1)
        return np.array(action), self.replace(rng=rng)

    # ==================================================================
    # Serialization
    # ==================================================================

    def save(self, modeldir, save_time):
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        file_name = f"model{save_time}.pickle"
        state_dict = flax.serialization.to_state_dict(self)
        with open(os.path.join(modeldir, file_name), "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, model_location):
        with open(model_location, "rb") as f:
            pkl_file = pickle.load(f)
        return flax.serialization.from_state_dict(target=self, state=pkl_file)
