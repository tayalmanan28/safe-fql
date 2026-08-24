"""Implementations of value function learning for safe RL.

This module implements the Qc/Vc value function learner.
The learned Vc is used as a Control Barrier Function (CBF) at evaluation time,
combined with a Behavior Cloning (BC) reference controller and a learned
control-affine dynamics model via CBF-QP safety filtering.
"""
import os
from functools import partial
from typing import Dict, Sequence, Tuple
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class VOCBF(Agent):
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    cost_critic_hyperparam: float
    critic_type: str = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        cost_critic_hyperparam: float = 0.9,
        num_qs: int = 2,
        value_layer_norm: bool = False,
        critic_type: str = 'hj',
        **kwargs,  # Accept and ignore unused kwargs for backward compat
    ):

        rng = jax.random.PRNGKey(seed)
        rng, safe_critic_key, safe_value_key = jax.random.split(rng, 3)
        actions = action_space.sample()
        observations = observation_space.sample()

        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)

        # --- Cost (Safety) Critic & Value ---
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_optimiser = optax.adam(learning_rate=critic_lr)

        safe_critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        safe_critic_def = Ensemble(safe_critic_cls, num=num_qs)

        safe_critic_params = safe_critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=safe_critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_optimiser = optax.adam(learning_rate=value_lr)

        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = safe_value_def.init(safe_value_key, observations)["params"]
        safe_value = TrainState.create(apply_fn=safe_value_def.apply,
                                       params=safe_value_params,
                                       tx=value_optimiser)

        return cls(
            actor=None,  # Base class attribute (unused)
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            cost_critic_hyperparam=cost_critic_hyperparam,
            critic_type=critic_type,
        )

    # ----------------------------------------------------------------
    # Cost (safety) value function updates
    # ----------------------------------------------------------------

    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch["observations"],
            batch["actions"],
        )
        # min over ensemble = pessimistic about safety (h > 0 = safe)
        qc = qcs.min(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])

            # Upper expectile: Vc(s) ≈ max_a Qc(s,a) — same as reward V
            safe_value_loss = expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()

            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        agent = agent.replace(safe_value=safe_value)
        return agent, info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.minimum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    # ----------------------------------------------------------------
    # Combined update step
    # ----------------------------------------------------------------

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self

        def slice(x):
            return x[:256]

        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, vc_info = new_agent.update_vc(mini_batch)
        new_agent, qc_info = new_agent.update_qc(mini_batch)

        return new_agent, {**vc_info, **qc_info}

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent


# ================================================================
# CBF utils for evaluation (PyTorch-based)
# ================================================================

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def torchify(x):
    """
    Convert environment observation / numpy arrays into a torch.Tensor on DEFAULT_DEVICE.
    Robust to Gymnasium-style returns (obs, info) or dict-like observations.
    """
    if isinstance(x, (tuple, list)):
        x = x[0]

    if isinstance(x, dict):
        if 'observation' in x:
            x = x['observation']
        else:
            x = next(iter(x.values()))

    x = np.asarray(x)
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    t = torch.from_numpy(x)
    t = t.to(device=DEFAULT_DEVICE)
    return t


def get_V(V_net, state, device=DEFAULT_DEVICE):
    """Get CBF value B(x)"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        return V_net(state_tensor).item()


def get_V_gradient(V_net, state, device=DEFAULT_DEVICE):
    """Compute gradient of B with respect to state"""
    state_tensor = torch.FloatTensor(state).to(device).requires_grad_(True)
    V_val = V_net(state_tensor)

    V_val.backward()
    grad_V = state_tensor.grad.cpu().numpy()

    return grad_V


def cbf_safe_control(V_net, dynamics_model, state, nominal_control, alpha=1.0, return_debug=False):
    """Pure continuous-time CBF-QP safety filter using Lie derivatives.

    Uses the learned Vc as a Control Barrier Function with the standard
    continuous-time CBF condition:

        L_f V(x) + L_g V(x) · u + alpha · V(x) >= 0

    where:
        L_f V = ∇V · f(x)     (Lie derivative along drift)
        L_g V = ∇V · g(x)     (Lie derivative along control, vector)

    When the condition is violated with the nominal control, the QP correction
    finds the minimum-norm modification:
        min  ||u - u_nom||^2
        s.t. L_f V + L_g V · u + alpha · V >= 0
    """
    state_t = torchify(state).unsqueeze(0)  # (1, state_dim)

    # --- Current V(x) ---
    with torch.no_grad():
        V_current = V_net(state_t).item()

    # --- Gradient of V w.r.t. state ---
    grad_V = get_V_gradient(V_net, state)  # (state_dim,)

    # --- Learned dynamics f(x), g(x) ---
    with torch.no_grad():
        f_x, g_x = dynamics_model.get_f_g(state_t)
        f_np = f_x.cpu().numpy()[0]  # (state_dim,)
        g_np = g_x.cpu().numpy()[0]  # (state_dim, action_dim)

    # --- Lie derivatives ---
    L_f_V = np.dot(grad_V, f_np)           # scalar
    L_g_V = np.dot(grad_V, g_np)           # (action_dim,)

    # --- CBF condition: L_f V + L_g V · u + alpha · V >= 0 ---
    cbf_constraint = L_f_V + np.dot(L_g_V, nominal_control) + alpha * V_current

    debug = {
        'V_val': V_current,
        'L_f_V': float(L_f_V),
        'L_g_V_norm': float(np.linalg.norm(L_g_V)),
        'cbf_constraint': float(cbf_constraint),
        'intervened': False,
    }

    if cbf_constraint >= 0:
        if return_debug:
            return nominal_control, debug
        return nominal_control

    # --- QP correction (closed-form single-constraint projection) ---
    # Solve: min ||u - u_nom||^2 s.t. L_f V + L_g V · u + alpha · V >= 0
    # => L_g V · u >= -(L_f V + alpha · V)
    # => L_g V · (u_nom + delta) >= -(L_f V + alpha · V)
    # => L_g V · delta >= -cbf_constraint  (cbf_constraint < 0)
    violation = -cbf_constraint  # positive
    denominator = np.dot(L_g_V, L_g_V) + 1e-6
    correction = (violation / denominator) * L_g_V
    safe_control = nominal_control + correction
    safe_control = np.clip(safe_control, -1.0, 1.0)

    debug['intervened'] = True
    debug['correction_norm'] = float(np.linalg.norm(correction))
    debug['L_g_V_sq'] = float(np.dot(L_g_V, L_g_V))

    if return_debug:
        return safe_control, debug
    return safe_control


def evaluate_policy(env, episode_no, policy, V_net, dynamics_model, max_episode_steps, deterministic=True, cbf_alpha=1.0):
    """
    Run a single episode using a BC reference policy with CBF-QP safety filtering.

    Args:
        env: The environment.
        episode_no: Episode seed offset.
        policy: A BC policy with .act(state_tensor, deterministic) method.
        V_net: Learned Vc network used as the CBF barrier function.
        dynamics_model: Learned control-affine dynamics (AffineDynamics).
        max_episode_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic policy actions.
        cbf_alpha: CBF class-K alpha. Higher = more conservative.

    Returns:
        total_reward, total_cost, total_Vh (number of CBF violations).
    """
    reset_ret = env.reset(seed=episode_no + 500)

    # Handle both Gymnasium-style (obs, info) tuple and old-style plain array returns
    if isinstance(reset_ret, tuple):
        obs = reset_ret[0]
    else:
        obs = reset_ret

    # Keep resetting until we get a safe starting state (Vc >= 0)
    seed_offset = 0
    V_start = V_net(torchify(obs).unsqueeze(0)).item()
    while V_start < 0.:
        seed_offset += 1
        reset_ret = env.reset(seed=episode_no + 500 + seed_offset * 100)
        if isinstance(reset_ret, tuple):
            obs = reset_ret[0]
        else:
            obs = reset_ret
        V_start = V_net(torchify(obs).unsqueeze(0)).item()
        if seed_offset > 100:
            print(f"  Warning: Could not find safe start after {seed_offset} resets, proceeding with Vc={V_start:.4f}")
            break
    print(f"  Episode {episode_no}: Safe start Vc={V_start:.4f} (resets={seed_offset})")

    total_reward = 0.
    total_cost = 0.
    total_Vh = 0.
    cbf_interventions = 0
    min_V = float('inf')
    debug_episode = (episode_no == 0)
    cost_step_count = 0
    should_render = getattr(env.unwrapped, 'render_mode', None) == 'human'
    for step in range(max_episode_steps):
        if should_render:
            env.unwrapped.mujoco_renderer.render('human')

        with torch.no_grad():
            nominal_action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()

        action, dbg = cbf_safe_control(V_net, dynamics_model, obs, nominal_action,
                                       alpha=cbf_alpha, return_debug=True)

        if dbg['intervened']:
            cbf_interventions += 1

        step_ret = env.step(action)
        if len(step_ret) == 6:
            next_obs, reward, cost, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        elif len(step_ret) == 5:
            next_obs, reward, terminated, truncated, info = step_ret
            cost = info.get("cost", 0.0)
            done = bool(terminated or truncated)
        elif len(step_ret) == 4:
            next_obs, reward, done, info = step_ret
            cost = info.get("cost", info.get("violation", 0.0))
        else:
            raise RuntimeError(f"Unexpected env.step() return shape: {len(step_ret)}")

        V_val = V_net(torchify(next_obs).unsqueeze(0)).item()
        min_V = min(min_V, V_val)

        total_reward += float(reward)
        total_cost += float(cost)
        total_Vh += float(V_val < 0.)

        # Debug: print first 5 cost violations in episode 0 with full CBF info
        if debug_episode and float(cost) > 0 and cost_step_count < 5:
            cost_step_count += 1
            print(f"    [step {step}] COST! V={dbg['V_val']:.4f}, L_f_V={dbg['L_f_V']:.4f}, "
                  f"|L_g_V|={dbg['L_g_V_norm']:.4f}, constraint={dbg['cbf_constraint']:.4f}, "
                  f"intervened={dbg['intervened']}")

        if done:
            break
        obs = next_obs

    if debug_episode:
        print(f"    Episode 0 summary: steps={step+1}, cbf_interventions={cbf_interventions}, "
              f"min_Vc={min_V:.4f}, total_cost={total_cost:.1f}, Vc<0_steps={int(total_Vh)}")

    return total_reward, total_cost, total_Vh


def evaluate_cbf(env, num_episodes, policy, V_net, dynamics_model, max_episode_steps, deterministic=True, cbf_alpha=1.0):
    total_rewards, total_costs, total_Vhs = [], [], []
    for ep in range(num_episodes):
        reward, cost, vh = evaluate_policy(
            env, ep, policy, V_net, dynamics_model, max_episode_steps,
            deterministic, cbf_alpha=cbf_alpha
        )
        total_rewards.append(reward)
        total_costs.append(cost)
        total_Vhs.append(vh)

    return {
        "return": np.mean(total_rewards),
        "cost": np.mean(total_costs),
        "avg_Vc_violations": np.mean(total_Vhs),
        "episodes_with_violations": np.mean([v > 0 for v in total_Vhs]),
    }


def mlp_dyn(sizes, activation, output_activation=nn.Identity):
    """Creates a multi-layer perceptron with the specified sizes and activations."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)


class AffineDynamics(nn.Module):
    """
    Neural network for learning control-affine dynamics: x_dot = f(x) + g(x)u
    Returns f(x) and g(x) directly for easy integration with CBF-QP.
    """
    def __init__(
        self,
        num_action: int,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dt: float = 0.1
    ):
        super().__init__()

        self.num_action = num_action
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt

        # Drift dynamics f(x)
        self.f = mlp_dyn(
            [self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim],
            activation=nn.ReLU
        )

        # Control dynamics g(x) - outputs flattened state_dim * num_action
        self.g = mlp_dyn(
            [self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim * self.num_action],
            activation=nn.ReLU
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f(x) and g(x) where g(x) is flattened"""
        return self.f(state), self.g(state)

    def forward_x_dot(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Computes state derivative: x_dot = f(x) + g(x)u"""
        f, g = self.forward(state)
        g_matrix = g.view(-1, self.state_dim, self.num_action)
        gu = torch.einsum('bsa,ba->bs', g_matrix, action)
        x_dot = f + gu
        return x_dot

    def forward_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts next state using Euler integration: x_next = x + (f(x) + g(x)u) * dt"""
        x_dot = self.forward_x_dot(state, action)
        return state + x_dot * self.dt

    def get_f_g(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns f(x) and properly shaped g(x) matrix.
        g(x) shape: (batch_size, state_dim, num_action)
        """
        f, g_flat = self.forward(state)
        g_matrix = g_flat.view(-1, self.state_dim, self.num_action)
        return f, g_matrix


class BCPolicy(nn.Module):
    """
    Simple Behavior Cloning (BC) policy.
    Maps observations -> actions via an MLP.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.max_action = max_action

        dims = [state_dim] + [hidden_dim] * num_layers + [action_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.net.to(dtype=torch.float32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.forward(state)
        return action.squeeze(0)


def build_vc_network(agent, state_dim):
    """
    Build a PyTorch V_net that mirrors the JAX safe_value network.
    Extracts weights from the trained JAX Vc and transfers them to PyTorch.

    JAX StateValue params are nested like:
        {'MLP_0': {'Dense_0': {kernel, bias}, 'Dense_1': {kernel, bias}, ...},
         'OutputVDense': {kernel, bias}}
    """
    params = agent.safe_value.params

    def _extract_dense_params(param_dict):
        """Recursively extract (kernel, bias) pairs from JAX params in order."""
        layers = []

        # First, collect layers from the base MLP (nested under MLP_0 or similar)
        mlp_keys = sorted([k for k in param_dict.keys() if k.startswith('MLP')])
        for mlp_key in mlp_keys:
            sub = param_dict[mlp_key]
            dense_keys = sorted(
                [k for k in sub.keys() if k.startswith('Dense_')],
                key=lambda x: int(x.split('_')[1])
            )
            for key in dense_keys:
                kernel = np.array(sub[key]['kernel'])
                bias = np.array(sub[key]['bias'])
                layers.append((kernel, bias))

        # Then, collect the top-level output dense layer
        if 'OutputVDense' in param_dict:
            kernel = np.array(param_dict['OutputVDense']['kernel'])
            bias = np.array(param_dict['OutputVDense']['bias'])
            layers.append((kernel, bias))

        # Fallback: if no MLP sub-key found, try flat Dense_* keys
        if not layers:
            dense_keys = sorted(
                [k for k in param_dict.keys() if k.startswith('Dense_')],
                key=lambda x: int(x.split('_')[1])
            )
            for key in dense_keys:
                kernel = np.array(param_dict[key]['kernel'])
                bias = np.array(param_dict[key]['bias'])
                layers.append((kernel, bias))

        return layers

    jax_layers = _extract_dense_params(params)

    torch_layers = []
    for i, (kernel, bias) in enumerate(jax_layers):
        in_dim, out_dim = kernel.shape
        linear = nn.Linear(in_dim, out_dim)
        # JAX Dense stores kernel as (in, out), PyTorch as (out, in)
        linear.weight.data = torch.FloatTensor(kernel.T)
        linear.bias.data = torch.FloatTensor(bias)
        torch_layers.append(linear)
        # Add ReLU after all layers except the last
        if i < len(jax_layers) - 1:
            torch_layers.append(nn.ReLU())

    V_net = nn.Sequential(*torch_layers)
    V_net.eval()
    V_net.to(DEFAULT_DEVICE)
    return V_net
