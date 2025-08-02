# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params  # pylint: disable=g-importing-member
from brax.training.types import PRNGKey  # pylint: disable=g-importing-member
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64
  apg_optimizer_state: Optional[optax.OptState] = None


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
  """Validates arguments for Madrona-MJX."""
  if madrona_backend:
    if eval_env:
      raise ValueError("Madrona-MJX doesn't support multiple env instances")
    if num_eval_envs != num_envs:
      raise ValueError('Madrona-MJX requires a fixed batch size')
    if action_repeat != 1:
      raise ValueError(
          "Implement action_repeat using PipelineEnv's _n_frames to avoid"
          ' unnecessary rendering!'
      )


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng  # pylint: disable=unexpected-keyword-arg
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """

  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    # HYBRID PPO/APG: New APG parameters
    apg_update_frequency: int = 5,
    apg_horizon_length: int = 16,
    apg_learning_rate: float = 1e-4,
    apg_discount_factor: float = 0.99,
    apg_num_env: int = 512,
    apg_num_updates_per_batch: int = 10,
    apg_stop_env_step: Optional[int] = None, #TODO: check this part.
    apg_use_value_function: bool = True,
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    max_devices_per_host: maximum number of chips to use per host process
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    madrona_backend: whether to use Madrona backend for training
    augment_pixels: whether to add image augmentation to pixel inputs
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    wrap_env_fn: a custom function that wraps the environment for training. If
      not specified, the environment is wrapped with the default training
      wrapper.
    randomization_fn: a user-defined callback function that generates randomized
      environments
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    max_grad_norm: gradient clipping norm value. If None, no clipping is done
    normalize_advantage: whether to normalize advantage estimate
    network_factory: function that generates networks for policy and value
      functions
    seed: random seed
    apg_update_frequency: How often to run APG update. 0 to disable. PPO/APG
      apg_update_frequency=10 means run APG update every 10 PPO updates.
    apg_horizon_length: Horizon for APG rollouts.
    apg_learning_rate: Optional separate learning rate for APG.
    apg_discount_factor: Discount factor for APG returns.
    apg_num_env: Number of parallel environments for APG rollouts.
    apg_num_updates_per_batch: Number of times to reuse APG data for updates.
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    eval_env: an optional environment for eval only, defaults to `environment`
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    deterministic_eval: whether to run the eval with a deterministic policy
    log_training_metrics: whether to log training metrics and callback to
      progress_fn
    training_metrics_steps: the number of environment steps between logging
      training metrics
    progress_fn: a user-defined callback function for reporting/plotting metrics
    policy_params_fn: a user-defined callback function that can be used for
      saving custom policy checkpoints or creating policy rollouts and videos
    save_checkpoint_path: the path used to save checkpoints. If None, no
      checkpoints are saved.
    restore_checkpoint_path: the path used to restore previous model params
    restore_params: raw network parameters to restore the TrainingState from.
      These override `restore_checkpoint_path`. These paramaters can be obtained
      from the return values of ppo.train().
    restore_value_fn: whether to restore the value function from the checkpoint
      or use a random initialization
    run_evals: if True, use the evaluator num_eval times to collect distinct
      eval rollouts. If False, num_eval_envs and eval_env are ignored.
      progress_fn is then expected to use training_metrics.

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0
  _validate_madrona_args(
      madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
  )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every PPO training step.
  ppo_env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  # The number of environment steps executed for every APG training step.
  apg_env_step_per_training_step = (
      apg_horizon_length * apg_num_env * apg_num_updates_per_batch * action_repeat
  )
  # Average step count per training step, used to approximate the total number
  # of training steps.
  avg_apg_step_per_training_step = (
      apg_env_step_per_training_step / apg_update_frequency
      if apg_update_frequency > 0
      else 0
  )
  env_step_per_training_step = (
      ppo_env_step_per_training_step + avg_apg_step_per_training_step
  )

  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )
  if local_devices_to_use > 1:
    reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
  else:
    reset_fn = jax.jit(jax.vmap(env.reset))
  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  env_state = reset_fn(key_envs)
  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  # HYBRID PPO/APG: Use apg_learning_rate if provided, otherwise PPO's LR
  ppo_optimizer = optax.adam(learning_rate=learning_rate)
  if max_grad_norm is not None:
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
  )

  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, ppo_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=training_metrics_steps
      or env_step_per_training_step,
      progress_fn=progress_fn,
  )

  ## HYBRID PPO/APG: Define APG update logic if enabled
  apg_grad_fn = None
  apg_optimizer = None
  if apg_update_frequency > 0:
    apg_lr = apg_learning_rate if apg_learning_rate is not None else learning_rate

    # APG Learning Rate Decay
    # total_training_steps = num_timesteps // env_step_per_training_step 
    # total_apg_optimizer_steps = (
    #     total_training_steps // apg_update_frequency
    # ) * apg_num_updates_per_batch

    # learning_rate_schedule = optax.exponential_decay(
    #     init_value=apg_lr,
    #     transition_steps=max(1, total_apg_optimizer_steps),
    #     decay_rate=0.9,
    #     end_value=0.0,
    # )

    # apg_optimizer = optax.adam(learning_rate=learning_rate_schedule)
    apg_optimizer = optax.adam(learning_rate=apg_lr)
    if max_grad_norm is not None:
      apg_optimizer = optax.chain(
          optax.clip_by_global_norm(max_grad_norm),
          optax.adam(learning_rate=apg_lr),
      )
    # This policy is just for APG's internal rollouts
    def make_apg_policy(
        params: types.PolicyParams, deterministic: bool = True
    ) -> types.Policy:
      # Note: This is a simplified policy maker for APG. It doesn't include the value function.
      def policy(
          observations: types.Observation, key_sample: PRNGKey
      ) -> Tuple[types.Action, types.Extra]:
        logits = ppo_network.policy_network.apply(*params, observations)
        if deterministic:
          return ppo_network.parametric_action_distribution.mode(logits), {}
        return (
            ppo_network.parametric_action_distribution.sample(
                logits, key_sample
            ),
            {},
        )

      return policy

    def make_apg_value_fn(
        normalizer_params: running_statistics.RunningStatisticsState, value_params
    ) -> Callable:
      def value_fn(observations: types.Observation) -> jnp.ndarray:
        return ppo_network.value_network.apply(
            normalizer_params, value_params, observations
        )

      return value_fn

    def env_step_for_apg(
        carry: Tuple[envs.State, PRNGKey], _, policy: types.Policy
    ):
      env_state, key = carry
      key, key_sample = jax.random.split(key)
      actions, _ = policy(env_state.obs, key_sample)
      nstate = env.step(env_state, actions)
      return (nstate, key), nstate.reward

    def select_first_n_envs(env_state: envs.State, n: int) -> envs.State:
      """Select the first n environments' state"""
      return jax.tree_util.tree_map(
          lambda x: x[:n] if isinstance(x, jnp.ndarray) else x, env_state
      )

    def apg_loss_fn(
        policy_params,
        value_params,
        normalizer_params,
        current_env_state,
        key,
    ):
      """Calculates APG loss, including the differentiable rollout."""
      # --- Rollout generation is now INSIDE the loss function ---
      apg_policy = make_apg_policy((normalizer_params, policy_params))
      # add the sanity check for the apg_num_env
      # assert apg_num_env <= current_env_state.obs.shape[0], "apg_num_env is greater than the number of environments"
      current_env_state_apg = select_first_n_envs(
          current_env_state, apg_num_env
      )
      f = functools.partial(env_step_for_apg, policy=apg_policy)
      (nstate, _), rewards = jax.lax.scan(
          f, (current_env_state_apg, key), None, length=apg_horizon_length
      )
      # --- End of rollout generation ---

      terminal_obs = nstate.obs
      discount_factor = apg_discount_factor
      discount_factor_array = (
          discount_factor ** jnp.arange(apg_horizon_length)
      )[:, None]
      sum_discounted_rewards = jnp.sum(rewards * discount_factor_array, axis=0)
      loss = -jnp.mean(sum_discounted_rewards)

      # TODO: check this part.
      if apg_use_value_function:
        value_fn = make_apg_value_fn(normalizer_params, value_params)
        terminal_state_values = value_fn(terminal_obs)
        loss -= jnp.mean(terminal_state_values) * discount_factor**apg_horizon_length

      return loss

    apg_grad_fn = jax.grad(
        apg_loss_fn, argnums=0
    )  # Grad only w.r.t policy_params


  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    # old_policy_params = params.policy
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state, grads = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state,
    )
    ppo_policy_grads = grads.policy
    ppo_value_grads = grads.value

    return (optimizer_state, params, key), (metrics, ppo_policy_grads)

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(data.observation),
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(data.next_observation),
          extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), (metrics, ppo_policy_grads) = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )
    # ppo_policy_grads = _unpmap(ppo_policy_grads)
    ppo_policy_grads = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0), ppo_policy_grads
    )
    return (optimizer_state, params, key), (metrics, ppo_policy_grads)

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey, int], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey, int], Metrics]:
    training_state, state, key, step_count = carry
    key_apg, key_generate_unroll, key_sgd, new_key = jax.random.split(key, 4)

    # Step 1: APG Update (policy_t -> temp_new_policy)
    # ---------------------------------------------------------
    if apg_update_frequency > 0:

      def get_exploratory_policy_params(
          ts_ppo: TrainingState, key: PRNGKey
      ) -> ppo_losses.PPONetworkParams:
        """Performs APG update to get a temporary exploratory policy."""

        def apg_update_step(inner_carry, key_step):
          apg_optimizer_state, params_to_update = inner_carry
          grads = apg_grad_fn(
              params_to_update.policy,
              params_to_update.value,
              ts_ppo.normalizer_params,
              state,
              key_step,
          )
          grads = jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
          grads = jax.tree_util.tree_map(
              lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), grads
          )
          full_grads_tree = ts_ppo.params.replace(
              policy=grads,
              value=jax.tree_util.tree_map(jnp.zeros_like, ts_ppo.params.value),
          )
          params_update, apg_optimizer_state = apg_optimizer.update(
              full_grads_tree, apg_optimizer_state, params=ts_ppo.params
          )
          updated_params = optax.apply_updates(ts_ppo.params, params_update)
          return (apg_optimizer_state, updated_params), None

        initial_apg_opt_state = apg_optimizer.init(ts_ppo.params)
        keys = jax.random.split(key, apg_num_updates_per_batch)
        (_, temp_params), _ = jax.lax.scan(
            apg_update_step, (initial_apg_opt_state, ts_ppo.params), keys
        )
        return temp_params

      def do_apg_update(
          carry: Tuple[TrainingState, PRNGKey]
      ) -> Tuple[ppo_losses.PPONetworkParams, jnp.ndarray]:
        """Runs APG update and returns new params and imaginary step count."""
        ts, k = carry
        temp_params = get_exploratory_policy_params(ts, k)
        return temp_params, jnp.array(
            apg_env_step_per_training_step, dtype=jnp.uint32
        )

      def no_apg_update(
          carry: Tuple[TrainingState, PRNGKey]
      ) -> Tuple[ppo_losses.PPONetworkParams, jnp.ndarray]:
        """Returns original params and 0 imaginary steps."""
        ts, _ = carry
        return ts.params, jnp.array(0, dtype=jnp.uint32)

      temp_params, imaginary_steps = jax.lax.cond(
          step_count % apg_update_frequency == 0,
          do_apg_update,
          no_apg_update,
          (training_state, key_apg),
      )
    else:
      temp_params = training_state.params
      imaginary_steps = jnp.array(0, dtype=jnp.uint32)

    # Step 2: Collect Data with both policies and Merge
    # ---------------------------------------------------------
    ppo_policy_explore = make_policy(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    temp_policy_explore = make_policy(
        (training_state.normalizer_params, temp_params.policy, temp_params.value)
    )

    assert num_envs % 2 == 0, (
        'num_envs must be even for parallel rollout with two policies.'
    )
    num_envs_per_policy = num_envs // 2
    # num_envs_per_policy = num_envs
    state_ppo = jax.tree_util.tree_map(lambda x: x[:num_envs_per_policy], state)
    state_apg = jax.tree_util.tree_map(lambda x: x[num_envs_per_policy:], state)
  
    def f_multi_policy(carry, unused_t):
      (current_state_ppo, current_state_apg, current_key) = carry
      key_ppo, key_apg_rollout, next_key = jax.random.split(current_key, 3)

      next_state_ppo, data_ppo = acting.generate_unroll(
          env,
          current_state_ppo,
          ppo_policy_explore,
          key_ppo,
          unroll_length,
          extra_fields=('truncation', 'episode_metrics', 'episode_done'),
      )
      next_state_apg, data_apg = acting.generate_unroll(
          env,
          current_state_apg,
          temp_policy_explore,
          key_apg_rollout,
          unroll_length,
          extra_fields=('truncation', 'episode_metrics', 'episode_done'),
      )

      # Create source labels for each dataset.
      ppo_labels = jnp.ones_like(data_ppo.reward, dtype=jnp.float32)
      apg_labels = jnp.zeros_like(data_apg.reward, dtype=jnp.float32)
      merged_labels = jnp.concatenate([ppo_labels, apg_labels], axis=1)

      data_merged = jax.tree_util.tree_map(
          lambda x, y: jnp.concatenate([x, y], axis=1), data_ppo, data_apg
      )

      # Add the merged labels to the extras dictionary.
      data_merged.extras['is_from_ppo_policy'] = merged_labels

      return (next_state_ppo, next_state_apg, next_key), data_merged

    final_carry, merged_data_from_scan = jax.lax.scan(
        f_multi_policy,
        (state_ppo, state_apg, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    (final_state_ppo, final_state_apg, _) = final_carry
    state = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        final_state_ppo,
        final_state_apg,
    )

    if log_training_metrics:
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          merged_data_from_scan.extras['state_extras']['episode_metrics'],
          merged_data_from_scan.extras['state_extras']['episode_done'],
      )

    merged_data = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 1, 2), merged_data_from_scan
    )
    merged_data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), merged_data
    )

    # Step 3: Final PPO Update with merged data
    # ---------------------------------------------------------
    normalizer_params_final = running_statistics.update(
        training_state.normalizer_params,
        _remove_pixels(merged_data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    (final_optimizer_state, final_params, _,), (
        final_metrics,
        _,
    ) = jax.lax.scan(
        functools.partial(
            sgd_step, data=merged_data, normalizer_params=normalizer_params_final
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )
    final_metrics = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0), final_metrics
    )

    total_steps_this_turn = ppo_env_step_per_training_step + imaginary_steps
    final_training_state = TrainingState(
        optimizer_state=final_optimizer_state,
        params=final_params,
        normalizer_params=normalizer_params_final,
        env_steps=training_state.env_steps + total_steps_this_turn,
        apg_optimizer_state=training_state.apg_optimizer_state,
    )

    # Placeholder metrics for now
    final_metrics['apg_grad_norm'] = jnp.array(0.0)
    final_metrics['apg_loss'] = jnp.array(0.0)
    final_metrics['apg_cosine_similarity'] = jnp.array(0.0)

    return (final_training_state, state, new_key, step_count + 1), final_metrics

  def training_epoch(
      carry: Tuple[TrainingState, envs.State, PRNGKey, int], ## HYBRID PPO/APG: Add step_count to carry
  ) -> Tuple[TrainingState, envs.State, Metrics, int]: ## HYBRID PPO/APG: Return step_count
    training_state, state, key, step_count = carry ## HYBRID PPO/APG: Unpack step_count
    (training_state, state, _, step_count), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, key, step_count),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics, step_count ## HYBRID PPO/APG: Return updated step_count

  training_epoch_pmap = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME) ## HYBRID PPO/APG: Renamed to avoid collision

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, 
      env_state: envs.State, 
      key: PRNGKey, 
      step_count: jnp.ndarray ## HYBRID PPO/APG: Pass in step_count
  ) -> Tuple[TrainingState, envs.State, Metrics, jnp.ndarray]: ## HYBRID PPO/APG: Return step_count
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch_pmap( (training_state, env_state, key, step_count) ) ## HYBRID PPO/APG: Pass tuple to pmapped func
    training_state, env_state, metrics, step_count = _strip_weak_type(result) ## HYBRID PPO/APG: Unpack step_count

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics, step_count  # pytype: disable=bad-return-type  # py311-upgrade

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=ppo_optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape)
      ),
      env_steps=types.UInt64(hi=0, lo=0),
  )

  if apg_optimizer:
    training_state = training_state.replace(
        apg_optimizer_state=apg_optimizer.init(init_params)
    )

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(
            policy=params[1], value=value_params
        ),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(
            policy=restore_params[1], value=value_params
        ),
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0
  training_step_count = 0 ## HYBRID PPO/APG: Initialize step counter

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Run initial policy_params_fn.
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      # HYBRID PPO/APG: Pass and receive the training_step_count
      step_count_array = jnp.array(training_step_count)
      replicated_step_count = jax.device_put_replicated(
          step_count_array, jax.local_devices()[:local_devices_to_use]
      )

      (training_state, env_state, training_metrics, returned_replicated_count) = (
          training_epoch_with_timing(training_state, env_state, epoch_keys, replicated_step_count)
      )
      training_step_count = int(_unpmap(returned_replicated_count))

      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(
          save_checkpoint_path, current_step, params, ckpt_config
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
