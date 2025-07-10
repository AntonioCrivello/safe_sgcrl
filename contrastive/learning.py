"""Contrastive RL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb
from jax.experimental.host_callback import id_print
from jax import debug
from jax.scipy.special import logsumexp
import numpy as np
from jax import random
import os
from default import make_default_logger
from pathlib import Path
import os, functools, tensorflow as tf, jax

class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  policy_params_prev: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class ContrastiveLearner(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      policy_optimizer,
      q_optimizer,
      iterator,
      counter,
      logger,
      obs_to_goal,
      config):
    """Initialize the Contrastive RL learner.

    Args:
      networks: Contrastive RL networks.
      rng: a key for random number generation.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      iterator: an iterator over training data.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      obs_to_goal: a function for extracting the goal coordinates.
      config: the experiment config file.
    """
    if config.add_mc_to_td:
      assert config.use_td
    adaptive_entropy_coefficient = config.entropy_coefficient is None
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._use_td = config.use_td
    self.config = config
    self._reset_counter = 0
    self._networks = networks
    self._policy_optimizer = policy_optimizer
    self._q_optimizer = q_optimizer
    self.adaptive_entropy_coefficient = adaptive_entropy_coefficient
    
    if adaptive_entropy_coefficient:
      # alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    else:
      if config.target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    def alpha_loss(log_alpha,
                   policy_params,
                   transitions,
                   key):
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jnp.exp(log_alpha)
      alpha_loss = alpha * jax.lax.stop_gradient(
          -log_prob - config.target_entropy)
      return jnp.mean(alpha_loss)


    def critic_loss(q_params,
                    policy_params,
                    target_q_params,
                    transitions,
                    key,
                    use_goal_neg: bool = False):
      batch_size = transitions.observation.shape[0]
      # Note: We might be able to speed up the computation for some of the
      # baselines to making a single network that returns all the values. This
      # avoids computing some of the underlying representations multiple times.
      if config.use_td:
        # For TD learning, the diagonal elements are the immediate next state.
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        next_s, _ = jnp.split(transitions.next_observation, [config.obs_dim],
                              axis=1)
        if config.add_mc_to_td:
          next_fraction = (1 - config.discount) / ((1 - config.discount) + 1)
          num_next = int(batch_size * next_fraction)
          new_g = jnp.concatenate([
              obs_to_goal(next_s[:num_next]),
              g[num_next:],
          ], axis=0)
        else:
          new_g = obs_to_goal(next_s)
        obs = jnp.concatenate([s, new_g], axis=1)
        transitions = transitions._replace(observation=obs)
      I = jnp.eye(batch_size)  # pylint: disable=invalid-name

      
      logits, _, _ = networks.q_network.apply(q_params, transitions.observation, transitions.action)

      if config.use_td:
        # Make sure to use the twin Q trick.
        assert len(logits.shape) == 3

        # We evaluate the next-state Q function using random goals
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        del s
        next_s = transitions.next_observation[:, :config.obs_dim]
        goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
        g = g[goal_indices]
        transitions = transitions._replace(
            next_observation=jnp.concatenate([next_s, g], axis=1))
        next_dist_params = networks.policy_network.apply(
            policy_params, transitions.next_observation)
        next_action = networks.sample(next_dist_params, key)
        
        next_q, _, _ = networks.q_network.apply(target_q_params,
                                          transitions.next_observation,
                                          next_action)  # This outputs logits.
        next_q = jax.nn.sigmoid(next_q)
        next_v = jnp.min(next_q, axis=-1)
        next_v = jax.lax.stop_gradient(next_v)
        next_v = jnp.diag(next_v)
        # diag(logits) are predictions for future states.
        # diag(next_q) are predictions for random states, which correspond to
        # the predictions logits[range(B), goal_indices].
        # So, the only thing that's meaningful for next_q is the diagonal. Off
        # diagonal entries are meaningless and shouldn't be used.
        w = next_v / (1 - next_v)
        w_clipping = 20.0
        w = jnp.clip(w, 0, w_clipping)
        # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
        pos_logits = jax.vmap(jnp.diag, -1, -1)(logits)
        loss_pos = optax.sigmoid_binary_cross_entropy(
            logits=pos_logits, labels=1)  # [B, 2]

        neg_logits = logits[jnp.arange(batch_size), goal_indices]
        loss_neg1 = w[:, None] * optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=1)  # [B, 2]
        loss_neg2 = optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=0)  # [B, 2]

        if config.add_mc_to_td:
          loss = ((1 + (1 - config.discount)) * loss_pos
                  + config.discount * loss_neg1 + 2 * loss_neg2)
        else:
          loss = ((1 - config.discount) * loss_pos
                  + config.discount * loss_neg1 + loss_neg2)
        # Take the mean here so that we can compute the accuracy.
        logits = jnp.mean(logits, axis=-1)

      else:  # For the MC losses.
        def loss_fn(_logits):  # pylint: disable=invalid-name
          if config.use_cpc:
            #jax.debug.print("[DBG] use goal neg {}", use_goal_neg)
            fixed_goal = self.config.fixed_goal
            fixed_goal = jnp.asarray(fixed_goal, dtype=jnp.float32)   # shape (d,)
            B = _logits.shape[0]
            
            # ## Adding goal as a negative example
            labels = I
            if (fixed_goal is not None) and use_goal_neg:
              # ensure fixed_goal is a JAX array, not a Python list
              #debug.print("[DBG] fixed_goal is not None, using it as a negative example")
              
              


              # split current states  s  |  g
              s, _ = jnp.split(transitions.observation,
                              [config.obs_dim], axis=1)
              

      

              # replicate the fixed goal so we have B copies
              g_fixed = jnp.broadcast_to(fixed_goal, (B, fixed_goal.shape[-1]))

              obs_fixed = jnp.concatenate([s, g_fixed], axis=1)        # (B , 2*obs_dim)
              fixed_logits, _, _ = networks.q_network.apply(
                  q_params, obs_fixed, transitions.action)             # (B [,2])
              fixed_logits = fixed_logits[:, 0]    

              # ensure shape is (B ,1)  or  (B ,1 ,2) in twin-Q case
              if fixed_logits.ndim == 1:
                  fixed_logits = fixed_logits[:, None]
              else:
                  fixed_logits = fixed_logits[:, None, :]

              # append as a new column on the right
              _logits = jnp.concatenate([_logits, fixed_logits], axis=1)
              #debug.print("[DBG] logits shape after concat {}", _logits.shape)

              # extend the label matrix (all zeros → still a negative)
              labels = jnp.concatenate(
                      [I, jnp.zeros((B, 1), I.dtype)],
                      axis=1)
            ## Using backward loss
            if config.backward_loss:
              _logits = _logits.T
              return (optax.softmax_cross_entropy(logits=_logits, labels=labels))
            else:
              return (optax.softmax_cross_entropy(logits=_logits, labels=labels)
                    + 0.01 * jax.nn.logsumexp(_logits, axis=1)**2)
          else:
            return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
        if len(logits.shape) == 3:  # twin q
          # loss.shape = [.., num_q]
          loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
          loss = jnp.mean(loss, axis=-1)
          # Take the mean here so that we can compute the accuracy.
          logits = jnp.mean(logits, axis=-1)
        else:
          loss = loss_fn(logits)

      loss = jnp.mean(loss)
      correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
      logits_pos = jnp.sum(logits * I) / jnp.sum(I)
      logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
      if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
      else:
        logsumexp = jax.nn.logsumexp(logits, axis=1)**2
      metrics = {
          'binary_accuracy': jnp.mean((logits > 0) == I),
          'categorical_accuracy': jnp.mean(correct),
          'logits_pos': logits_pos,
          'logits_neg': logits_neg,
          'logsumexp': logsumexp.mean(),
      }

      return loss, metrics

    def actor_loss(policy_params,
                   q_params,
                   alpha,
                   transitions,
                   key,
                   ):
      obs = transitions.observation

      state = obs[:, :config.obs_dim]
      goal = obs[:, config.obs_dim:]

      if config.random_goals == 0.0:
        new_state = state
        new_goal = goal
      elif config.random_goals == 0.5:
        new_state = jnp.concatenate([state, state], axis=0)
        new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      else:
        assert config.random_goals == 1.0
        new_state = state
        new_goal = jnp.roll(goal, 1, axis=0)

      new_obs = jnp.concatenate([new_state, new_goal], axis=1) 
      dist_params = networks.policy_network.apply(policy_params, new_obs)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)

      q_action, sa_repr, sf_repr = networks.q_network.apply(q_params, new_obs, action)

      if len(q_action.shape) == 3:  # twin q trick
        assert q_action.shape[2] == 2
        q_action = jnp.min(q_action, axis=-1)

      actor_loss = -jnp.diag(q_action) # negative -(Q): maximize Q

      # action entropy loss
      approx_entropy = -log_prob

      if config.use_action_entropy:
        actor_loss -= alpha * approx_entropy # negative -(-log prob): maximize entropy

      metrics = {
          'entropy_mean': jnp.mean(approx_entropy),
      }

      
      return jnp.mean(actor_loss), metrics

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def update_step(
            state,
            transitions,
            *,                       # keep it a keyword-only arg
            use_goal_neg: bool,
    ):
  
      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      # (critic_loss, critic_metrics), critic_grads = critic_grad(
      #     state.q_params, state.policy_params, state.target_q_params,
      #     transitions, key_critic)
      (critic_loss, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic, use_goal_neg=use_goal_neg)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(critic_grads, state.q_optimizer_state)

      q_params = optax.apply_updates(state.q_params, critic_update)
       
      new_target_q_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                         state.target_q_params, q_params)
      metrics = critic_metrics
      
      # compute actor loss                               
      (actor_loss, actor_metrics), actor_grads = actor_grad(state.policy_params, state.q_params, 
                                                            alpha, transitions, key_actor)
                                     
      # Apply policy gradients
      policy_params_prev = state.policy_params
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)
                                     
      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      })
      
      metrics.update(actor_metrics)
  
      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev=policy_params_prev,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key
      )
    
      if adaptive_entropy_coefficient:
        # Apply alpha gradients
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)
    
      return_state = new_state
      return return_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
        time_delta=10.0)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    # update_step = utils.process_multiple_batches(update_step,config.num_sgd_steps_per_step)
    # # Use the JIT compiler.
    # if config.jit:
    #   self._update_step = jax.jit(update_step)
    # else:
    #   self._update_step = update_step

    def make_update(use_goal_neg: bool):
      # 1. freeze the flag so the inner fn now has *only* (state, trans)
      step_fn = functools.partial(update_step, use_goal_neg=use_goal_neg)
      # 2. let Acme split big batches if requested
      step_fn = utils.process_multiple_batches(
          step_fn, config.num_sgd_steps_per_step
      )
      # 3. JIT if desired (no extra static args now)
      return jax.jit(step_fn) if config.jit else step_fn

    self._update_step_true  = make_update(True)   # goal is a negative
    self._update_step_false = make_update(False)  # stop using it


    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev = policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key)
      print("Initialising networks with random parameters.", flush = True)
      if adaptive_entropy_coefficient:
        print("Using adaptive entropy coefficient.", flush = True)
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)
        
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online
    # and fill the replay buffer.
    self._timestamp = None

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)



      def reset_network_state():
        """Reset policy, Q, and target Q params — keep everything else."""
        key_policy, key_q, key = jax.random.split(self._state.key, 3)
        new_policy_params = self._networks.policy_network.init(key_policy)
        new_q_params = self._networks.q_network.init(key_q)

        # Replace only the desired fields
        new_state = self._state._replace(
            policy_params=new_policy_params,
            policy_params_prev=new_policy_params,
            q_params=new_q_params,
            target_q_params=new_q_params,
            key=key
        )
        
        return new_state

      ## Added logic to support adding fixed goal as negative example
      counts = self._counter.get_counts()          # Python dict
      actor_steps = counts.get('actor_steps', 0)


      
      ## reset the network weights
      self._reset_counter += 1
      if actor_steps > 0 and self.config.weight_reset_interval> 0:
        if self._reset_counter > self.config.weight_reset_interval:
          self._reset_counter = 0
          print(f"💥 Resetting networks at actor step {actor_steps}", flush= True)
          self._state = reset_network_state()
          




    
      # NEW -- randomly replace 20 % of future states with the fixed goal
      ## this function only works for point env enviornment needs to be adjusted for other enviornments
      # ──────────────────────────────────────────────────────────────
      if self.config.fixed_goal is not None and actor_steps < self.config.goal_pos_actor_steps:
        obs        = transitions.observation           # (B, obs_dim*2)
        actions    = transitions.action                # (B, act_dim)
        B          = obs.shape[0]
        obs_dim    = self.config.obs_dim               # e.g. 2
        act_dim    = actions.shape[-1]                 # e.g. 2
        frac       = self.config.goal_pos_frac         # e.g. 0.2
        k1, k2     = jax.random.split(self._state.key)

        idx        = jax.random.choice(
                      k1, B,
                      (max(1, int(B * frac)),),
                      replace=False)

        fixed_goal = jnp.asarray(self.config.fixed_goal, dtype=obs.dtype)

        # ---- DEBUG PRINT: Before changing anything
        # idx0 = idx[0]
        # before_obs = obs[idx0]
        # before_action = actions[idx0]
        # print("▶ Before update:")
        # print("  obs[idx0]:", np.array(before_obs))
        # print("  state    :", np.array(before_obs[:obs_dim]))
        # print("  goal     :", np.array(before_obs[obs_dim:2*obs_dim]))
        # print("  action   :", np.array(before_action))

        # Set goal part
        obs = obs.at[idx, obs_dim:2*obs_dim].set(
                jnp.broadcast_to(fixed_goal, (idx.size, obs_dim)))

        # Sample random actions ∈ [-1, 1]
        k3, k2 = jax.random.split(k2)
        random_actions = jax.random.uniform(
            k3, shape=(idx.size, act_dim),
            minval=-1.0, maxval=1.0)

        # Set the actions directly in transitions.action
        actions = actions.at[idx].set(random_actions)

        # ---- DEBUG PRINT: After update
        # after_obs = obs[idx0]
        # after_action = actions[idx0]
        # print("▶ After update:")
        # print("  obs[idx0]:", np.array(after_obs))
        # print("  state    :", np.array(after_obs[:obs_dim]))
        # print("  goal     :", np.array(after_obs[obs_dim:2*obs_dim]))
        # print("  action   :", np.array(after_action))

        # Replace transitions
        transitions = transitions._replace(
            observation=obs,
            action=actions
        )
        self._state = self._state._replace(key=k2)



      ### Goal negative sampling
      # If the number of actor steps is less than goal_neg_actor_steps then use the goal as a negative example
      use_goal_neg  = actor_steps < self.config.goal_neg_actor_steps
      update_fn = (self._update_step_true
                 if use_goal_neg
                 else self._update_step_false)

      
      self._state, metrics = update_fn(self._state, transitions)

      #self._state, metrics = self._update_step(self._state, transitions) 
    
    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    
    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    
    if elapsed_time > 0:
      metrics['steps_per_second'] = (
          self._num_sgd_steps_per_step / elapsed_time)
    else:
      metrics['steps_per_second'] = 0.
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
