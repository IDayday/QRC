"""
High-efficiency episode-level HER replay buffer.

Key properties
--------------
1) Stores complete episodes in dedicated slots; no transition-level wraparound logic.
2) Preserves original RIS/HER rollout-goal / env-goal / replay-goal / future-goal sampling semantics.
3) Exposes episode/time-step metadata for adaptive local backup.
4) Uses vectorized episode/time-step sampling and vectorized future-goal sampling.
"""

import math
import numpy as np
from gym.spaces import Dict

from multiworld.core.image_env import unormalize_image, normalize_image

GOAL_TYPE_ROLLOUT = 0
GOAL_TYPE_ENV = 1
GOAL_TYPE_REPLAY = 2
GOAL_TYPE_FUTURE = 3
GOAL_TYPE_TO_NAME = {
    GOAL_TYPE_ROLLOUT: 'rollout',
    GOAL_TYPE_ENV: 'env',
    GOAL_TYPE_REPLAY: 'replay',
    GOAL_TYPE_FUTURE: 'future',
}


class PathBuilder(dict):
    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        return {k: stack_list(v) for k, v in self.items()}

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        return lst
    return np.array(lst)


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    return {key: flatten_n([d[key] for d in dicts]) for key in keys}


def preprocess_obs_dict(obs_dict):
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


class HERReplayBuffer:
    def __init__(
        self,
        max_size,
        env,
        max_episode_length,
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.0,
        fraction_resampled_goals_are_replay_buffer_goals=0.5,
        ob_keys_to_save=None,
        internal_keys=None,
        desired_goal_keys=None,
        goal_keys=None,
        observation_key='observation',
        desired_goal_key='desired_goal',
        achieved_goal_key='achieved_goal',
        vectorized=False,
        episode_slot_multiplier=4.0,
    ):
        self.env = env
        self.ob_spaces = self.env.observation_space.spaces
        self.max_size = int(max_size)  # nominal transition budget
        self.max_episode_length = int(max_episode_length)
        self.episode_slot_multiplier = float(episode_slot_multiplier)
        base_slots = int(math.ceil(self.max_size / float(self.max_episode_length)))
        self.max_episode_slots = max(8, int(math.ceil(base_slots * self.episode_slot_multiplier)) + 2)

        if ob_keys_to_save is not None:
            ob_keys_to_save = list(ob_keys_to_save)
        else:
            ob_keys_to_save = []
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = list(internal_keys)
        if goal_keys is None:
            goal_keys = [desired_goal_key]
        self.goal_keys = list(goal_keys)
        if desired_goal_keys is None:
            desired_goal_keys = [desired_goal_key]
        self.desired_goal_keys = list(desired_goal_keys)
        if desired_goal_key not in self.goal_keys:
            self.goal_keys.append(desired_goal_key)

        assert isinstance(env.observation_space, Dict)
        self.fraction_goals_rollout_goals = float(fraction_goals_are_rollout_goals)
        self.fraction_resampled_goals_env_goals = float(fraction_resampled_goals_are_env_goals)
        self.fraction_resampled_goals_replay_buffer_goals = float(fraction_resampled_goals_are_replay_buffer_goals)
        self.ob_keys_to_save = ob_keys_to_save
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.vectorized = bool(vectorized)

        self._rng = np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))

        self._action_dim = int(env.action_space.low.size)
        self._actions = np.zeros((self.max_episode_slots, self.max_episode_length, self._action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.max_episode_slots, self.max_episode_length, 1), dtype=np.float32)
        self._terminals = np.ones((self.max_episode_slots, self.max_episode_length, 1), dtype=np.uint8)

        self._obs = {}
        self._next_obs = {}
        for key in [observation_key, desired_goal_key, achieved_goal_key]:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)

        for key in ob_keys_to_save + self.internal_keys:
            assert key in self.ob_spaces, f"Key not found in observation space: {key}"
            dtype = np.uint8 if key.startswith('image') else np.float32
            dim = int(self.ob_spaces[key].low.size)
            self._obs[key] = np.zeros((self.max_episode_slots, self.max_episode_length, dim), dtype=dtype)
            self._next_obs[key] = np.zeros((self.max_episode_slots, self.max_episode_length, dim), dtype=dtype)

        self._episode_lengths = np.zeros(self.max_episode_slots, dtype=np.int32)
        self._valid_episodes = np.zeros(self.max_episode_slots, dtype=np.bool_)
        self._top_episode = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return int(self._size)

    def add_path(self, path):
        obs = path['observations']
        actions = path['actions']
        next_obs = path['next_observations']
        terminals = path['terminals']
        path_len = len(actions)

        if path_len <= 0:
            return
        if path_len > self.max_episode_length:
            raise ValueError(
                f"Episode length {path_len} exceeds max_episode_length={self.max_episode_length}. "
                "Increase max_episode_length or truncate episodes before insertion."
            )

        actions = flatten_n(actions).astype(np.float32, copy=False)
        terminals = flatten_n(terminals).astype(np.uint8, copy=False)
        obs = preprocess_obs_dict(flatten_dict(obs, self.ob_keys_to_save + self.internal_keys))
        next_obs = preprocess_obs_dict(flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys))

        slot = int(self._top_episode)
        if self._valid_episodes[slot]:
            self._size -= int(self._episode_lengths[slot])

        self._actions[slot, :path_len] = actions
        self._terminals[slot, :path_len] = terminals
        for key in self.ob_keys_to_save + self.internal_keys:
            self._obs[key][slot, :path_len] = obs[key]
            self._next_obs[key][slot, :path_len] = next_obs[key]

        if path_len < self.max_episode_length:
            self._actions[slot, path_len:] = 0
            self._terminals[slot, path_len:] = 1
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slot, path_len:] = 0
                self._next_obs[key][slot, path_len:] = 0

        self._episode_lengths[slot] = int(path_len)
        self._valid_episodes[slot] = True
        self._size += int(path_len)
        self._top_episode = (self._top_episode + 1) % self.max_episode_slots

    def _active_episode_slots(self):
        return np.flatnonzero(self._valid_episodes & (self._episode_lengths > 0))

    def _sample_episode_time_indices(self, batch_size):
        active_slots = self._active_episode_slots()
        if active_slots.size == 0:
            raise ValueError("Cannot sample from empty replay buffer.")
        lengths = self._episode_lengths[active_slots].astype(np.float64)
        probs = lengths / lengths.sum()
        episode_slots = self._rng.choice(active_slots, size=batch_size, replace=True, p=probs)
        chosen_lengths = self._episode_lengths[episode_slots].astype(np.int64)
        step_indices = self._rng.integers(np.zeros(batch_size, dtype=np.int64), chosen_lengths, dtype=np.int64)
        return episode_slots.astype(np.int64), step_indices.astype(np.int64)

    def _batch_obs_dict(self, episode_slots, step_indices):
        return {key: self._obs[key][episode_slots, step_indices].copy() for key in self.ob_keys_to_save}

    def _batch_next_obs_dict(self, episode_slots, step_indices):
        return {key: self._next_obs[key][episode_slots, step_indices].copy() for key in self.ob_keys_to_save}

    def sample_replay_buffer_states_as_goals(self, batch_size, return_indices=False):
        ep_slots, step_indices = self._sample_episode_time_indices(batch_size)
        goals = {}
        keys = list(set(self.goal_keys + self.desired_goal_keys))
        for key in keys:
            source_key = key.replace('desired', 'achieved') if 'desired' in key else key
            if source_key in self._next_obs:
                goals[key] = self._next_obs[source_key][ep_slots, step_indices].copy()
        if return_indices:
            return goals, ep_slots.astype(np.int64), step_indices.astype(np.int64)
        return goals

    def random_state_batch(self, batch_size):
        ep_slots, step_indices = self._sample_episode_time_indices(batch_size)
        new_obs = {self.observation_key: self._obs[self.observation_key][ep_slots, step_indices]}
        new_obs = postprocess_obs_dict(new_obs)
        return new_obs[self.observation_key]

    def get_future_transition_segment(self, episode_slot, start_step, future_goal_step, horizon_cap=None):
        episode_slot = int(episode_slot)
        start_step = int(start_step)
        future_goal_step = int(future_goal_step)
        if episode_slot < 0 or episode_slot >= self.max_episode_slots:
            return None
        if not self._valid_episodes[episode_slot]:
            return None
        length = int(self._episode_lengths[episode_slot])
        if start_step < 0 or future_goal_step < 0 or start_step >= length or future_goal_step >= length:
            return None
        if future_goal_step < start_step:
            return None
        if horizon_cap is None:
            end_step = future_goal_step
        else:
            end_step = min(future_goal_step, start_step + int(horizon_cap) - 1)
        if end_step < start_step:
            return None
        return np.arange(start_step, end_step + 1, dtype=np.int64)

    def get_segment_next_states(self, episode_slot, step_indices):
        return self._next_obs[self.observation_key][int(episode_slot), np.asarray(step_indices, dtype=np.int64)]

    def _assign_goal_block(self, new_obs_dict, new_next_obs_dict, resampled_goals, start, end, source_goals):
        for desired_goal_key in self.desired_goal_keys:
            if desired_goal_key in source_goals:
                resampled_goals[desired_goal_key][start:end] = source_goals[desired_goal_key]
            else:
                achieved_key = desired_goal_key.replace('desired', 'achieved')
                resampled_goals[desired_goal_key][start:end] = source_goals[achieved_key]

        for goal_key in self.goal_keys:
            if goal_key in source_goals:
                value = source_goals[goal_key]
            else:
                source_key = goal_key.replace('desired', 'achieved') if 'desired' in goal_key else goal_key
                value = source_goals[source_key]
            new_obs_dict[goal_key][start:end] = value
            new_next_obs_dict[goal_key][start:end] = value

    def random_batch(self, batch_size):
        ep_slots, step_indices = self._sample_episode_time_indices(batch_size)

        goal_source_type = np.full(batch_size, GOAL_TYPE_ROLLOUT, dtype=np.int64)
        goal_source_episode_slot = np.full(batch_size, -1, dtype=np.int64)
        goal_source_index = np.full(batch_size, -1, dtype=np.int64)

        new_obs_dict = self._batch_obs_dict(ep_slots, step_indices)
        new_next_obs_dict = self._batch_next_obs_dict(ep_slots, step_indices)
        resampled_goals = {
            desired_goal_key: self._next_obs[desired_goal_key][ep_slots, step_indices].copy()
            for desired_goal_key in self.desired_goal_keys
        }

        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_env_goals = int(batch_size * (1.0 - self.fraction_goals_rollout_goals) * self.fraction_resampled_goals_env_goals)
        num_replay_buffer_goals = int(batch_size * (1.0 - self.fraction_goals_rollout_goals) * self.fraction_resampled_goals_replay_buffer_goals)
        num_future_goals = batch_size - (num_rollout_goals + num_env_goals + num_replay_buffer_goals)

        # Environment goals.
        if num_env_goals > 0:
            from inspect import signature
            sig = signature(self.env.sample_goals)
            if len(sig.parameters) == 2:
                keys = self.goal_keys + self.desired_goal_keys
                env_goals = self.env.sample_goals(num_env_goals, keys=set(keys))
            else:
                env_goals = self.env.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            s0 = num_rollout_goals
            s1 = s0 + num_env_goals
            self._assign_goal_block(new_obs_dict, new_next_obs_dict, resampled_goals, s0, s1, env_goals)
            goal_source_type[s0:s1] = GOAL_TYPE_ENV

        # Replay-buffer goals.
        if num_replay_buffer_goals > 0:
            replay_goals, replay_ep_slots, replay_steps = self.sample_replay_buffer_states_as_goals(
                num_replay_buffer_goals, return_indices=True
            )
            s0 = num_rollout_goals + num_env_goals
            s1 = s0 + num_replay_buffer_goals
            self._assign_goal_block(new_obs_dict, new_next_obs_dict, resampled_goals, s0, s1, replay_goals)
            goal_source_type[s0:s1] = GOAL_TYPE_REPLAY
            goal_source_episode_slot[s0:s1] = replay_ep_slots
            goal_source_index[s0:s1] = replay_steps

        # Future goals: vectorized within-episode future sampling.
        if num_future_goals > 0:
            s0 = batch_size - num_future_goals
            future_ep_slots = ep_slots[s0:]
            future_step_starts = step_indices[s0:]
            future_step_ends = self._episode_lengths[future_ep_slots] - 1
            future_steps = self._rng.integers(future_step_starts, future_step_ends + 1, dtype=np.int64)
            future_goal_dict = {}
            needed_keys = list(set(self.goal_keys + self.desired_goal_keys))
            for key in needed_keys:
                source_key = key.replace('desired', 'achieved') if 'desired' in key else key
                if source_key in self._next_obs:
                    future_goal_dict[key] = self._next_obs[source_key][future_ep_slots, future_steps]
            self._assign_goal_block(new_obs_dict, new_next_obs_dict, resampled_goals, s0, batch_size, future_goal_dict)
            goal_source_type[s0:] = GOAL_TYPE_FUTURE
            goal_source_episode_slot[s0:] = future_ep_slots
            goal_source_index[s0:] = future_steps

        for desired_goal_key in self.desired_goal_keys:
            new_obs_dict[desired_goal_key] = resampled_goals[desired_goal_key]
            new_next_obs_dict[desired_goal_key] = resampled_goals[desired_goal_key]

        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)

        new_actions = self._actions[ep_slots, step_indices]
        new_rewards = self.env.compute_rewards(new_actions, new_next_obs_dict)
        if not self.vectorized:
            new_rewards = new_rewards.reshape(-1, 1)

        batch = {
            'observations': new_obs_dict[self.observation_key],
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[ep_slots, step_indices],
            'next_observations': new_next_obs_dict[self.observation_key],
            'resampled_goals': new_next_obs_dict[self.desired_goal_key],
            'episode_slots': ep_slots.reshape(-1, 1),
            'step_indices': step_indices.reshape(-1, 1),
            'goal_source_type': goal_source_type.reshape(-1, 1),
            'goal_source_episode_slot': goal_source_episode_slot.reshape(-1, 1),
            'goal_source_index': goal_source_index.reshape(-1, 1),
        }
        return batch
