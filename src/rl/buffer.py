"""
Experience Replay Buffer for offline/batch RL training.

Stores transitions: (state, action, reward, next_state, done, info)
Supports episodic and flat storage modes.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from collections import deque


class Transition:
    """A single transition tuple."""
    __slots__ = ("state", "action", "reward", "next_state", "done", "info")

    def __init__(self, state: Dict, action: int, reward: float,
                 next_state: Dict, done: bool, info: Dict):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info


class ReplayBuffer:
    """
    Flat replay buffer for offline RL.

    Stores individual transitions. For offline RL, the buffer is typically
    pre-populated with data collected by a logging/behavior policy.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.transitions: deque = deque(maxlen=capacity)

    def add(self, state: Dict, action: int, reward: float,
            next_state: Dict, done: bool, info: Dict = None):
        self.transitions.append(Transition(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done, info=info or {},
        ))

    def add_episode(self, episode: List[Tuple]):
        """Add a full episode of (s, a, r, s', d, info) tuples."""
        for s, a, r, ns, d, info in episode:
            self.add(s, a, r, ns, d, info)

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Returns a dict of tensors ready for training:
          - user_emb, memory_context, retrieval_confidence, periodic_flags
          - action_mask (for current state)
          - actions, rewards, dones
          - next_user_emb, next_memory_context, next_retrieval_confidence,
            next_periodic_flags, next_action_mask
        """
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)

        batch = {
            "user_emb": [],
            "memory_context": [],
            "retrieval_confidence": [],
            "periodic_flags": [],
            "action_mask": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_user_emb": [],
            "next_memory_context": [],
            "next_retrieval_confidence": [],
            "next_periodic_flags": [],
            "next_action_mask": [],
        }

        for idx in indices:
            t = self.transitions[idx]
            s = t.state
            ns = t.next_state

            batch["user_emb"].append(s["user_emb"])
            batch["memory_context"].append(s["memory_context"])
            batch["retrieval_confidence"].append(s["retrieval_confidence"])
            batch["periodic_flags"].append(s.get("periodic_flags",
                                                  np.zeros(10, dtype=np.float32)))
            batch["action_mask"].append(s.get("action_mask",
                                              np.ones(16, dtype=np.float32)))
            batch["actions"].append(t.action)
            batch["rewards"].append(t.reward)
            batch["dones"].append(float(t.done))

            batch["next_user_emb"].append(ns["user_emb"])
            batch["next_memory_context"].append(ns["memory_context"])
            batch["next_retrieval_confidence"].append(ns["retrieval_confidence"])
            batch["next_periodic_flags"].append(ns.get("periodic_flags",
                                                       np.zeros(10, dtype=np.float32)))
            batch["next_action_mask"].append(ns.get("action_mask",
                                                    np.ones(16, dtype=np.float32)))

        for k, v in batch.items():
            if k in ("actions",):  # int64 indices
                batch[k] = torch.tensor(v, dtype=torch.long, device=device)
            elif k in ("dones",):  # float32 for dones
                batch[k] = torch.tensor(v, dtype=torch.float32, device=device)
            else:
                if isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack([x.to(device) for x in v])
                else:
                    batch[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)

        return batch

    def __len__(self) -> int:
        return len(self.transitions)

    def save(self, path: str):
        torch.save({
            "transitions": list(self.transitions),
            "capacity": self.capacity,
        }, path)

    def load(self, path: str):
        data = torch.load(path)
        self.capacity = data["capacity"]
        self.transitions = deque(data["transitions"], maxlen=self.capacity)


class EpisodicReplayBuffer(ReplayBuffer):
    """
    Episodic replay buffer that preserves episode boundaries.
    Useful for return-to-go conditioning and trajectory-level operations.
    """

    def __init__(self, capacity: int = 100_000, max_episode_length: int = 50):
        super().__init__(capacity)
        self.max_episode_length = max_episode_length
        self.episodes: List[List[Transition]] = []
        self._current_episode: List[Transition] = []

    def begin_episode(self):
        self._current_episode = []

    def end_episode(self):
        if self._current_episode:
            self.episodes.append(self._current_episode)
            # Also add to flat buffer
            for t in self._current_episode:
                self.transitions.append(t)
            self._current_episode = []

    def add_to_episode(self, state: Dict, action: int, reward: float,
                       next_state: Dict, done: bool, info: Dict = None):
        t = Transition(state, action, reward, next_state, done, info or {})
        self._current_episode.append(t)
        if done:
            self.end_episode()

    def sample_episodes(self, num_episodes: int) -> List[List[Transition]]:
        """Sample full episodes."""
        indices = np.random.choice(len(self.episodes), num_episodes, replace=False)
        return [self.episodes[i] for i in indices]
