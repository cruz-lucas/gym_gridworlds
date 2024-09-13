import gymnasium
import numpy as np
from typing import Tuple


def get_env_specs(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Retrieves environment specifications.

    This includes reward matrix, transition probability matrix, terminal state indicators,
    and the number of states and actions.

    Args:
        name (str): The name of the environment to be created. This should be a
            valid Gymnasium environment identifier.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]: A tuple containing:
            - R (np.ndarray): A reward matrix of shape (n_states, n_actions), where
              R[s, a] represents the reward received after taking action `a` in state `s`.
            - P (np.ndarray): A transition probability matrix of shape
              (n_states, n_actions, n_states), where P[s, a, s_next] represents
              the probability of transitioning to state `s_next` from state `s`
              when action `a` is taken.
            - T (np.ndarray): A terminal matrix of shape (n_states, n_actions),
              where T[s, a] is 1 if taking action `a` in state `s` leads to a terminal
              state, otherwise 0.
            - n_states (int): The total number of states in the environment.
            - n_actions (int): The total number of actions available in the environment.
    """
    env = gymnasium.make(name)
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    R = np.zeros((n_states, n_actions))
    P = np.zeros((n_states, n_actions, n_states))
    T = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            env.unwrapped.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            R[s, a] = r
            P[s, a, s_next] = 1.0
            T[s, a] = terminated

    return R, P, T, n_states, n_actions
