import numpy as np
from typing import List


class Evaluator:
    def __init__(self, env, initialization: int = 0, gamma: float = 0.99, threshold: float = 0.1) -> None:
        """
        Initializes the Evaluator class.

        Args:
            env: The environment object with observation and action spaces.
            initialization (int, optional): Initialization value for state and action values. Defaults to 0.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            threshold (float, optional): Threshold value for the stopping criterion. Defaults to 0.1.
        """
        self.n_states: int = env.observation_space.n
        self.n_actions: int = env.action_space.n

        self.state_values: np.ndarray = np.zeros(self.n_states) + initialization
        self.q_values: np.ndarray = np.zeros((self.n_states, self.n_actions)) + initialization
        self.gamma: float = gamma
        self.threshold: float = threshold
        self.errors: List[float] = []

    def evaluate_state_values(self, policy: np.ndarray, R: np.ndarray, P: np.ndarray) -> None:
        """
        Evaluates state values using the provided policy.

        Args:
            policy (np.ndarray): The policy array where each entry policy[state] is a probability distribution over actions.
            R (np.ndarray): The reward array where R[state] is the reward for each state.
            P (np.ndarray): The transition probability array where P[state, action, next_state] represents the probability
                            of transitioning to next_state from state when taking action.

        Returns:
            None
        """
        delta: float = np.inf
        while delta >= self.threshold:
            delta = 0.0
            iteration_error: float = 0.0
            for state in range(self.n_states):
                current_value: float = self.state_values[state]
                next_states: np.ndarray = np.argmax(P[state], axis=1)
                self.state_values[state] = np.sum(policy[state] * (R[state] + self.gamma * self.state_values[next_states]))
                error: float = np.abs(current_value - self.state_values[state])
                iteration_error += error
                delta = max(delta, error)
            self.errors.append(iteration_error)

    def evaluate_action_state_values(self, R: np.ndarray, P: np.ndarray) -> None:
        """
        Evaluates action-state values using the provided reward and transition probabilities.

        Args:
            R (np.ndarray): The reward array where R[state, action] is the reward for each state-action pair.
            P (np.ndarray): The transition probability array where P[state, action, next_state] represents the probability
                            of transitioning to next_state from state when taking action.

        Returns:
            None
        """
        delta: float = np.inf
        while delta >= self.threshold:
            delta = 0.0
            iteration_error: float = 0.0
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    current_value: float = self.q_values[state, action]
                    next_state: int = np.argmax(P[state, action])
                    self.q_values[state, action] = R[state, action] + self.gamma * np.max(self.q_values[next_state, :])
                    error: float = np.abs(current_value - self.q_values[state, action])
                    iteration_error += error
                delta = max(delta, error)
            self.errors.append(iteration_error)
