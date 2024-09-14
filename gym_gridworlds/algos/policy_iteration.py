from gym_gridworlds.algos.policy_evaluation import PolicyEvaluation
from typing import Optional, Tuple
import numpy as np


class PolicyIteration(PolicyEvaluation):
    """ Class for performing Policy Iteration algorithm for Markov Decision Processes (MDPs).

    Inherits from `PolicyEvaluation` and provides methods to improve a policy iteratively
    until convergence using both a loop-based approach and a matrix-based approach.

    Attributes:
        gamma (float): Discount factor for future rewards.
        policy (np.ndarray): Current policy represented as a one-hot encoded matrix.
        rewards (np.ndarray): Reward matrix of shape (n_states, n_actions).
        terminal (np.ndarray): Terminal states matrix.
        transitions (np.ndarray): Transition probability matrix of shape (n_states, n_actions, n_states).
        n_states (int): Number of states in the MDP.
        n_actions (int): Number of actions in the MDP.
        initialization (float): Initial value for state-value function. Default is 0.
    """

    def __init__(
        self,
        gamma: float,
        rewards: np.ndarray,
        terminal: np.ndarray,
        transitions: np.ndarray,
        n_states: int,
        n_actions: int,
        initialization: float = 0,
        policy: Optional[np.ndarray] = None,
    ) -> None:
        """Initialization method.

        Args:
            gamma (float): Discount factor for future rewards.
            policy (Optional[np.ndarray]): Initial policy. If None, a uniform random policy is used.
            rewards (np.ndarray): Reward matrix of shape (n_states, n_actions).
            terminal (np.ndarray): Terminal states matrix.
            transitions (np.ndarray): Transition probability matrix of shape (n_states, n_actions, n_states).
            n_states (int): Number of states in the MDP.
            n_actions (int): Number of actions in the MDP.
            initialization (float): Initial value for state-value function. Default is 0.
        """
        super().__init__(
            gamma=gamma,
            policy=(
                policy
                if policy is not None
                else np.ones((n_states, n_actions)) / n_actions
            ),
            rewards=rewards,
            terminal=terminal,
            transitions=transitions,
            n_states=n_states,
            n_actions=n_actions,
            initialization=initialization,
        )

    def _reset(self):
        """Resets the policy to a uniform random policy.

        This method is called at the beginning of policy iteration to ensure the policy is initialized correctly.
        """
        super()._reset()
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def loop_policy_improvement(self) -> None:
        """Improves the policy using a loop-based approach.

        For each state, calculates the action values for all actions, selects the best action,
        and updates the policy accordingly.
        """
        for state in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for action in range(self.n_actions):
                action_value = 0.0
                for next_state in range(self.n_states):
                    action_value += (
                        self.transitions[state, action, next_state]
                        * (
                            self.rewards[state, action]
                            + self.gamma * self.state_values[next_state]
                        )[0]
                    )
                action_values[action] = action_value
            best_action = np.argmax(action_values)
            self.policy[state] = np.eye(self.n_actions)[best_action]

    def matrix_policy_improvement(self) -> None:
        """Placeholder for the matrix-based policy improvement.

        This method should be implemented to perform policy improvement using matrix operations.
        Currently raises a NotImplementedError.
        """
        # TODO: implement matrix form for PI
        raise NotImplemented(
            "Matrix calculations for policy improvement are not implemented yet."
        )

    def policy_iteration(
        self, threshold: float = 0.01, n_iter: Optional[int] = None
    ) -> Tuple[np.ndarray, list, int]:
        """Executes the policy iteration algorithm until the policy is stable.

        The process involves evaluating the policy, improving it, and checking for stability
        until convergence.

        Args:
            threshold (float): Convergence threshold for policy evaluation.
            n_iter (int, optional): Number of iterations for GPI. Defaults to None.

        Returns:
            tuple: Contains the final policy, a list of policy iteration errors, and the number of iterations performed.
        """
        self._reset()
        is_policy_stable = False
        pi_errors = []

        while not is_policy_stable:
            old_policy = self.policy.copy()
            values, errors = self.v_pi_iterative_form(threshold=threshold, n_iter=n_iter)
            pi_errors.append(errors)
            self.loop_policy_improvement()
            is_policy_stable = True if np.allclose(old_policy, self.policy) else False
        return self.policy, pi_errors, len(pi_errors)

    
    def loop_value_iteration(self, threshold: float = 0.01) -> Tuple[np.ndarray, list]:
        """Executes the value iteration algorithm until value convergence.

        It saves the best action while iterating over the state values. A matrix form method will be implemented soon.

        Args:
            threshold (float): Convergence threshold for value iteration.

        Returns:
            tuple: Contains the final policy, a list of value iteration errors.
        """
        delta: float = np.inf
        evaluation_errors = []

        while delta >= threshold:
            iter_error = 0.0
            delta = 0.0
            for state in range(self.n_states):
                current_value = self.state_values[state].copy()
                future_reward = self.gamma * np.multiply(np.dot(self.transitions[state], self.state_values).reshape(1,self.n_actions), (1-self.terminal[state]))
                action_values = self.rewards[state] + future_reward
                self.state_values[state] = np.max(action_values) 
                self.policy[state] = np.eye(self.n_actions)[np.argmax(action_values)]
                state_error = np.abs(current_value - self.state_values[state])[0]
                iter_error += state_error
                delta = np.max([delta, state_error])
            evaluation_errors.append(iter_error)

        return self.policy, evaluation_errors
