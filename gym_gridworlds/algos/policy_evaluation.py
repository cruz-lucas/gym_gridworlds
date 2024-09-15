from gym_gridworlds.algos.base_algo import BaseAlgo
import numpy as np
from typing import Tuple, Optional


class PolicyEvaluation(BaseAlgo):
    """A class to evaluate a policy using both closed-form and iterative methods.

    This class provides methods to evaluate the value function of a policy
    in a Markov Decision Process (MDP) with both closed-form and iterative
    approaches. The evaluation accounts for terminal states, ensuring that
    state values for terminal actions only consider immediate rewards without
    discounting future state values.

    Attributes:
        gamma (float): Discount factor for future rewards, controlling the
            weighting of future state values.
        policy (np.ndarray): A 2D array representing the policy, where
            policy[state, action] is the probability of taking an action
            in a given state.
        rewards (np.ndarray): A 2D array of rewards, where rewards[state, action]
            represents the immediate reward received after taking an action in a state.
        terminal (np.ndarray): A 2D array indicating terminal actions, where
            terminal[state, action] is 1 if the action in the state leads to a terminal state.
        transitions (np.ndarray): A 3D array of transition probabilities, where
            transitions[state, action, next_state] indicates the probability of
            transitioning to the next state from the current state when taking a given action.
        n_states (int): The total number of states in the MDP.
        n_actions (int): The total number of actions in the MDP.
        initialization (float): The initial value assigned to each state value during evaluation.
    """

    def __init__(
        self,
        gamma: float,
        policy: np.ndarray,
        rewards: np.ndarray,
        terminal: np.ndarray,
        transitions: np.ndarray,
        n_states: int,
        n_actions: int,
        initialization: float = 0,
    ) -> None:
        """Initializes the PolicyEvaluation class.

        Args:
            gamma (float): Discount factor for future rewards.
            policy (np.ndarray): The policy array with shape (n_states, n_actions),
                where each row is a probability distribution over actions for a given state.
            rewards (np.ndarray): Reward matrix of shape (n_states, n_actions),
                where each element is the reward for a specific state-action pair.
            terminal (np.ndarray): Terminal state-action matrix of shape (n_states, n_actions),
                indicating which actions lead to terminal states.
            transitions (np.ndarray): Transition probability matrix with shape (n_states, n_actions, n_states),
                defining the probabilities of reaching next states given current state-action pairs.
            n_states (int): Number of states in the environment.
            n_actions (int): Number of actions in the environment.
            initialization (float, optional): Initialization value for state values.
                Defaults to 0.
        """
        super().__init__(gamma=gamma)
        self.policy = policy
        self.rewards = rewards
        self.transitions = transitions
        self.terminal = terminal
        self.initialization = initialization
        self.n_states = n_states
        self.n_actions = n_actions
        self._reset()

    def _reset(self, initialization: Optional[float] = None) -> None:
        """Resets the state values to the initialization value.

        This method initializes the state values to a specified value,
        preparing the object for evaluation computations.

        Args:
            initialization (float, optional): Initialization value for state values.
                Defaults to 0.
        """
        self.initialization = initialization if initialization else self.initialization
        self.state_values: np.ndarray = (
            np.zeros((self.n_states, 1)) + self.initialization
        )
        self.q_values = np.zeros((self.n_states, self.n_actions)) + self.initialization

    def v_pi_closed_form(self) -> np.ndarray:
        """Evaluates the policy using a closed-form solution of the Bellman equation.

        This method solves the value function using a matrix inversion approach, it solves
        the system V = (I - gamma * P_pi)^(-1) * R_pi, accounting for terminal states
        by adjusting transition probabilities and rewards appropriately.

        Returns:
            np.ndarray: A 1D array of evaluated state values of shape (n_states,).
        """
        P_pi = np.einsum("sa,san->sn", self.policy, self.transitions)
        R_pi = np.einsum("sa,sa->s", self.policy, self.rewards)
        terminal_mask = np.einsum("sa->s", self.policy * self.terminal) > 0
        P_pi[terminal_mask, :] = 0
        R_pi[terminal_mask] = np.einsum(
            "sa,sa->s", self.policy, self.rewards * self.terminal
        )[terminal_mask]
        I = np.eye(P_pi.shape[0])
        V = np.linalg.solve(I - self.gamma * P_pi, R_pi)
        return V

    def v_pi_iterative_form(self, threshold: float = 0.01, n_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Iteratively evaluates the policy until convergence using the Bellman equation.

        This method performs iterative updates of the state values, accounting
        for terminal states by directly setting values based on immediate rewards
        without considering future values for terminal transitions. There is no loop through the
        states, it solves the update in the matrix form.

        Args:
            threshold (float, optional): Convergence threshold for stopping criterion,
                determining when the updates have become sufficiently small. Defaults to 0.01.
            n_iter (int, optional): Number of iterations for GPI. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: A 2D array of evaluated state values of shape (n_states, 1).
                - np.ndarray: A 1D array containing the sum of iteration errors for each iteration.
        """
        delta: float = np.inf
        evaluation_errors = []
        counter = 0 if n_iter is not None else None

        while delta >= threshold:
            current_value = self.state_values.copy()
            state_transition = np.einsum("sa,san->sn", self.policy, self.transitions)

            assert state_transition.shape == (
                self.n_states,
                self.n_states,
            ), f"The state transition matrix must map state to next state, and have shape {(self.n_states, self.n_states)}. Got: {state_transition.shape}"

            next_state_value = np.dot(state_transition, self.state_values)
            current_reward = np.einsum("sa,sa->s", self.rewards, self.policy)[
                :, np.newaxis
            ]

            next_state_terminal = np.einsum(
                "sn,sa->n", state_transition, self.terminal
            )[:, np.newaxis]
            self.state_values = current_reward + np.multiply(
                self.gamma * next_state_value, (1 - next_state_terminal)
            )
            iteration_error = np.abs(current_value - self.state_values)
            delta = np.max([0, np.max(iteration_error)])
            evaluation_errors.append(np.sum(iteration_error))

            if n_iter is not None:
                counter += 1
                if counter >= n_iter:
                    return self.state_values, np.array(evaluation_errors)            

        return self.state_values, np.array(evaluation_errors)
    
    def q_pi_iterative_form(self, threshold: float = 0.01, n_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Iteratively evaluates the action-state values until convergence using the Bellman equation.

        Args:
            threshold (float, optional): Convergence threshold for stopping criterion.
            n_iter (int, optional): Number of iterations for GPI. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: Evaluated action-state values of shape (n_actions, n_states).
                - np.ndarray: Iteration errors for each iteration.
        """
        delta: float = np.inf
        evaluation_errors = []
        counter = 0 if n_iter is not None else None
        while delta >= threshold:
            delta = 0.0
            iteration_error: float = 0.0
            for state in range(self.n_states):
                state_error = 0.0
                for action in range(self.n_actions):
                    current_value: float = self.q_values[state, action]
                    next_state: int = np.argmax(self.transitions[state, action])
                    if self.terminal[state, action]:
                        self.q_values[state, action] = self.rewards[state, action]
                    else:
                        self.q_values[state, action] = self.rewards[state, action] + self.gamma * np.dot(self.q_values[next_state], self.policy[next_state])
                    state_error += np.abs(current_value - self.q_values[state, action])
                delta = max(delta, state_error)
                iteration_error += state_error
            evaluation_errors.append(iteration_error)

            if n_iter is not None:
                counter += 1
                if counter >= n_iter:
                    return self.state_values, np.array(evaluation_errors)  
        return self.q_values, evaluation_errors
