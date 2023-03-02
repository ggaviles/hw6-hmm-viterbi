import numpy as np
class HiddenMarkovModel:
    """An implementation of a Hidden Markov Model."""

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """Initializes the HiddenMarkovModel object.

        Args:
            observation_states (np.ndarray): A NumPy array representing the observation states.
            hidden_states (np.ndarray): A NumPy array representing the hidden states.
            prior_probabilities (np.ndarray): A NumPy array representing the prior probabilities.
            transition_probabilities (np.ndarray): A NumPy array representing the transition probabilities.
            emission_probabilities (np.ndarray): A NumPy array representing the emission probabilities.
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}

        self.prior_probabilities = prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities