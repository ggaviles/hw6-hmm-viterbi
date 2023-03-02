import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0, :] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale
        first_obs_index = self.hmm_object.observation_states_dict[decode_observation_states[0]]
        delta = np.multiply(self.hmm_object.prior_probabilities,
                            self.hmm_object.emission_probabilities[:, first_obs_index])
        if np.sum(delta) == 0:
            delta = np.full_like(delta, np.finfo(float).eps)
        # delta = delta / np.sum(delta)  # Scale

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            # Compute the product of delta and transition and emission probabilities
            product_of_delta_and_transition_emission = np.multiply(delta, self.hmm_object.transition_probabilities.T)
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission,
                                                                   self.hmm_object.emission_probabilities[:,
                                                                   self.hmm_object.observation_states_dict[
                                                                       decode_observation_states[trellis_node]]])

            # TODO: comment the initialization, recursion, and termination steps

            # product_of_delta_and_transition_emission = np.multiply(delta*self.hmm_object.transition_probabilities)


            # Update delta and scale
            delta = np.max(product_of_delta_and_transition_emission, axis=1)
            if np.sum(delta) == 0:
                delta = np.full_like(delta, np.finfo(float).eps)

            # Select the hidden state sequence with the maximum probability
            best_path[trellis_node, :] = np.argmax(product_of_delta_and_transition_emission, axis=1)

            # Update best path
            for hidden_state in range(len(self.hmm_object.hidden_states)):
                best_hidden_state_index = int(best_path[trellis_node, hidden_state])
                path[trellis_node, hidden_state] = best_hidden_state_index
                best_path[trellis_node, hidden_state] = np.multiply(delta[best_hidden_state_index],
                                                                    product_of_delta_and_transition_emission[
                                                                        best_hidden_state_index, hidden_state])

                """
                #temp_product = np.multiply(self.hmm_object.transition_probabilities[:, hidden_state], delta[:, trellis_node - 1])
                #delta[hidden_state, trellis_node] = np.max(temp_product) * self.hmm_object.emission_probabilities[hidden_state, path[trellis_node]]
                #best_path[hidden_state, trellis_node - 1] = np.argmax(temp_product)
                
                k = np.argmax(k in decode_observation_states[k, trellis_node - 1]
                              * self.hmm_object.transition_probabilities[k, hidden_state]
                              * self.hmm_object.emission_probabilities[hidden_state, trellis_node])
                decode_observation_states[hidden_state, trellis_node] = decode_observation_states[k, trellis_node] \
                                                                        * self.hmm_object.transition_probabilities[k, hidden_state] \
                                                                        * self.hmm_object.emission_probabilities[hidden_state, trellis_node]
                                                                        
                """

            # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path
            #path = best_path.copy()

        # Select the last hidden state, given the best path (i.e., maximum probability)
        best_hidden_state_path = np.zeros(len(decode_observation_states), dtype='U11')
        best_hidden_state_path[-1] = self.hmm_object.hidden_states[int(np.argmax(delta))]

        for trellis_node in range(len(decode_observation_states) - 2, -1, -1):
            best_hidden_state_index = path[trellis_node + 1, int(best_path[trellis_node + 1, best_hidden_state_index])]
            best_hidden_state_index = int(
                best_path[trellis_node, self.hmm_object.hidden_states.index(best_hidden_state_path[trellis_node + 1])])
            best_hidden_state_path[trellis_node] = self.hmm_object.hidden_states[best_hidden_state_index]
        """
        best_hidden_state_path = np.zeros(len(decode_observation_states))
        best_hidden_state_path[-1] = np.argmax(path[:, -1])
        for n in range(len(decode_observation_states)-2, -1, -1):
            best_hidden_state_path[n] = best_path[(best_hidden_state_path[n+1], n)]

        best_hidden_state_path = np.array([best_hidden_state_path])
        """

        return best_hidden_state_path