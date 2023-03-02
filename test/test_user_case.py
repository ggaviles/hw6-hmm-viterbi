"""
UCSF BMI203: Biocomputing Algorithms
Author: Giovanni Aviles
Date: 3/1/23
Program: hw6-hmm-viterbi
Description: Testing Viterbi Algorithm Implementation using three different test cases
"""
import pytest
import pathlib
import numpy as np
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm

def test_user_case_one():
    """Test if Viterbi algorithm implementation returns best hidden state sequence
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['motivated','burned-out']  # A graduate student's mental health observed after a rotation

    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21']  # The NIH funding source of the graduate student's rotation project

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
    use_case_one_data_file = data_dir / "UserCase-One.npz"
    use_case_one_data = np.load(use_case_one_data_file)

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_one_data['prior_probabilities'],  # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_one_data['transition_probabilities'],  # transition_probabilities[:,hidden_states[i]]
                                         use_case_one_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    #assert use_case_one_hmm.hidden_states.ndim == use_case_one_viterbi.best_hidden_state_sequence()
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """Test if Viterbi algorithm implementation returns a best hidden state sequence with an accuracy of at least 50%
    """
    # index annotation observation_states=[i,j]
    observation_states = ['motivated', 'burned-out']  # A graduate student's mental health observed after a rotation

    # index annotation hidden_states=[i,j]
    hidden_states = ['R01', 'R21']  # The NIH funding source of the graduate student's rotation project

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
    use_case_one_data_file = data_dir / "UserCase-One.npz"
    use_case_one_data = np.load(use_case_one_data_file)

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_one_data['prior_probabilities'],  # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_one_data['transition_probabilities'],  # transition_probabilities[:,hidden_states[i]]
                                         use_case_one_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

    # Decode the hidden states (i.e., CRE selection strategy) for the progenitor CMs and evaluate the model performace
    evaluate_viterbi_decoder_using_observation_states_of_use_case_one = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_one_data['observation_states'])

    # Evaluate the accuracy of using the progenitor cardiomyocyte HMM and Viterbi algorithm to decode the progenitor CM's CRE selection strategies
    # NOTE: Model is expected to perform with 50% accuracy
    assert np.sum(use_case_one_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_use_case_one) / len(
        use_case_one_data['observation_states']) == 0.5

def test_user_case_three():
    """Toy example: Test if Viterbi Algorithm implementation returns the best hidden state sequence
    """
    np.savez('../data/Health.npz',
             prior_probabilities=np.array([0.6, 0.4]),
             transition_probabilities=np.array([[0.7, 0.3],
                                                [0.4, 0.6]]),
             emission_probabilities=np.array([[0.5, 0.4, 0.1],
                                              [0.1, 0.3, 0.6]]),
             observation_states=np.array(
                 ['normal', 'cold', 'dizzy']),
             hidden_states=np.array(['Healthy', 'Healthy', 'Ill']))

    data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
    health_data_file = data_dir / "Health.npz"
    health_data = np.load(health_data_file)

    # index annotation observation_states=[i,j]
    observation_states = ['normal', 'cold', 'dizzy']

    # index annotation hidden_states=[i,j]
    hidden_states = ['Healthy', 'Ill']

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    health_hmm = HiddenMarkovModel(observation_states,
                                   hidden_states,
                                   health_data['prior_probabilities'],# prior probabilities of hidden states in the order specified in the hidden_states list
                                   health_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                                   health_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    health_viterbi = ViterbiAlgorithm(health_hmm)

    # Check that Viterbi algorithm implementation returns best hidden state sequence
    health_decoded_hidden_states = health_viterbi.best_hidden_state_sequence(health_data['observation_states'])
    assert np.alltrue(health_decoded_hidden_states == health_data['hidden_states'])


