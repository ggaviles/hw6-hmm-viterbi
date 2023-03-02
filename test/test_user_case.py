"""
UCSF BMI203: Biocomputing Algorithms
Author: Giovanni Aviles
Date: 3/1/23
Program: hw6-hmm-viterbi
Description:
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm

def test_user_case_one():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['motivated','burned-out']  # A graduate student's mental health observed after a rotation

    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21']  # The NIH funding source of the graduate student's rotation project

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

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
    assert
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """_summary_
    """
    # index annotation observation_states=[i,j]
    observation_states = ['regulatory',
                          'regulatory-potential']  # observed regulatory activity in the TBX5 TAD of cardiomyocytes

    # index annotation hidden_states=[i,j]
    hidden_states = ['encode-atac', 'atac']  # In order of the two cCRE selection strategies (encode_atac, atac)

    # Import the HMM input data for progenitor cardiomyocytes (prefix: prog_cm)
    prog_cm_data = np.load('../data/ProjectDeliverable-ProgenitorCMs.npz')

    # Instantiate submodule class models.HiddenMarkovModel with progenitor cardiomyocytes
    # observation and hidden states and prior, transition, and emission probabilities.
    prog_cm_hmm_object = HiddenMarkovModel(observation_states,
                                           hidden_states,
                                           prog_cm_data['prior_probabilities'],
                                           # prior probabilities of hidden states in the order specified in the hidden_states list
                                           prog_cm_data['transition_probabilities'],
                                           # transition_probabilities[:,hidden_states[i]]
                                           prog_cm_data[
                                               'emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prog_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states (i.e., CRE selection strategy) for the progenitor CMs and evaluate the model performace
    evaluate_viterbi_decoder_using_observation_states_of_prog_cm = prog_cm_viterbi_instance.best_hidden_state_sequence(
        prog_cm_data['observation_states'])

    # Evaluate the accuracy of using the progenitor cardiomyocyte HMM and Viterbi algorithm to decode the progenitor CM's CRE selection strategies
    # NOTE: Model is expected to perform with 80% accuracy
    assert np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm) / len(
        prog_cm_data['observation_states']) == 0.8

    ### Evaluate Primitive Cardiomyocyte Regulatory Observation Sequence ###
    # Import primitive cardiomyocyte data (prefix: prim_cm)
    prim_cm_data = np.load('./data/ProjectDeliverable-PrimitiveCMs.npz')

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prim_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states of the primitive cardiomyocyte's regulatory observation states
    decoded_hidden_states_for_observed_states_of_prim_cm = prim_cm_viterbi_instance.best_hidden_state_sequence(
        prim_cm_data['observation_states'])
    assert np.sum(prim_cm_data['hidden_states'] == decoded_hidden_states_for_observed_states_of_prim_cm) / len(
        prim_cm_data['observation_states']) == 0.8


def test_user_case_three():
    """_summary_
    """
    # TODO