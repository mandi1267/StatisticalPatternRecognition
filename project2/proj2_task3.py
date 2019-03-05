# Amanda Adkins
# COMP 136
# Project 2
# File for classes needed by task 3 of the assignment

from proj2_common import *
from numpy.linalg import inv
from numpy.linalg import eigh

"""
Get the eigenvalues for the given matrix. Matrix is assumed to be a real
symmetric matrix.

:param in_matrix: Matrix to get eigenvalues for.

:returns: eigenvalues for the given matrix
"""
def getEigenValues(in_matrix):
    return eigh(in_matrix)[0]

"""
Class for computing alpha and beta hyperparameters that maximize the evidence
function, given a training set.
"""
class EvidenceFunctionMaximizer:

    """
    Initialize the evidence function maximizer.

    :param training_set: training set to use
    :param initial_alpha: initial value of alpha to use
    :param initial_beta: initial value of beta to use
    :param convergence_threshold: Updates of alpha and beta must by less than
        this amount in a single update to be considered "converged"
    """
    def __init__(self, training_set, initial_alpha, initial_beta, convergence_threshold):
        self.inputs = training_set.inputs
        self.outputs = training_set.outputs
        self.inputs_transpose_times_inputs = np.transpose(self.inputs) @ self.inputs
        self.eigenvalues_without_beta_factor = getEigenValues(self.inputs_transpose_times_inputs)
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.convergence_threshold = convergence_threshold

    """
    Determine if the updated alpha and beta values are close enough to the
    current alpha and beta values to be considered "converged"

    :param new_alpha: Updated alpha value
    :param new_eta: Updated beta value

    :returns True if the current values and the updated values are within the
    convergence threshold
    """
    def areValuesConverged(self, new_alpha, new_beta):
        return (abs(self.alpha - new_alpha) < self.convergence_threshold) and (
            abs(self.beta - new_beta) < self.convergence_threshold)

    """
    Compute the value for gamma using the class variables.

    Gamma = sum over all eigenvalues of ((beta*eigenvalue) / (alpha + beta*eigenvalue).

    :returns: gamma value
    """
    def computeGamma(self):
        gamma = 0
        for eigenvalue in np.nditer(self.eigenvalues_without_beta_factor):
            lambda_i = self.beta * eigenvalue
            gamma += (lambda_i / (self.alpha + lambda_i))
        return gamma

    """
    Compute the posterior mean for w after an update of alpha and beta.

    :returns: Posterior mean for w after an update of alpha and beta
    """
    def computeM_n(self):
        s_n = inv((self.alpha * np.identity(self.inputs.shape[1])) + (self.beta * self.inputs_transpose_times_inputs))
        return self.beta * (s_n @ (np.transpose(self.inputs) @ self.outputs))

    """
    Perform one update of alpha and beta.

    :returns: Tuple with updated alpha and beta values after one update iteration.
    """
    def performAlphaBetaConvergenceIteration(self):
        gamma = self.computeGamma()
        m_n = self.computeM_n()

        new_alpha = gamma / (np.transpose(m_n) @ m_n)
        vector_to_sum = self.outputs - (self.inputs @ m_n)
        inv_beta = (1 / (self.inputs.shape[0] - gamma)) * (
            np.transpose(vector_to_sum) @ vector_to_sum)

        return (new_alpha, 1/inv_beta)

    """
    Update alpha and beta until the values converge.
    """
    def updateAlphaAndBetaUntilConvergence(self):
        values_converged = False
        while (not values_converged):
            (new_alpha, new_beta) = self.performAlphaBetaConvergenceIteration()
            values_converged = self.areValuesConverged(new_alpha, new_beta)
            self.alpha = new_alpha
            self.beta = new_beta

"""
Class for chosing the values of alpha and beta that maximize the probability of
the observed data.
"""
class Task3ModelSelector(ModelSelector):
    """
    Create the model selector for task 3.

    :param initial_alpha: Value to use for alpha initially while doing
        iterative updates to find the optimal value for alpha.
    :param initial_beta: Value to use for beta initially while doing
        iterative updates to find the optimal value for beta.
    :param convergence_threshold: Updates of alpha and beta must by less than
        this amount in a single update to be considered "converged"
    """
    def __init__(self, initial_alpha, initial_beta, convergence_threshold):
        ModelSelector.__init__(self, "(Alpha, Beta)", "Task 3")
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.convergence_threshold = convergence_threshold

    """
    Method for getting model selection results using this model selector
    on a training and test set. Choses the hyperparameters alpha and beta by
    iteratively updating them until convergence.

    :param training_set: Training set to use to train on and select
        hyperparameters using.
    :param test_set: Test set to evaluate the model selector on.

    :returns: Model selection results.
    """
    def getModelSelectionResults(self, training_set, test_set):
        model_selection_start_time = time.time()

        evidence_function_maximizer = EvidenceFunctionMaximizer(training_set,
            self.initial_alpha, self.initial_beta, self.convergence_threshold)
        evidence_function_maximizer.updateAlphaAndBetaUntilConvergence()
        alpha = evidence_function_maximizer.alpha
        beta = evidence_function_maximizer.beta
        w_vec = evidence_function_maximizer.computeM_n()

        test_set_mse = test_set.meanSquareError(w_vec)
        training_set_mse = training_set.meanSquareError(w_vec)

        model_selection_end_time = time.time()

        return ModelSelectionResults(training_set_mse, test_set_mse, (alpha, beta),
            (model_selection_end_time - model_selection_start_time))
