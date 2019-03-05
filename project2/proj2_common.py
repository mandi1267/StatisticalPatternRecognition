# Amanda Adkins
# COMP 136
# Project 2
# File for classes needed by all tasks

from numpy.linalg import inv
import numpy as np
import time

"""
Class that contains model selection results for a particular training set and
test set for a model selection approach.
"""
class ModelSelectionResults:

    """
    Create the model selection results.

    :param training_set_mse: Mean square error on the training set for the given
        hyperparameters
    :param test_set_mse: Mean square error on the test set for the given
        hyperparameters
    :param hyperparameters: Values for hyperparameters chosen based on the
        particular model selection approach. May be a single value (lambda) or
        a tuple (alpha, beta)
    :param run_time: Time that it took to select the hyperparameters and
        calculate the mean square errors on the training and test sets.
    """
    def __init__(self, training_set_mse, test_set_mse, hyperparameters, run_time):
        self.training_set_mse = training_set_mse
        self.test_set_mse = test_set_mse
        self.hyperparameters = hyperparameters
        self.run_time = run_time

"""
Abstract class that provides the interface and common variables needed for
model selection evalation.
"""
class ModelSelector:

    """
    Create the model selector.

    :param hyperparameters_label: Label for the hyperparameters used in this
        model selection approach
    :param task_name: Name to display when displaying results for this model
        selector.
    """
    def __init__(self, hyperparameters_label, task_name):
        self.hyperparameters_label = hyperparameters_label
        self.task_name = task_name

    """
    Abstract method for getting model selection results using this model selector
    on a training and test set.

    :param training_set: Training set to use to train on and select
        hyperparameters using.
    :param test_set: Test set to evaluate the model selector on.

    :returns: Model selection results.
    """
    def getModelSelectionResults(self, training_set, test_set):
        pass

"""
Class representing any data set with inputs and outputs.
"""
class DataSet:

    """
    Create a data set given inputs and outputs

    :param inputs: Numpy matrix of inputs with each row as a single sample
    :param outputs: Numpy array of outputs with the entry at place i
        corresponding to the sampel at row i in the pinputs
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    """
    Use lambda value to calculate the vector w that minimizes the regularized
    sum of squares error for this data set.

    :param lambda_val: regularization parameter to use in minimization

    :returns: vector w that minimizes the regularized sum of squares error for
    this data set
    """
    def calculateW(self, lambda_val):
        inputs_transpose = np.transpose(self.inputs)
        lambda_mat = lambda_val * np.identity(inputs_transpose.shape[0])
        return ((inv(lambda_mat + (inputs_transpose @ self.inputs))
            @ inputs_transpose) @ self.outputs)

    """
    Calculate the mean square error for this data set and the weight vector w.

    This assumes that the data set's outputs and w are 1D arrays (vectors) and
    that the data set's inputs have the same number of rows as outputs has
    entries and the same number of columns as w has entries.

    :param w: numpy array representing the weights to apply to the inputs

    :returns: the mean square error
    """
    def meanSquareError(self, w):
        x_y_difference = self.inputs @ w - self.outputs
        squared_error = x_y_difference @ x_y_difference
        return squared_error / self.outputs.size

"""
Class for holding a training set. Also contains the name of the test set on
which to evaluate the model trained using this training set.
"""
class TrainingSet(DataSet):

    """
    Create the training set.

    :param data_set_name: Name of the data set
    :param inputs: Inputs of the training set
    :param outputs: Outputs of the training set
    :param test_set_name: Name of the test set on which to evaluate a model
        trained using this training set.
    """
    def __init__(self, data_set_name, inputs, outputs, test_set_name):
        DataSet.__init__(self, inputs, outputs)
        self.data_set_name = data_set_name
        self.test_set_name = test_set_name

"""
Class representing a test data set that will be used to evaluate trained models.
"""
class TestSet(DataSet):

    """
    Create the test set.

    :param test_set_name: Name of the test set
    :param inputs: Inputs of the test set
    :param outputs: Outputs of the test set
    """
    def __init__(self, test_set_name, inputs, outputs):
        DataSet.__init__(self, inputs, outputs)
        self.test_set_name = test_set_name
