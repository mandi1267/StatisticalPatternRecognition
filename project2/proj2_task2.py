# Amanda Adkins
# COMP 136
# Project 2
# File for classes needed by task 2 of the assignment

from proj2_common import *

"""
Class that selects the best lambda value for linear regression using cross
validation.
"""
class CrossValidator:

    """
    Create the cross validator.

    :param base_training_set: Training set to use cross validation to select
    lambda on.
    :param num_folds: Number of folds to use during cross validation.
    """
    def __init__(self, base_training_set, num_folds):
        self.partitioned_inputs = np.array_split(base_training_set.inputs,
            num_folds)
        self.partitioned_outputs = np.array_split(base_training_set.outputs,
            num_folds)
        self.num_folds = num_folds

    """
    Generate the training set and test set to use on one iteration of cross
    validation. Takes the partitioned inputs and outputs and uses the partition
    of the set at index=fold_index and uses that as the test set and combines
    the other partitions and uses that as the training set. Assumes fold_index
    is between (0, num_folds - 1), inclusive.

    :param fold_index: Number of the fold to use as the test set.

    :returns: Tuple with entry 1 as the training set and entry 2 as the test set
    """
    def generateTrainingAndTestSetForFold(self, fold_index):
        test_set = DataSet(self.partitioned_inputs[fold_index],
            self.partitioned_outputs[fold_index])

        training_set_fold_indices = [training_set_fold_index for
            training_set_fold_index in range(self.num_folds) if
            training_set_fold_index != fold_index]
        training_set_inputs_list = [self.partitioned_inputs[training_fold_index]
            for training_fold_index in training_set_fold_indices]
        training_set_outputs_list = [self.partitioned_outputs[
            training_fold_index] for training_fold_index in training_set_fold_indices]
        training_set = DataSet(np.concatenate(training_set_inputs_list),
            np.concatenate(training_set_outputs_list))

        return (training_set, test_set)

    """
    Select the best lambda out of the set of values to try using cross validation.

    :param lambda_values: Lambda values to train on and evaluate.

    :returns: Lambda value that performed best on the test folds when evaluated
    using cross validation
    """
    def selectBestLambda(self, lambda_values):
        # Dictionary of lambda value to list of mean square errors on each
        # fold of the training set
        # The list will be averaged after fully populated to find the lambda
        # resulting in the smallest average mean square error
        test_fold_mse_results_per_lambda = {lambda_val:[] for lambda_val in lambda_values}

        # For each fold in the training set, find the mean square error for
        # each value of lambda on the portion of data used for testing
        for i in range(self.num_folds):
            (training_set_for_fold_i, test_set_for_fold_i) = self.generateTrainingAndTestSetForFold(i)
            for lambda_val in lambda_values:
                w_vec = training_set_for_fold_i.calculateW(lambda_val)
                test_fold_mse_results_per_lambda[lambda_val].append(
                    test_set_for_fold_i.meanSquareError(w_vec))

        # Average the mean square error on each fold and select the lambda value
        # with the smallest average mean square error

        # Dictionary with key as lambda value and value as average mean square
        # error
        average_test_set_mse = {}
        for lambda_val in lambda_values:
            test_set_mses = test_fold_mse_results_per_lambda[lambda_val]
            average_test_set_mse[lambda_val] = np.mean(test_set_mses)
        best_lambda = min(average_test_set_mse, key=average_test_set_mse.get)
        return best_lambda

"""
Model selector implementation that uses the cross validation to select the
value of lambda to use, as described in task 2 of the assignment.
"""
class Task2ModelSelector(ModelSelector):

    """
    Create the model selector for task 2.

    :param lambda_values: Lambda values to evaluate training and test set mean
        square error for.
    """
    def __init__(self, lambda_values):
        ModelSelector.__init__(self, "Lambda", "Task 2")
        self.lambda_values = lambda_values

    """
    Method for getting model selection results using this model selector
    on a training and test set. Choses the hyperparameter lambda by finding the
    lambda that minimizes the average mean square error when using 10-fold
    cross validation.

    :param training_set: Training set to use to train on and select
        hyperparameters using.
    :param test_set: Test set to evaluate the model selector on.

    :returns: Model selection results.
    """
    def getModelSelectionResults(self, training_set, test_set):
        model_selection_start_time = time.time()

        cross_validator = CrossValidator(training_set, 10)
        best_lambda = cross_validator.selectBestLambda(self.lambda_values)
        w_for_best_lambda = training_set.calculateW(best_lambda)
        training_set_mse = training_set.meanSquareError(w_for_best_lambda)
        test_set_mse = test_set.meanSquareError(w_for_best_lambda)

        model_selection_end_time = time.time()

        return ModelSelectionResults(training_set_mse, test_set_mse, best_lambda,
            (model_selection_end_time - model_selection_start_time))
