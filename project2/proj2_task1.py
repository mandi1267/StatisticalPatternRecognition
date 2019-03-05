# Amanda Adkins
# COMP 136
# Project 2
# File for classes needed by task 1 of the assignment

from proj2_common import *
import matplotlib.pyplot as plt
from math import inf

"""
Contains data to plot.
"""
class PlottableData:

    """
    Create the plottable data.

    :param data_label: Label for the data
    :param x_values: X values for the data
    :param y_values: Y values for the data
    :param color: Color to use to plot the data
    :param line_style: Style of the line connecting the data points
    :param marker: Marker to use for the data points
    """
    def __init__(self, data_label, x_values, y_values, color, line_style, marker):
        self.data_label = data_label
        self.x_values = x_values
        self.y_values = y_values
        self.color = color
        self.line_style = line_style
        self.marker = marker

"""
Plot data in a single figure. All data sets should use the same y axis. A
legend, title, and axis labels will be added.

:param chart_title: title to display on the figure
:param x_axis_label: label to use for the x axis
:param y_axis_label: label to use for the y axis
:param data_sets: list of PlottableData objects to display on the chart
"""
def plotData(chart_title, x_axis_label, y_axis_label, data_sets):
    for data_set in data_sets:
        plt.plot(data_set.x_values, data_set.y_values, color=data_set.color,
            marker=data_set.marker, linestyle=data_set.line_style,
            label=data_set.data_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(chart_title)
    plt.legend()
    plt.show()

"""
Model selector implementation that uses the regularized least squares as the
model selection approach as described in task 1 of the assignment.
"""
class Task1ModelSelector(ModelSelector):

    """
    Create the model selector for task 1.

    :param lambda_values: Lambda values to evaluate training and test set mean
        square error for.
    :param should_generate_plots: True if plots of the training and test set
        mean square error at each of the evaluated lambda values should be
        displayed
    """
    def __init__(self, lambda_values, should_generate_plots):
        ModelSelector.__init__(self, "Lambda", "Task 1")
        self.lambda_values = lambda_values
        self.should_generate_plots = should_generate_plots

    """
    Plot the training and test set mean square error as a function of the lambda
    value used to generate that error.

    For the training and test set mean square errors, the value at index i in
    the mean square error list corresponds to the value at index i in the lambda
    values list.

    :param data_set_name: Name of the data set that the error is for
    :param training_set_mses: Training set mean square error for each value of lambda.
    :param test_set_mses: Test set mean square error for each value of lambda.
    """
    def plotTrainAndTestMses(self, data_set_name, training_set_mses, test_set_mses):
        plottable_data = []
        x_axis_label = "lambda value"
        y_axis_label = "mean square error"

        train_marker = 'o'
        test_marker = 's'
        plottable_data_entries = [
            PlottableData("train-" + data_set_name, self.lambda_values,
                training_set_mses, 'b', "-", train_marker),
            PlottableData("test-" + data_set_name, self.lambda_values,
                test_set_mses, 'r', "-", test_marker)
        ]
        plotData("Mean square error for varying lambdas for " + data_set_name +
            " data set", x_axis_label, y_axis_label, plottable_data_entries)


    """
    Method for getting model selection results using this model selector
    on a training and test set. Choses the hyperparameter lambda by finding the
    lambda that gives the smallest test set mean square error.

    :param training_set: Training set to use to train on and select
        hyperparameters using.
    :param test_set: Test set to evaluate the model selector on.

    :returns: Model selection results.
    """
    def getModelSelectionResults(self, training_set, test_set):
        model_selection_start_time = time.time()

        # Mean square error for each value of lambda on the training and
        # test sets, respectively
        training_set_mses = []
        test_set_mses = []

        # Initialize the minimum test set mean square error and lambda and
        # training set mean square error that correspond to it
        min_test_set_mse = inf
        best_lambda = self.lambda_values[0]
        training_set_mse_for_best_lambda = inf

        # Iterate through all values of lambda and calculate the value of w
        # and the mean square error for the training and test set
        # If the test set mean square error is smaller than any previously,
        # update the min test set mean square error and the lambda and training
        # set mean square error that correspond to it
        for lambda_val in self.lambda_values:
            w_vec = training_set.calculateW(lambda_val)
            training_set_mse = training_set.meanSquareError(w_vec)
            training_set_mses.append(training_set_mse)
            test_set_mse = test_set.meanSquareError(w_vec)
            test_set_mses.append(test_set_mse)

            if (test_set_mse < min_test_set_mse):
                min_test_set_mse = test_set_mse
                best_lambda = lambda_val
                training_set_mse_for_best_lambda = training_set_mse

        model_selection_end_time = time.time()

        if (self.should_generate_plots):
            self.plotTrainAndTestMses(training_set.data_set_name,
                training_set_mses, test_set_mses)

        return ModelSelectionResults(training_set_mse_for_best_lambda,
            min_test_set_mse, best_lambda, (model_selection_end_time -
            model_selection_start_time))
