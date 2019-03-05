# Amanda Adkins
# COMP 136
# Programming project #2: http://www.cs.tufts.edu/comp/136/HW/pp2.pdf

import numpy as np
import sys
from proj2_common import *
from proj2_task1 import *
from proj2_task2 import *
from proj2_task3 import *

"""
Read data from a CSV and store it in a numpy array.

Data is expected to be numeric. Each line in the file should contain the same
number of entries.

:param file_name: Name of the file to read CSV data from

:returns: numpy matrix or vector containing numeric data from the CSV
"""
def readFromCsv(file_name):

    # Read the data into a python list
    unconverted_array = []
    with open(file_name) as file_pointer:
        lines = file_pointer.readlines()
        for line in lines:
            entries_in_line = [float(csv_entry) for csv_entry in line.rstrip().split(",")]
            # If there is one entry in the line, we want to add the single entry
            # to the output list, since we want a numpy vector and not a 2d matrix
            # If there are multiple entries, then we want to generate a matrix,
            # so we'll add the contents of the line as a list, rather than
            # adding the entries in the list
            if (len(entries_in_line) == 1):
                unconverted_array.extend(entries_in_line)
            else:
                unconverted_array.append(entries_in_line)
    return np.array(unconverted_array)

"""
Read inputs and outputs from files.

:param data_set_prefix: Prefix to put at the beginning of the file name. Either
    "train" or "test", depending on if we're reading training or test sets.
:param data_set_name: Name of the data set to read in.

:returns: Tuple with the inputs and outputs of the data set.
"""
def readInputsAndOutputsFromFiles(data_set_prefix, data_set_name):
    inputs_file_name = data_set_prefix + "-" + data_set_name + ".csv"
    outputs_file_name = data_set_prefix + "R-" + data_set_name + ".csv"

    return (readFromCsv(inputs_file_name), readFromCsv(outputs_file_name))

"""
Create a populated training set object given a training set name. Read the
inputs and outputs for the training set from files derived from the training
set name.

:param data_set_name: Data set name

:returns: training set object for the data set
"""
def instantiateTrainingSetFromDataSetName(data_set_name):
    (inputs, outputs) = readInputsAndOutputsFromFiles("train", data_set_name)
    return TrainingSet(data_set_name, inputs, outputs, data_set_name)

"""
Create a populated test set from the given data set name. Read the inputs and
outputs for the test set from files derived from the test set name.

:param data_set_name: Data set name

:returns: test set object for the data set
"""
def instantiateTestSetFromDataSetName(data_set_name):
    (inputs, outputs) = readInputsAndOutputsFromFiles("test", data_set_name)
    return TestSet(data_set_name, inputs, outputs)

"""
Generate a new training set that is a subset of the given training set. The
subset should be the first n samples from the base training set,
where n=new_training_set_size. Assumes that the given training set has at least
new_training_set_size values in it.

:param new_training_set_size: number of samples to take from the given training
    set to use in the new training set
:param base_training_set: populated training set to take the samples from

:returns: training set that is a subset of the given training set
"""
def generateTrainingSetFromSubset(new_training_set_size, base_training_set):
    base_training_set_name_components = base_training_set.data_set_name.split("-")

    new_data_set_name = str(new_training_set_size) + ("(" +
        base_training_set_name_components[0] + ")-" +
        base_training_set_name_components[1])
    new_data_inputs = base_training_set.inputs[0:new_training_set_size]
    new_data_outputs = base_training_set.outputs[0:new_training_set_size]
    return TrainingSet(new_data_set_name, new_data_inputs, new_data_outputs,
        base_training_set.test_set_name)

"""
Read training and test data from files and populate training and test set
objects and generate additional training sets that are subsets of the 1000-100
training set.

:returns: a tuple with a training set dictionary in the first entry and a test
    set dictionary in the second entry. Each dictionary has data set names as
    the keys and the populated training/test set objects as the values.
"""
def readAndGenerateTrainingAndTestSets():
    data_set_names = ["100-10", "100-100", "1000-100", "forestfire", "realestate"]
    populated_training_sets = {data_set_name:instantiateTrainingSetFromDataSetName(
        data_set_name) for data_set_name in data_set_names}

    generated_data_set_sizes = [50, 100, 150]
    generated_training_sets = [generateTrainingSetFromSubset(new_data_set_size,
        populated_training_sets["1000-100"]) for new_data_set_size in
        generated_data_set_sizes]
    populated_training_sets.update({data_set.data_set_name:data_set for
        data_set in generated_training_sets})
    populated_test_sets = {test_set_name:instantiateTestSetFromDataSetName(
        test_set_name) for test_set_name in data_set_names}

    return (populated_training_sets, populated_test_sets)

"""
Display one line of results from a model selection approach.

:param data_set_name_entry: Entry to display in the data set name column
:param num_samples_entry: Entry to display in the num samples column
:param num_features_entry: Entry to display in the num features column
:param training_set_mse_entry: Entry to display in the training set mean square
    error column
:param test_set_mse_entry: Entry to display in the test set mean square error column
:param hyperparameters_entry: Entry to display in the hyperparameters column
:param run_time_entry: Entry to display in the run time column
"""
def displayResultsData(data_set_name_entry, num_samples_entry,
    num_features_entry, training_set_mse_entry, test_set_mse_entry,
    hyperparameters_entry, run_time_entry):
    print('%-20s%-15s%-15s%-15s%-25s%-25s%-40s' % (data_set_name_entry,
        num_samples_entry, num_features_entry, run_time_entry,
        training_set_mse_entry, test_set_mse_entry, hyperparameters_entry))

"""
Run a model selection approach (one per assignmet task) on all of the training
sets given.

:param model_selector: Provides model selection approach
:param training_sets: Training sets to select hyperparameters for. This is a
    dictionary of the data set name as the key and the training set object as
    the value.
:param test_sets: Test sets to evaluate the trained models on. This is a
    dictionary of the test set name mapped to the test set object for that test
    set name.
"""
def trainAndEvaluateModelResults(model_selector, training_sets, test_sets):
    print()
    print(model_selector.task_name + "----------------------------------------------------")
    data_set_results = {}

    # For each training set, find the corresponding test set, and then select
    # the hyperparameters and return the optimal hyperparameters and other
    # statistics about the model selection
    for training_set in training_sets.values():
        test_set = test_sets[training_set.test_set_name]
        data_set_results[training_set.data_set_name] = model_selector.getModelSelectionResults(training_set, test_set)

    # Display the results of the model selection approach for each training set
    displayResultsData("Training Set", "Num Samples", "Num Features",
        "Train Set MSE", "Test Set MSE",
        "Best " + model_selector.hyperparameters_label, "Run Time")
    for training_set in training_sets.values():
        training_set_dimensions = training_set.inputs.shape
        results = data_set_results[training_set.data_set_name]
        displayResultsData(training_set.data_set_name,
            training_set_dimensions[0], training_set_dimensions[1],
            results.training_set_mse, results.test_set_mse, results.hyperparameters, round(results.run_time, 5))

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=200)

    # If an argument is specified and equals "f" or "F", then skip the plots
    should_generate_plots = True
    if (len(sys.argv) > 1):
        if (sys.argv[1].lower() == 'f'):
            should_generate_plots = False

    # Task 0: Read in the training sets and generate the 3 training sets that are subsets of the "1000-100" training set
    (training_sets, test_sets) = readAndGenerateTrainingAndTestSets()

    # Configure the parameters needed for various tasks
    lambda_values = range(0, 150 + 1)
    initial_alpha = 100
    initial_beta = 100
    convergence_threshold = 0.00001

    # Set up each of the task runners
    model_selectors = [
        Task1ModelSelector(lambda_values, should_generate_plots),
        Task2ModelSelector(lambda_values),
        Task3ModelSelector(initial_alpha, initial_beta, convergence_threshold)
    ]

    # Run each task on all of the training sets
    for model_selector in model_selectors:
        trainAndEvaluateModelResults(model_selector, training_sets, test_sets)

    print("\n")
