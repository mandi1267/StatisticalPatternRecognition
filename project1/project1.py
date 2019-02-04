# Amanda Adkins
# COMP 136
# Programming project #1: http://www.cs.tufts.edu/comp/136/HW/pp1.pdf
import sys

from math import log
from math import exp
from math import factorial

import matplotlib.pyplot as plt

"""
Read all of the strings from the given file and store them in order into a
string_list.

:param file_name: Name of the file to read strings from

:returns: A list of the strings in the file, in the same order that they
occurred in the file
"""
def readStringsFromFile(file_name):
    strings_to_return = []
    with open(file_name) as file_pointer:
        lines = file_pointer.readlines()
        for line in lines:
            strings_to_return += line.split()
    return strings_to_return

"""
Build the vocabulary from a list of strings. Returns a dictionary where the
keys are the distinct words in the vocabulary and the value is the value k that
will be used to refer to that particular word. The values k should range from 0
 to K-1, inclusive, where K is the total number of words in the vocabulary.

:param string_list: list of all strings

:returns: Dictionary where the keys are the distinct words in the vocabulary and
the value is the value k that will be used to refer to that word (ex. m_k).
"""
def buildVocabulary(string_list):
    vocabulary = {}
    next_k = 0

    # Go through all strings and assign any string not yet seen the next
    # value for k in the vocabulary dictionary
    for string_entry in string_list:
        if string_entry not in vocabulary:
            vocabulary[string_entry] = next_k
            next_k += 1

    return vocabulary

"""
Transform a data set of words into a data set where the ith entry is the
index k (from the vocabulary) of the ith word in the data set. For example, if
the first word in the data set was 'sample' and 'sample' is the 57th word in
the vocabulary, the first value in the returned list would be 57.

:param vocabulary_dict: Dictionary representing the vocabulary where the key is
the vocabulary word and the value is the index k of the word in the vocabulary
:param string_data_set: data set composed of strings to transform to use the
index representation

:returns: Data set where the ith entry is the index k (from the vocabulary) of
the ith word in the data set.
"""
def transformStringDataSetToSetOfVocabIndices(vocabulary_dict, string_data_set):
    data_set_by_indices = []
    for word in string_data_set:
        # Find the index k corresponding to the word from the vocabulary and
        # add it to the transformed data set
        data_set_by_indices.append(vocabulary_dict[word])

    return data_set_by_indices

"""
For all words in the vocabulary, generate a list where an entry at index k in
in the list is the number of occurences in the data set of the word at
index k in the vocabulary.

:param k_words_in_vocab: number of distinct words in the vocabulary
:param data_set_by_indices: List representing the data set of words, where
instead of the entry being a word, it is the index of the word within the
vocabulary

:returns: list of the occurences of vocabulary words in the training set where
the value at index k in the returned list represents the number of occurrences
in the data set of the word at index k in the vocabulary
"""
def getCountOfWordsInDataSet(k_words_in_vocab, data_set_by_indices):

    # Initialize the dictionary to have a count 0 for all entries
    count_list = [0 for i in range(k_words_in_vocab)]

    for vocab_word_index in data_set_by_indices:
        count_list[vocab_word_index] += 1

    return count_list

"""
Train the model using the maximum likelihood distribution. Computes a list
representing mu where the value at index k in the vocabulary corresponds to the
probability of the word assigned label k in the vocabulary based on the maximum
likelihood distibution.

:param count_vocab_words_in_train_set: list where the index is the index k of
the vocab word in the vocabulary and the value in the list is the number of
occurrences of the word in the training set (m_k)
:param n_words_in_training_set: number of words in the training set

:returns: list representing mu where the value at index k in the vocabulary
corresponds to the probability of the word assigned label k in the vocabulary
based on the maximum likelihood distribution
"""
def trainModelUsingMaximumLikelihoodDistribution(count_vocab_words_in_train_set,
    n_words_in_training_set):

    # Compute the following for all values of k
    # Maximum likelihood distribution: p(y* = w_k) = m_k/N
    maximum_likelihood_probabilities = []
    for m_k in count_vocab_words_in_train_set:
        maximum_likelihood_probabilities.append(float(m_k)
            / n_words_in_training_set)

    return maximum_likelihood_probabilities

"""
Train the model using the MAP distribution. Computes a list representing mu
where the value at index k in the vocabulary corresponds to the probability of
the word assigned label k in the vocabulary based on the MAP distribution.

:param count_vocab_words_in_train_set: list where the index is the index k of
the vocab word in the vocabulary and the value in the list is the number of
occurrences of the word in the training set (m_k)
:param n_words_in_training_set: number of words in the training set
:param k_words_in_vocab: number of words in the vocabulary
:param alpha_prime: scalar value with which to multiply a vector of all ones to
get alpha

:returns: list representing mu where the value at index k in the vocabulary
corresponds to the probability of the word assigned label k in the vocabulary
based on the MAP distribution
"""
def trainModelUsingMAPDistribution(count_vocab_words_in_train_set,
    n_words_in_training_set, k_words_in_vocab, alpha_prime):

    # Compute the following for all values of k
    # MAP distribution: p(y* = w_k) = (m_k + alpha_k - 1)/(N + alpha_0 - K)
    # Note that here alpha_k is alpha_prime for every value of k
    alpha_0 = alpha_prime * k_words_in_vocab
    map_probabilities = []
    for m_k in count_vocab_words_in_train_set:
        map_probabilities.append((float(m_k) + alpha_prime - 1) / (
            n_words_in_training_set + alpha_0 - k_words_in_vocab))

    return map_probabilities

"""
Train the model using the predictive distribution. Computes a list representing
mu where the value at index k in the vocabulary corresponds to the probability
of the word assigned label k in the vocabulary based on the predictive
distribution.

:param count_vocab_words_in_train_set: list where the index is the index k of
the vocab word in the vocabulary and the value in the list is the number of
occurrences of the word in the training set (m_k)
:param n_words_in_training_set: number of words in the training set
:param k_words_in_vocab: number of words in the vocabulary
:param alpha_prime: scalar value with which to multiply a vector of all ones to
get alpha

:returns: list representing mu where the value at index k in the vocabulary
corresponds to the probability of the word assigned label k in the vocabulary
based on the predictive distribution
"""
def trainModelUsingPredictiveDistribution(count_vocab_words_in_train_set,
    n_words_in_training_set, k_words_in_vocab, alpha_prime):

    # Compute the following for all values of k
    # Predictive distribution: p(y* = w_k) = (m_k + alpha_k)/(N + alpha_0)
    # Note that here alpha_k is alpha for every value of k
    alpha_0 = alpha_prime * k_words_in_vocab
    predictive_probabilities = []
    for m_k in count_vocab_words_in_train_set:
        predictive_probabilities.append((float(m_k) + alpha_prime) / (
            n_words_in_training_set + alpha_0))

    return predictive_probabilities

"""
Calculate the perplexity of the given model on the given data set.

:param probability_distribution_model: list representing the trained probability
model where the probability of the word at index k in the vocabulary is given
by the value at index k in the list
:param data_set_by_indices: representation of the data set to calculate the
perplexity for, where the instead of containing the word in the data set, the
list contains the indices of the words from the vocabulary. For example, if
the first word in the data set was 'sample' and 'sample' is the 57th word in
the vocabulary, the first value in this list would be 57.

:returns: the perplexity
"""
def calculatePerplexity(probability_distribution_model, data_set_by_indices):
    n_words = len(data_set_by_indices)

    # This will be the sum from i = 1 to N of ln(p(y_i))
    log_probability_sum = 0.0
    for data_set_entry_vocab_index in data_set_by_indices:

        # find p(y_i)
        data_set_entry_probability = float(probability_distribution_model[
            data_set_entry_vocab_index])

        # Add ln(p(y_i)) to the log probability sum
        if (data_set_entry_probability != 0):
            log_probability_sum += log(data_set_entry_probability)
        else:
            log_probability_sum += float("-inf")

    return exp(-1 * log_probability_sum / n_words)

"""
Print the preplexity results from Task 1. Assumes that data_values has 5
entries.

:param data_label: Label for the data to print
:param data_values: Data to print. Assumed to have 5 entries.
"""
def printTask1PerplexityData(data_label, data_values):
    print('%-50s%-20s%-20s%-20s%-20s%-20s' % (data_label, data_values[0],
        data_values[1], data_values[2], data_values[3], data_values[4]))

"""
Print the results from Task 2. Assumes that data_values has 10
entries.

:param data_label: Label for the data to print
:param data_values: Data to print. Assumed to have 10 entries.
"""
def printTask2VaryingAlphaResults(data_label, data_values):
    print('%-25s%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s' % (
        data_label, data_values[0], data_values[1], data_values[2],
        data_values[3], data_values[4], data_values[5], data_values[6],
        data_values[7], data_values[8], data_values[9]))

"""
Take the log of the gamma function of the given value. Requires that the value
to take a the log of the gamma of is an integer. The gamma function of the
value equals the factorial of <value> - 1. The log of this is the sum from
i = 1 to (<value> - 1) of log(i).

:param value_to_take_log_gamma_of: value to compute the log of the gamma of the
value of

:return log(gamma(value))
"""
def logGammaOfInt(value_to_take_log_gamma_of):
    sum = 0
    for i in range(value_to_take_log_gamma_of - 1):
        sum += log(float(1 + i))
    return sum

"""
Calculate the log of the evidence function.

:param count_vocab_words_in_train_set: list where the index is the index k of
the vocab word in the vocabulary and the value in the list is the number of
occurrences of the word in the training set (m_k)
:param n_words_in_training_set: number of words in the training set
:param k_words_in_vocab: number of words in the vocabulary
:param alpha_prime: scalar value with which to multiply a vector of all ones to
get alpha

:returns log of the evidence function for the given training set and alpha value
"""
def calculateLogEvidence(count_vocab_words_in_train_set,
    n_words_in_training_set, k_words_in_vocab, alpha_prime):

    logEvidenceSum = 0.0

    # When taking the log of the evidence function, we can convert all
    # products to sums and divisions to subtractions, and then take the log of
    # the interior
    alpha_0 = alpha_prime * k_words_in_vocab
    logEvidenceSum += logGammaOfInt(alpha_0)
    logEvidenceSum -= logGammaOfInt(alpha_0 + n_words_in_training_set)
    for i in range(k_words_in_vocab):
        logEvidenceSum += logGammaOfInt(
            alpha_prime + count_vocab_words_in_train_set[i])
        logEvidenceSum -= logGammaOfInt(alpha_prime)

    return logEvidenceSum

"""
Run tasks 1 and 2 from project 1.
"""
def task1and2():
    training_set_file_name = "training_data.txt"
    test_set_file_name = "test_data.txt"

    # Read in all strings in the training and test sets
    training_set_strings = readStringsFromFile(training_set_file_name)
    test_set_strings = readStringsFromFile(test_set_file_name)

    # Build a vocabulary based on the strings in both the test and training sets
    all_strings = training_set_strings + test_set_strings
    vocabulary_dict = buildVocabulary(all_strings)

    # Convert the training and test sets to refer to words by their index k
    # in the vocabulary rather than using the strings (for space efficiency +
    # ease of look up)
    training_set_by_indices = transformStringDataSetToSetOfVocabIndices(
        vocabulary_dict, training_set_strings)
    test_set_by_indices = transformStringDataSetToSetOfVocabIndices(
        vocabulary_dict, test_set_strings)

    # Get the sizes of the training set and the vocabulary
    n_training_set_size = len(training_set_by_indices)
    k_vocab_size = len(vocabulary_dict)

    # Execute tasks 1 and 2
    task1(vocabulary_dict, training_set_by_indices, test_set_by_indices,
        n_training_set_size, k_vocab_size)
    task2(training_set_by_indices, test_set_by_indices, n_training_set_size,
        k_vocab_size)

"""
Run task 1.

:param vocabulary_dict: Dictionary of all distinct strings in the vocabulary
as keys and with the values as a number in the range 0 to K - 1 inclusive, where
K is the number of distinct words in the vocabulary
:param training_set_by_indices: Representation of the full training set, where
the instead of containing the word in the training set, the list contains the
indices of the words from the vocabulary. For example, if the first word in the
data set was 'sample' and 'sample' is the 57th word in
the vocabulary, the first value in this list would be 57.
:param test_set_by_indices: Representation of the full test set, represented in
the same way (using vocab word index instead of vocab string) as the
training_set_by_indices
:param n_training_set_size: Number of words in the full training set.
:param k_vocab_size: Number of distinct words in the vocabulary
"""
def task1(vocabulary_dict, training_set_by_indices, test_set_by_indices,
    n_training_set_size, k_vocab_size):

    print("Task 1 ------------------------------------------------------------")

    # Make a list of the training set sizes to use
    training_subset_sizes_to_use = [int(n_training_set_size / 128),
        int(n_training_set_size / 64), int(n_training_set_size / 16),
        int(n_training_set_size / 4), n_training_set_size]

    # Make variables for storing the perplexities for each of the training set
    # sizes above. The value at index i will be the perplexity of the model on
    # the training set size at index i in training_subset_sizes_to_use
    training_set_perplexity_ml = []
    test_set_perplexity_ml = []

    training_set_perplexity_map = []
    test_set_perplexity_map = []

    training_set_perplexity_predictive = []
    test_set_perplexity_predictive = []

    # For each training set size, determine the count of each word in the
    # training subset, train the unigram model according to the 3 methods, and
    # then calculate the perplexity for the training subset and test set
    for training_subset_size in training_subset_sizes_to_use:

        # Get the subset of words (first training_subset_size words) from the
        # full training set
        training_subset = training_set_by_indices[:training_subset_size]

        # Find the count of each word in the vocabulary (m_k for all k) from the
        # training subset. m_k for vocab word k will be stored at index k
        # Note: k here ranges from 0 to K - 1, inclusive
        word_count = getCountOfWordsInDataSet(k_vocab_size, training_subset)

        # Train the models using the maximum likelihood, maximum a posteriori,
        # and predictive distributions
        # In the trained model, the value at index k will be the probability of
        # the vocab word assigned label k
        ml_model = trainModelUsingMaximumLikelihoodDistribution(word_count,
            training_subset_size)
        map_model = trainModelUsingMAPDistribution(word_count,
            training_subset_size, k_vocab_size, 2)
        predictive_model = trainModelUsingPredictiveDistribution(word_count,
            training_subset_size, k_vocab_size, 2)

        # Calculate the perplexity of each of the models on the training
        # subset
        training_set_perplexity_ml.append(calculatePerplexity(ml_model,
            training_subset))
        test_set_perplexity_ml.append(calculatePerplexity(ml_model,
            test_set_by_indices))

        training_set_perplexity_map.append(calculatePerplexity(map_model,
            training_subset))
        test_set_perplexity_map.append(calculatePerplexity(map_model,
            test_set_by_indices))

        training_set_perplexity_predictive.append(calculatePerplexity(
            predictive_model, training_subset))
        test_set_perplexity_predictive.append(calculatePerplexity(
            predictive_model, test_set_by_indices))

    # Print out the perplexity results
    printTask1PerplexityData("Training Set Size",
        training_subset_sizes_to_use)
    printTask1PerplexityData("Training Set Perplexity, Maximum Likelihood",
        training_set_perplexity_ml)
    printTask1PerplexityData("Training Set Perplexity, MAP",
        training_set_perplexity_map)
    printTask1PerplexityData("Training Set Perplexity, Predictive",
        training_set_perplexity_predictive)
    printTask1PerplexityData("Test Set Perplexity, Maximum Likelihood",
        test_set_perplexity_ml)
    printTask1PerplexityData("Test Set Perplexity, MAP",
        test_set_perplexity_map)
    printTask1PerplexityData("Test Set Perplexity, Predictive",
        test_set_perplexity_predictive)

    # Plot the perplexity results
    plt.plot(training_subset_sizes_to_use, training_set_perplexity_predictive,
        'rD-.', label="training set, predictive")
    plt.plot(training_subset_sizes_to_use, training_set_perplexity_map,
        'gD-.', label="training set, MAP")
    plt.plot(training_subset_sizes_to_use, training_set_perplexity_ml,
        'cD-.', label="training set, ML")
    plt.plot(training_subset_sizes_to_use, test_set_perplexity_predictive,
        'rs:', label="test set, predictive")
    plt.plot(training_subset_sizes_to_use, test_set_perplexity_map,
        'gs:', label="test set, MAP")
    plt.plot(training_subset_sizes_to_use, test_set_perplexity_ml,
        'cs:', label="test set, ML")

    plt.xlabel("Training set size")
    plt.ylabel("Perplexity")
    plt.title("Training and Test Set Perplexity by Model and Training Set Size")
    plt.legend()

    plt.show()

"""
Run task 2.

:param training_set_by_indices: Representation of the full training set, where
the instead of containing the word in the training set, the list contains the
indices of the words from the vocabulary. For example, if the first word in the
data set was 'sample' and 'sample' is the 57th word in
the vocabulary, the first value in this list would be 57.
:param test_set_by_indices: Representation of the full test set, represented in
the same way (using vocab word index instead of vocab string) as the
training_set_by_indices
:param n_training_set_size: Number of words in the full training set.
:param k_vocab_size: Number of distinct words in the vocabulary
"""
def task2(training_set_by_indices, test_set_by_indices, n_training_set_size,
    k_vocab_size):

    print("Task 2 ------------------------------------------------------------")

    # Create a list with all of the alpha_prime values to use
    alpha_prime_values = [(i + 1) for i in range(10)]

    # Initialize lists to keep track of the perplexities and log(evidence fn)
    # for each of the above values of alpha_prime
    test_set_perplexities_for_varying_alphas = []
    log_evidence_for_varying_alphas = []

    # Get the subset of the full training set to train on for this task
    task_2_training_set_size = int(n_training_set_size / 128)
    task_2_training_subset = training_set_by_indices[:task_2_training_set_size]

    # Get the count of each words in the training subset
    # This is a list of length k_vocab_size, where the value at index i is the
    # number of occurrences of the word assigned index i from the vocab in the
    # training subset (m_k for k in 0 to k_words_in_vocab - 1, inclusive)
    task_2_word_count = getCountOfWordsInDataSet(k_vocab_size,
        task_2_training_subset)

    # For each value of alpha_prime, train the model using the preditive
    # distribution, calculate the perplexity on the test set using the trained
    # model, then calculate the log of the evidence function for the value of
    # alpha_prime
    for alpha_prime in alpha_prime_values:

        # Find the predictive distribution for the training subset and value of
        # alpha_prime
        predictive_model = trainModelUsingPredictiveDistribution(
            task_2_word_count, task_2_training_set_size, k_vocab_size,
            alpha_prime)

        # Compute the test set perplexity and add it to the test set
        # perplexities list
        test_set_perplexities_for_varying_alphas.append(calculatePerplexity(
            predictive_model, test_set_by_indices))

        # Compute the log of the evidence function for the given value of
        # alpha_prime and the word counts from the training subset
        log_evidence_for_varying_alphas.append(calculateLogEvidence(
            task_2_word_count,task_2_training_set_size, k_vocab_size,
            alpha_prime))

    # Print the log evidence and perplexity results for the alpha values
    printTask2VaryingAlphaResults("Alpha prime value", alpha_prime_values)
    printTask2VaryingAlphaResults("Log evidence",
        log_evidence_for_varying_alphas)
    printTask2VaryingAlphaResults("Test set perplexity",
        test_set_perplexities_for_varying_alphas)

    # Plot the log evidence and perplexity results for the alpha values
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(alpha_prime_values, test_set_perplexities_for_varying_alphas,
        'gD-', label="Test set perplexities")
    ax2.plot(alpha_prime_values, log_evidence_for_varying_alphas, 'rs-',
        label="Log evidence")

    ax1.set_xlabel('Alpha prime value')
    ax1.set_ylabel("Test set perplexity", color='g')
    ax2.set_ylabel("Log evidence", color='r')

    fig.tight_layout()
    plt.title("Test set perplexity and log evidence for varying alpha prime values")
    plt.show()

    # Find the best alpha prime value (by finding the index of the maximum
    # log evidence value, and getting the corresponding alpha prime value)
    best_alpha_prime = alpha_prime_values[max(range(
        len(log_evidence_for_varying_alphas)),
        key = lambda x: log_evidence_for_varying_alphas[x])]
    print("The best value of alpha_prime based on the log evidence is "
        + str(best_alpha_prime))

"""
Run task 3.
"""
def task3():

    print("Task 3 ------------------------------------------------------------")
    training_set_file_name = "pg345.txt.clean"
    test_set_1_file_name = "pg84.txt.clean"
    test_set_2_file_name = "pg1188.txt.clean"

    # Read in the training set and both test sets
    training_set_strings = readStringsFromFile(training_set_file_name)
    test_set_1_strings = readStringsFromFile(test_set_1_file_name)
    test_set_2_strings = readStringsFromFile(test_set_2_file_name)

    # Form a vocabulary from the strings in all of the data sets
    all_strings = training_set_strings + test_set_1_strings + test_set_2_strings
    vocabulary_dict = buildVocabulary(all_strings)

    k_vocab_size = len(vocabulary_dict)

    # Convert the data sets to be represented by the index of the word in the
    # vocabulary_dict
    training_set_by_indices = transformStringDataSetToSetOfVocabIndices(
        vocabulary_dict, training_set_strings)
    test_set_1_by_indices = transformStringDataSetToSetOfVocabIndices(
        vocabulary_dict, test_set_1_strings)
    test_set_2_by_indices = transformStringDataSetToSetOfVocabIndices(
        vocabulary_dict, test_set_2_strings)

    # Get the number of occurrences of each vocabulary word in the training set
    training_set_word_count = getCountOfWordsInDataSet(k_vocab_size,
        training_set_by_indices)

    # Train the model using the predictive distribution
    predictive_model = trainModelUsingPredictiveDistribution(
        training_set_word_count, len(training_set_strings), k_vocab_size, 2)

    # Calculate and print the perplexities of both test sets given the
    # determined predictive distribution
    test_set_1_perplexity = calculatePerplexity(predictive_model,
        test_set_1_by_indices)
    test_set_2_perplexity = calculatePerplexity(predictive_model,
        test_set_2_by_indices)
    print("Perplexity for " + test_set_1_file_name + ": " + str(
        test_set_1_perplexity))
    print("Perplexity for " + test_set_2_file_name + ": " + str(
        test_set_2_perplexity))

    # We can predict that the test set with the lower perplexity was written
    # by the same author as the training set
    # Print the authorship findings
    same_author_file = test_set_2_file_name
    different_author_file = test_set_1_file_name
    if (test_set_1_perplexity < test_set_2_perplexity):
        same_author_file = test_set_1_file_name
        different_author_file = test_set_2_file_name
    print("The file with the same author as the training set file (" +
        training_set_file_name + ") is " + same_author_file +
        " and the file with a different author than the training set is "
        + different_author_file + ".")


if __name__ == "__main__":
    task1and2()
    task3()
