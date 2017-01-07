#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    # Get absolute errors of elements (corresponding indices) of `predictions`.
    errors = {}
    abs_errors = {}
    for i in range(len(predictions)):
        errors[i] = net_worths[i][0] - predictions[i][0]
        abs_errors[i] = abs(errors[i])

    # Sort by decreasing absolute error.
    sorted_errors_indices = sorted(abs_errors, key=abs_errors.get, reverse=True)
    # print sorted_errors_indices

    # Get the 10% with highest absolute error.
    sorted_highest_error_indices = \
        sorted(sorted_errors_indices[:int(len(sorted_errors_indices) * 0.1)])

    # Removed 10% of samples with highest absolute error.
    for i in reversed(sorted_highest_error_indices):
        predictions = np.concatenate((predictions[:i], predictions[i+1:]))
        ages = np.concatenate((ages[:i], ages[i+1:]))
        net_worths = np.concatenate((net_worths[:i], net_worths[i + 1:]))
    # print sorted_highest_error_indices

    # Put the remaining values into tuples.
    for i in range(len(ages)):
        cleaned_data.append((ages[i][0], net_worths[i][0], errors[i]))

    print "Cleaned data (age, net worth, error):\n", cleaned_data
    return cleaned_data

