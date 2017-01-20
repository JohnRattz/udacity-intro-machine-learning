#!/usr/bin/python

import sys
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
# Top-level structures for grid search
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
# Selectors
from sklearn.feature_selection import SelectKBest
# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# Cross validation methods (obtaining training and testing data)
from sklearn.cross_validation import \
    train_test_split, StratifiedShuffleSplit
# Scoring
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, make_scorer, classification_report
from sklearn.cross_validation import cross_val_score

from random import random

sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


def remove_outliers(data_dict, features_list,
                    allow_poi_removal=False, make_plots=False):
    def make_scatter_plots(features_list_to_check):
        # Figures used to make scatter plots of the features
        # in `features_list_to_check`.
        figs = OrderedDict()
        for feature_indx in range(1, len(features_list_to_check)):
            feature_name = features_list_to_check[feature_indx]
            figs[feature_name] = plt.figure().add_subplot(111)
            figs[feature_name].set_title('POI vs. ' + feature_name)

        # For each person's set of features...
        for features in data_dict.itervalues():
            is_poi = 1.0 if features['poi'] else 0.0
            # Add this person's features to scatter plots.
            for feature_name, fig in figs.iteritems():
                feature_value = features[feature_name]
                if feature_value != 'NaN':
                    fig.scatter(features[feature_name], is_poi)
        # Show the figures.
        plt.show()

    # Ignore newly added features.
    features_list_to_check = features_list
    indices_to_remove = []
    for feature_indx in range(len(features_list)):
        if features_list_to_check[feature_indx] == 'has_recorded_address':
            indices_to_remove.append(feature_indx)
    for indx in reversed(indices_to_remove):
        del features_list_to_check[indx]
    # print features_list_to_check

    if make_plots:
        # Plot before removing outliers.
        make_scatter_plots(features_list_to_check)

    # Find the outliers.
    outlier_names = set()
    for person_name, features in data_dict.iteritems():
        if (allow_poi_removal) or \
           (not allow_poi_removal and not features['poi']):
            # Check 'salary'.
            if features['salary'] != 'NaN' and \
               features['salary'] > 5.0 * pow(10, 5):
                outlier_names.add(person_name)
            # Check 'bonus'.
            if features['bonus'] != 'NaN' and \
               features['bonus'] > 4.0 * pow(10, 6):
                outlier_names.add(person_name)
            # Check 'total_stock_value'.
            if features['total_stock_value'] != 'NaN' and \
               features['total_stock_value'] > 2.0 * pow(10, 7):
                outlier_names.add(person_name)
            # Check 'from_poi_to_this_person'.
            if features['from_poi_to_this_person'] != 'NaN' and \
               features['from_poi_to_this_person'] > 500:
                outlier_names.add(person_name)
            # Check 'shared_receipt_with_poi'.
            if features['shared_receipt_with_poi'] != 'NaN' and \
               features['shared_receipt_with_poi'] > 3000:
                outlier_names.add(person_name)
    # Show the list of outliers to be removed.
    # print outlier_names

    # Remove the outliers.
    for outlier_name in outlier_names:
        data_dict.pop(outlier_name, None)

    if make_plots:
        # Plot after removing outliers.
        make_scatter_plots(features_list_to_check)

def scoring_func_clf(clf, features, labels):
    """
    A scoring function to be used by GridSearchCV in `main()`.
    Taken from `tester.py` in this directory.
    """
    cv = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        # total_predictions = true_negatives + false_negatives + false_positives + true_positives
        # accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        # f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        return f1 if precision > 0.3 and recall > 0.3 else 0.0
    except:
        return 0.0

def scoring_func(features, labels):
    """
    A scoring function to be used by GridSearchCV in `main()`.
    Taken from `tester.py` in this directory.
    """
    cv = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        # total_predictions = true_negatives + false_negatives + false_positives + true_positives
        # accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        # f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        return f1 if precision > 0.3 and recall > 0.3 else 0.0
    except:
        return 0.0

def main():
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    # features_list = ['poi','salary'] # You will need to use more features
    # TODO: Implement more features?
    # I added the feature 'has_recorded_address'.
    features_list = ['poi', 'salary', 'bonus',
                     'total_stock_value', 'has_recorded_address',
                     'from_poi_to_this_person', 'shared_receipt_with_poi']

    ### Load the dictionary containing the dataset
    with open("../final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # BEGIN DEBUG
    print "Original # Data Points:", len(data_dict)
    # for key in data_dict.iterkeys():
    #     print key, data_dict[key]['salary']
    # END DEBUG

    ### Task 2: Remove outliers
    remove_outliers(data_dict, features_list,
                    allow_poi_removal=False, make_plots=False)
    print "# Data Points After Outlier Removal:", len(data_dict)

    ### Task 3: Create new feature(s)
    # Create the 'has_recorded_address' feature.
    for features in data_dict.itervalues():
        features['has_recorded_address'] = \
            1.0 if features['email_address'] != 'NaN' else 0.0

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    # for i in range(len(labels)):
    #     print labels[i], features[i]


    ### Task 4: Try a variety of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    # TODO: Use a greater variety of selectors and classifiers.

    # Define Pipelines.
    pipe_GaussianNB = \
        Pipeline([('selector', SelectKBest()), ('clf', GaussianNB())])
    pipe_SVC = \
        Pipeline([('selector', SelectKBest()), ('clf', SVC())])
    pipe_DecisionTreeClassfier = \
        Pipeline([('selector', SelectKBest()), ('clf', DecisionTreeClassifier())])

    # Show available parameters.
    print "SelectKBest Params:", SelectKBest().get_params().keys()
    print "GaussianNB Params:", GaussianNB().get_params().keys()
    print "SVC Params:", SVC().get_params().keys()
    print "DecisionTreeClassifier Params:", \
        DecisionTreeClassifier().get_params().keys()

    # Define Parameter Sets.
    params_GaussianNB = \
        {'selector': [SelectKBest()],
         # 'selector__k': [1, 2, 3, 4, 'all'],
         'selector__k': ['all'],
         'selector__score_func': [],
         'clf': [GaussianNB()]}

    params_SVC = \
        {'selector': [SelectKBest()],
         # 'selector__k': [1, 2, 3, 4, 'all'],
         'selector__k': ['all'],
         'clf': [SVC()],
         'clf__kernel': ['rbf', 'sigmoid'],
         # 'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
         'clf__C': [0.1, 1.0, 10.0, 100.0],
         # 'clf__C': [0.1, 0.5, 1.0, 10.0, 100.0],
         'clf__gamma': [0.001, 0.01, 0.1, 1.0, 'auto']}
         # 'clf__gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 'auto']}

    params_DecisionTreeClassifier = \
        {'selector': [SelectKBest()],
         # 'selector__k': [1, 2, 3, 4, 'all'],
         'selector__k': ['all'],
         'clf': [DecisionTreeClassifier()],
         'clf__criterion': ['gini', 'entropy'],
         'clf__splitter': ['best', 'random'],
         'clf__min_samples_split': [2, 10, 20, 30, 40, 50]}

    param_sets = [params_GaussianNB, params_SVC, params_DecisionTreeClassifier]
    pipes = [pipe_GaussianNB, pipe_SVC, pipe_DecisionTreeClassfier]

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Specify the scoring function for the `GridSearchCV` object below.
    scoring = scoring_func_clf
    # scoring = 'f1'

    # Tuples of scores and the corresponding classifiers.
    score_clf_tuples = []

    print "Commencing Grid Search"

    # num_CV_methods = 2
    # # For each train-test splitting method (acquiring training and testing data).
    # for i in range(num_CV_methods):
    #     print "Testing Train-test Splitting Method {}".format(i)
    #     features_train = []
    #     features_test = []
    #     labels_train = []
    #     labels_test = []
    #     if i == 0:
    #         # Train-test Splitting Method 1
    #         features_train, features_test, labels_train, labels_test = \
    #             train_test_split(features, labels, test_size=0.2, random_state=42)
    #     elif i == 1:
    #         # Train-test Splitting Method 2
    #         splitter = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.2, random_state=42)
    #         for train_idx, test_idx in splitter:
    #             for ii in train_idx:
    #                 features_train.append(features[ii])
    #                 labels_train.append(labels[ii])
    #             for jj in test_idx:
    #                 features_test.append(features[jj])
    #                 labels_test.append(labels[jj])
    #
    #     # Cross-validation object to use in the grid search (cross-validates on training set).
    #     cv = StratifiedShuffleSplit(labels_train, n_iter=5, test_size=0.2, random_state=42)
    #
    #     # print "features_train:", features_train[:20]
    #     # print "features_test:",  features_test[:20]
    #     # print "labels_train:",   labels_train[:20]
    #     # print "labels_test:",    labels_test[:20]
    #
    #     # For each pipeline and corresponding parameter set
    #     # to perform grid searches over...
    #     for j in range(len(pipes)):
    #         print "Testing Pipeline {}".format(j)
    #         # `n_jobs=-2` to run with all but one logical processor.
    #         grid_search = GridSearchCV(pipes[j], param_sets[j],
    #                                    scoring=scoring, cv=cv,
    #                                    n_jobs=-2, verbose=1)
    #         # Train on training samples so that scores reflect generalization performance.
    #         grid_search.fit(features_train, labels_train)
    #         clf = grid_search.best_estimator_
    #         score = grid_search.best_score_
    #         score_clf_tuples.append((score, clf))

    cv = StratifiedShuffleSplit(labels, n_iter=10, test_size=0.1, random_state=42)
    # For each pipeline and corresponding parameter set
    # to perform grid searches over...
    for j in range(len(pipes)):
        print "Testing Pipeline {}".format(j)
        # `n_jobs=-2` to run with all but one logical processor.
        grid_search = GridSearchCV(pipes[j], param_sets[j],
                                   scoring=scoring, cv=cv,
                                   n_jobs=-2, verbose=1)
        grid_search.fit(features, labels)
        clf = grid_search.best_estimator_
        score = grid_search.best_score_
        score_clf_tuples.append((score, clf))


    # Of all classifiers selected in the grid searches,
    # choose the one with the best score.
    score_clf_tuples.sort(key=lambda tup: tup[0], reverse=True)
    best_score = score_clf_tuples[0][0]
    clf = score_clf_tuples[0][1]
    print "Best classifier:", clf
    print "Best classifier report:\n", \
        classification_report(labels, clf.predict(features))

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    dump_classifier_and_data(clf, my_dataset, features_list)

# This may be necessary for the 'GridSearchCV' object
# created in `main()` to run in parallel.
# Specifically, this may be necessary to run with the parameter `n_jobs=-1`.
if __name__ == '__main__':
    main()