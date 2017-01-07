#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print "% Accuracy:", accuracy_score(labels_test, pred)

# How many POIs are predicted?
num_pred_POIs = 0
for is_poi in pred:
    if is_poi == 1:
        num_pred_POIs += 1
print "# predicted POIs:", num_pred_POIs
print "# people in test set:", len(labels_test)

# If the identifier predicted 0.0 (not POI) for everyone in test set,
# what would the accuracy be?
hypothetical_pred = [0.0] * len(pred)
print "% Accuracy (for all people in test set predicted not POI):", \
    accuracy_score(labels_test, hypothetical_pred)

# How many true positives?
num_true_pos = 0
for label, prediction in zip(labels_test, pred):
    if label == prediction == 1.0:
        num_true_pos += 1

print "# True Pos:", num_true_pos
from sklearn.metrics import precision_score, recall_score
print "Precision Score:", precision_score(labels_test, pred)
print "Recall Score:", recall_score(labels_test, pred)


