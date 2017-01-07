#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
# Remove outlier.
del data_dict["TOTAL"]

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
outliers = {k:v for k,v in data_dict.items()
            if (v['salary'] != 'NaN' and v['salary'] > 1000000)
            and (v['bonus'] != 'NaN' and v['bonus'] > 5000000)}
print outliers.keys()

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


