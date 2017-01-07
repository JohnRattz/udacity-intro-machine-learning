#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print enron_data.keys()
print "# data points:", len(enron_data)
print "# features", len(enron_data['METTS MARK'])

print [enron_data[person_name]["poi"]
       for person_name in enron_data.keys()]
print "# POIs:", len([is_poi for is_poi in [enron_data[person_name]["poi"]
                                            for person_name in enron_data.keys()] if is_poi])

print enron_data["SKILLING JEFFREY K"]

print "# Salaries:", len([salary for salary in
                          [enron_data[person_name]['salary']
                           for person_name in enron_data.keys()]
                          if salary != 'NaN'])

print "# Email Addresses:", len([addr for addr in
                                 [enron_data[person_name]['email_address']
                                  for person_name in enron_data.keys()]
                                 if addr != 'NaN'])

payments = [enron_data[person_name]['total_payments']
            for person_name in enron_data.keys()]
num_payments = len(payments)
num_NaN_payments = len([payment for payment in payments
                        if payment == 'NaN'])
percent_NaN_payments = float(num_NaN_payments) / num_payments
print "% NaN Payments:", percent_NaN_payments

total_payments = [enron_data[person_name]['total_payments']
                  for person_name in enron_data.keys()]
num_payments = len(payments)
num_NaN_payments = len([payment for payment in payments
                        if payment == 'NaN'])
percent_NaN_payments = float(num_NaN_payments) / num_payments
print "# NaN Payments:", num_NaN_payments
print "% NaN Payments:", percent_NaN_payments

pois = [poi for poi in
        [enron_data[person_name]
         for person_name in enron_data.keys()]
        if poi['poi']]
num_pois = len(pois)
num_poi_NaN_payments = len([payment for payment in
                            [poi['total_payments']
                             for poi in pois]
                            if payment == 'NaN'])
print "% NaN Payments (PoIs):", num_poi_NaN_payments / num_pois
