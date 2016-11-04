#!/usr/bin/python

import numpy
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

full_list = ["salary", "to_messages", "deferral_payments", "total_payments", "loan_advances", "bonus", "email_adddress", "restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses", "from_poi_to_this_person", "exercised_stock_options", "from_messages", "other", "from_this_person_to_poi", "poi", "long_term_incentive", "shared_receipt_with_poi", "restricted_stock", "director_fees"]

for feat in full_list:
    
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi',feat] # You will need to use more features

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    ### Task 2: Remove outliers
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    nan_count = 0
    
    for x in features:
        if x == "NaN":
            nan_count += 1

    print feat, (len(features) - nan_count) / len(features)
