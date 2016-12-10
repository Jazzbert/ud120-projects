#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments', 'net_early_out']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### CC - Create new feature for total early papments =
###      deferral_payments + exercised_stock_options
new_feature = "net_early_out"
for person in my_dataset:
    if my_dataset[person]['deferral_payments'] == "NaN":
        my_def_payment = 0
    else:
        my_def_payment = my_dataset[person]['deferral_payments']

    if my_dataset[person]['exercised_stock_options'] == "NaN":
        my_exer_opt = 0
    else:
        my_exer_opt = my_dataset[person]['exercised_stock_options']

    if my_dataset[person]['loan_advances'] == "NaN":
        my_loan_adv = 0
    else:
        my_loan_adv = my_dataset[person]['loan_advances']

    my_dataset[person][new_feature] = my_def_payment + my_exer_opt + my_loan_adv


### CC - Show base metrics on each feature
##full_list = ["salary", "to_messages", "deferral_payments", "total_payments", "loan_advances", "bonus", "restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses", "from_poi_to_this_person", "exercised_stock_options", "from_messages", "other", "from_this_person_to_poi", "poi", "long_term_incentive", "shared_receipt_with_poi", "restricted_stock", "director_fees"]
##for feat in full_list:
##    feat_test = ['poi', feat]
##    data = featureFormat(data_dict, feat_test, sort_keys = True)
##    labels, features = targetFeatureSplit(data)
##    print feat, "values:", len(features), " percent:", float(len(features))/float(len(data_dict))
##
##cols = len(data_dict)
##print "cols:", cols
##for x in range(len(cols)):
##    print "items in col", x, len(data_dict[x])

### Add together the total payments and exercised options for a estimate of net "cash out"


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Code for visualizeing and looking a different data characteristics

import matplotlib.pyplot
for point in data:
    x = point[0]
    y = point[1]
    matplotlib.pyplot.scatter(x, y)

matplotlib.pyplot.xlabel("poi")
matplotlib.pyplot.ylabel("total_payments")
#matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf = 4)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = clf.fit(features_train, labels_train)
scr = clf.score(features_test, labels_test)
print scr

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


### TODO: Cleanup this test code later on:
### CC - Test code to run metrics each time
from tester import test_classifier
test_classifier(clf, my_dataset, features_list)
