#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Show base metrics on each feature
full_list = ["salary", "to_messages", "deferral_payments", "total_payments", "loan_advances", "bonus", "restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses", "from_poi_to_this_person", "exercised_stock_options", "from_messages", "other", "from_this_person_to_poi", "poi", "long_term_incentive", "shared_receipt_with_poi", "restricted_stock", "director_fees"]
for feat in full_list:
    feat_test = ['poi', feat]
    data = featureFormat(data_dict, feat_test, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    print feat, "values:", len(features), " percent:", float(len(features))/float(len(data_dict))

### Task 2: Remove outliers


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

import matplotlib.pyplot
for point in data:
    salary = point[1]
    total_payments = point[2]
    matplotlib.pyplot.scatter(salary, total_payments)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_payments")
matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


### Outlier cleaner modified from previous lesson
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

    size = len(predictions)
    cleanSize = int(size * .1)
    bigError = [0] * cleanSize
    
    print "size: ", size
    print "cleanSize", cleanSize

    for x in range(0, size):

        errorVal = abs(predictions[x] - net_worths[x]) 
        
        # set value of cleaned data
        cleaned_data.append((ages[x], net_worths[x], errorVal))

        # keep track of the 10 largest error values
        if errorVal >= bigError[0]:
            bigError[0] = errorVal
            bigError = sorted(bigError)

    print bigError

    # remove biggest errors from list
    for y in range (0, cleanSize):
        for z in xrange (0, len(cleaned_data)):
            if cleaned_data[z][2] == bigError[y]:
                del cleaned_data[z]
                break
    
    return cleaned_data

