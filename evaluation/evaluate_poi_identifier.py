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

## split training/testing data
testHold = .3
randState = 42
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=testHold, random_state=randState)


## fit to decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

## test on data set
print "score: ", clf.score(features_test, labels_test)

## # of predicted pois in test set
pred = clf.predict(features_test)
cnt = 0
tot = 0
for x in pred:
    cnt += int(x)
    tot += 1
print pred
print "# of predicted POIs in test:", cnt, "out of", tot

## # of actual pois in test set
cnt = 0
tot = 0
for x in labels_test:
    cnt += int(x)
    tot += 1
print "# of actual POIs in test:", cnt, "out of", tot

## print confusion matrix of prediction
from sklearn import metrics
print metrics.confusion_matrix(labels_test, pred)
print "recall:", metrics.recall_score(labels_test, pred)
print "precision:", metrics.precision_score(labels_test, pred)

## sepcial quz test
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print metrics.confusion_matrix(true, predictions)
print metrics.recall_score(true, predictions)
print metrics.precision_score(true, predictions)

