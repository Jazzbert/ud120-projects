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
enron_data.pop("TOTAL", 0) # outlier

# list features
print enron_data.items()[0][1].keys()

# count poi
count = 0
total = 0
for d, v in enron_data.items():
    total += 1
    if v['poi'] == True:
        count += 1

print "# of PoIs: ", count
print "Out of: ", total

# find min and max 'exercised_stock_options'
maxVal = 0
minVal = -1
f1 = 'exercised_stock_options'
for d, v, in enron_data.items():
    curVal = v[f1]
    if curVal != "NaN":
        if minVal == -1:
            minVal = curVal
            minName = d
        if curVal < minVal:
            minVal = curVal
            minName = d
        if curVal > maxVal:
            maxVal = curVal
            maxName = d
print "Max", f1, "=", maxVal, "by", maxName
print "Min", f1, "=", minVal, "by", minName

# find min and max 'salary'
maxVal = 0
minVal = -1
f1 = 'salary'
for d, v, in enron_data.items():
    curVal = v[f1]
    if curVal != "NaN":
        if minVal == -1:
            minVal = curVal
            minName = d
        if curVal < minVal:
            minVal = curVal
            minName = d
        if curVal > maxVal:
            maxVal = curVal
            maxName = d
print "Max", f1, "=", maxVal, "by", maxName
print "Min", f1, "=", minVal, "by", minName
