#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

print len(data[ : , 0])
print len(data[ : , 1])
### your code below

maxError = [(0, 0)]*4

### get prediction to find bigest outliers
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(data[ : , 0].reshape(94, 1), data[ : , 1].reshape(94, 1))

for index, point in enumerate(data):

    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

    # track outliers by comparing against prediction
    error = abs(bonus - reg.predict(salary)[0])
    if error >= maxError[0][0]:
        maxError[0] = (error, index)
        maxError = sorted(maxError)
        print maxError

# find keys for max points
for x in range(0, 4):
    curIndex = maxError[x][1]
    print curIndex
    print data_dict.keys()[curIndex], maxError[x][0], data[curIndex, 0]

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

