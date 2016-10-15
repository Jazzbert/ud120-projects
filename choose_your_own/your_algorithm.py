#!/usr/bin/python

#import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
#from class_vis import prettyPicture
from time import time
import csv

### Test parameters
#ALG_TYPE = "GaussianNB"

ALG_TYPE = "SVM"
SVM_C = 1.0
SVM_KERNEL = 'rbf'
SVM_GAMMA = 'auto'

#ALG_TYPE = "DecisionTree"

#ALG_TYPE = "AdaBoost"

PARAM = "C"
PARAM_MIN = 10000
PARAM_MAX = 10**20
def PARAM_CHANGE( x, n ):
	return x * 10**n


features_train, labels_train, features_test, labels_test = makeTerrainData()

### Set Up File Save
ofile = open('output.csv', 'wb')
writer = csv.writer(ofile)


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
#grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

if ALG_TYPE == "GaussianNB":
	print "GaussianNB default"
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
elif ALG_TYPE == "SVM":
	print "SWM C =", SVM_C, " kernel =", SVM_KERNEL, " gamma =", SVM_GAMMA
	from sklearn.svm import SVC
	clf = SVC(C = SVM_C, kernel = SVM_KERNEL, gamma = SVM_GAMMA)
elif ALG_TYPE == "DecisionTree":
	print "DecisionTree default"
	from sklearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier()
elif ALG_TYPE == "AdaBoost":
	print "AdaBoost default"
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier()
else:
	print "Error in code"
	sys.exit()

incr = 0
val = float(PARAM_CHANGE( PARAM_MIN, incr ))
while ( val <= PARAM_MAX ):
	param_setting = {PARAM:val}
	print param_setting
	clf.set_params( **param_setting )
	t0 = time()
	clf = clf.fit(features_train, labels_train)
	dur0 = time()-t0
	print "training time: ", round(dur0, 3), "s"
	t2 = time()
	acc = clf.score(features_test, labels_test)
	print acc
	dur2 = time()-t2
	print "test time: ", round(dur2, 3), "s"
	writer.writerow([ALG_TYPE, param_setting, acc, dur0, dur2])
	incr += 1
	val = float(PARAM_CHANGE( PARAM_MIN, incr ))

ofile.close

#try:
#    prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass
