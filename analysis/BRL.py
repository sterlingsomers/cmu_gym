from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
import time
import sys
sys.path.append("/Users/constantinos/Documents/Projects/sklearn_expertsys")
from BigDataRuleListClassifier import * # You might need this file to be inside the folder sklearn_exoertsys
from SVMBigDataRuleListClassifier import *


''' You can have one BRL for V(s) and one for the policy π(a|s)'''

'''Get the data'''
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/All_maps_random_500_drop_traj.tj','rb')
dict = pickle.load(pickle_in)


''' CHECK WHAT KIND OF DATA STRUCTURE EACH OF THOSE IS '''
feature_labels = ["FL object's alt", "L object's alt", "C object's alt",
                  "R object's alt", "FR object's alt", "Drone's alt", "hiker pos"]

data = fetch_mldata("diabetes")  # get dataset
y = -(data.target - 1) / 2  # target labels (0: healthy, or 1: diabetes) - the original dataset contains -1 for diabetes and +1 for healthy

###############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y)  # split