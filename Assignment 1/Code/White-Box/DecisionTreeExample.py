# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:30:34 2018

@author: sande
"""

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import train_test_split

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
X = iris.data
y = iris.target
dot_data = tree.export_graphviz(clf, out_file=None,class_names=["clas1","class2","class3"], node_ids=True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

path = clf.decision_path(X_test)
print(type(path))
print(path)

print("Entry type:", type(path[0]))
print("First entry:", path[0])
print("Second entry:", path[1])

path_array = path.toarray()
print("path_array:\n", path_array)

#0, 2, 13, 16

#graph = graphviz.Source(dot_data) 
#graph.render("iris") 
#print ("Zeroth sample",X_test[0])
#pred = clf.predict(X_test)
#print (pred[0])