import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, linear_model
from sklearn.metrics import mean_squared_error, r2_score

clf = tree.DecisionTreeClassifier(criterion='entropy')
df = pd.read_csv('C:/Users/user/Desktop/ML/01/introML2019F_task1_train_shuffled.csv')
cov = df[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
          'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11',
          'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17',
          'feature_18', 'feature_19', 'feature_20']].to_numpy()
y = df[['label']].to_numpy()
clf = clf.fit(cov, y)

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
                                                'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
                                                'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
                                                'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19',
                                                'feature_20'],
                                class_names=['label'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph