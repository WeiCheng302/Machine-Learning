import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, linear_model
from sklearn.metrics import mean_squared_error, r2_score

depth = 20
count = 0
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
df = pd.read_csv('C:/Users/user/Desktop/ML/02/introML2019F_task2_train.csv')

feature = df[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
           'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11',
           'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17',
           'feature_18', 'feature_19', 'feature_20']]
label = df[['label']]

feature_train = feature[:-1000]
label_train = label[:-1000]
feature_test = feature[-1000:]

clf = clf.fit(feature_train, label_train)
k = clf.predict(feature_test)

error = 0
for i, v in enumerate(k):
    if v != label[-1000:]['label'].values[i]:
        print(i, v)
        error += 1
print(error)


df2=pd.read_csv('C:/Users/user/Desktop/ML/02/introML2019F_task2_test_shuffled_noanswers.csv')
feature2 = df2[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
            'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11',
            'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17',
            'feature_18', 'feature_19', 'feature_20']]
a = clf.predict(feature2)


import csv
f = open('answer2.csv', 'w', newline='')
w = csv.writer(f)
w.writerow(['ID', 'Category'])
q = 0
for i in a:
    w.writerow([q+1, a[q]])
    q += 1
f.close()