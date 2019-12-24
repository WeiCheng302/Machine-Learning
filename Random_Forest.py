import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('introML2019F_task3_train.csv')
test_data = pd.read_csv('introML2019F_task3_test_shuffled_noanswers.csv')

feature_train = train_data[train_data.columns[0:-1]].values
feature_train2 = feature_train[:-100]
feature_train2_test = feature_train[-100:]
label_train = train_data[train_data.columns[-1]].values
label_train2= label_train[:-100]
label_train2_test = label_train[-100:]
feature_test = test_data.values
print('Start Training')

forest = RandomForestClassifier(criterion='entropy',n_estimators=5,
                                random_state=3,n_jobs=2)
print('Start Predicting')
forest.fit(feature_train, label_train)
predict = forest.predict(feature_test)
#print(predict)
#plot_decision_regions(feature_test, label_test,classifier=forest)

importances = forest.feature_importances_
print(importances)
#std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
#indices = np.argsort(importances)[::-1]




import csv
file = open('answer3.csv', 'w', newline='')
w = csv.writer(file)
w.writerow(['ID', 'Category'])
q = 0
for i in predict:
    #print('4')
    w.writerow([q+1, predict[q]])
    q += 1
file.close()
