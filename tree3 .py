import csv
import pandas as pd
from sklearn import tree


train_data = pd.read_csv('introML2019F_task3_train.csv')
test_data = pd.read_csv('introML2019F_task3_test_shuffled_noanswers.csv')

print('Data Processing Begin')
feature_train = train_data[train_data.columns[4:-1]].values
label_train = train_data[train_data.columns[-1]].values
feature_test = test_data[test_data.columns[4:]].values

print('Start Training')
model = tree.DecisionTreeClassifier(criterion='entropy')   # , max_depth=depth)
model = model.fit(feature_train, label_train)

print('Start Predicting')
prediction = model.predict(feature_test)

#test_pred = model.predict(test_data)

print('Start Writing')
file = open('answer3.csv', 'w', newline='')
w = csv.writer(file)
w.writerow(['ID', 'Category'])
q = 0
for i in prediction:
    w.writerow([q+1, prediction[q]])
    q += 1
file.close()
