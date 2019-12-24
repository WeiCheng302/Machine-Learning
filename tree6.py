import csv
import numpy as np
import pandas as pd
from sklearn import tree

# import files
train_data = pd.read_csv('D:/ML/06/introML2019F_task6_train.csv')
test_data = pd.read_csv('introML2019F_task6_test_shuffled_noanswers.csv')

# Training/ Testing data
numeric_feature_train = train_data[train_data.columns[0:-2]].values
numeric_feature_test = test_data[test_data.columns[0:-1]].values
label_train = train_data[train_data.columns[-1]].values

# Nominal Data
nominal_features_train = train_data[train_data.columns[-2]].values
nominal_features_test = test_data[test_data.columns[-1]].values
other_features_train = np.zeros(shape=(200000, 4))
other_features_train2 = np.zeros(shape=(200000, 1))
other_features_test = np.zeros(shape=(50000, 4))
other_features_test2 = np.zeros(shape=(50000, 1))

# Nominal Data Processing
print('Data Processing Begin')
count = 0
for word in nominal_features_train:
    if word == 'A':
        other_features_train[count][0] = 1
        other_features_train2[count][0] = 1
        count += 1
    elif word == 'B':
        other_features_train[count][1] = 1
        other_features_train2[count][0] = 3
        count += 1
    elif word == 'C':
        other_features_train[count][2] = 1
        other_features_train2[count][0] = 7
        count += 1
    else:
        other_features_train[count][3] = 1
        other_features_train2[count][0] = 10
        count += 1


count = 0
for word in nominal_features_test:
    if word == 'A':
        other_features_test[count][0] = 1
        other_features_test2[count][0] = 1
        count += 1
    elif word == 'B':
        other_features_test[count][1] = 1
        other_features_test2[count][0] = 3
        count += 1
    elif word == 'C':
        other_features_test[count][2] = 1
        other_features_test2[count][0] = 7
        count += 1
    else:
        other_features_test[count][3] = 1
        other_features_test2[count][0] = 10
        count += 1

feature_train = np.hstack([numeric_feature_train, other_features_train])
feature_test = np.hstack([numeric_feature_test, other_features_test])
print('Data Processing Finish')

# Training and Predicting
print('Start Training')
model = tree.DecisionTreeClassifier(criterion='entropy')#, max_depth= 25)
model = model.fit(feature_train, label_train)

print('Start Predicting' )
prediction = model.predict(feature_test)

# File Writing
print('Start Writing')
file = open('answer6.csv', 'w', newline='')
w = csv.writer(file)
w.writerow(['ID', 'Category'])
q = 0
for i in prediction:
    w.writerow([q+1, prediction[q]])
    q += 1
file.close()

print('Finish Writing')
print('Finish')
