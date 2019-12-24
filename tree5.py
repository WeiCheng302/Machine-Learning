import csv
import numpy as np
import pandas as pd
from sklearn import tree

# import files
train_data = pd.read_csv('D:/ML/05/introML2019F_task5_train.csv')
test_data = pd.read_csv('introML2019F_task5_test_shuffled_noanswers.csv')

# Training/ Testing data
numeric_feature_train = train_data[train_data.columns[0:-2]].values
numeric_feature_test = test_data[test_data.columns[0:-1]].values
label_train = train_data[train_data.columns[-1]].values

train_data_2 = train_data[train_data.columns[0:-2]].replace(-1, pd.np.nan)
test_data_2 = test_data[test_data.columns[0:-1]].replace(-1, pd.np.nan)
Data_nonan_train = train_data_2.dropna()
Data_nonan_test = test_data_2.dropna()
med3 = Data_nonan_train.median()
med4 = Data_nonan_test.median()
# Nominal Data
nominal_features_train = train_data[train_data.columns[-2]].values
nominal_features_test = test_data[test_data.columns[-1]].values
other_features_train = np.zeros(shape=(900000, 4))
other_features_train2 = np.zeros(shape=(900000, 1))
other_features_test = np.zeros(shape=(225000, 4))
other_features_test2 = np.zeros(shape=(225000, 1))

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

# Missing Value Processing
#med = np.zeros(shape=(1, 26))
#med2 = np.zeros(shape=(1, 26))
#for i in range(25):
 #   med[0][i] = np.median(train_data[train_data.columns[:900000][i]].values)
  #  med2[0][i] = np.median(test_data[test_data.columns[:225000][i]].values)

for row in range(900000):
    for value in range(25):
        if numeric_feature_train[row][value] == -1:
            numeric_feature_train[row][value] = med3[value]

for row in range(225000):
    for value in range(25):
        if numeric_feature_test[row][value] == -1:
            numeric_feature_test[row][value] = med4[value]

feature_train = np.hstack([numeric_feature_train, other_features_train2])
feature_test = np.hstack([numeric_feature_test, other_features_test2])
print('Data Processing Finish')

# Training and Predicting
print('Start Training')
model = tree.DecisionTreeClassifier(criterion='entropy')# , max_depth= 25)
model = model.fit(feature_train, label_train)

print('Start Predicting' )
prediction = model.predict(feature_test)

# File Writing
print('Start Writing')
file = open('answer5.csv', 'w', newline='')
w = csv.writer(file)
w.writerow(['ID', 'Category'])
q = 0
for i in prediction:
    w.writerow([q+1, prediction[q]])
    q += 1
file.close()

print('Finish Writing')
print('Finish')
