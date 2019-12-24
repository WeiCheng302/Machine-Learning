import csv
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

# import files
train_data = pd.read_csv('D:/ML/04/introML2019F_task4_train.csv')
test_data = pd.read_csv('introML2019F_task4_test_shuffled_noanswers.csv')

# Training/ Testing data
feature_train = train_data[train_data.columns[0:-1]].values
label_train = train_data[train_data.columns[-1]].values
feature_test = test_data[test_data.columns].values

train_data_2 = train_data.replace(-1, pd.np.nan)
test_data_2 = test_data.replace(-1, pd.np.nan)
Data_nonan_train = train_data_2.dropna()
Data_nonan_test = test_data_2.dropna()
med3 = Data_nonan_train.median()
med4 = Data_nonan_test.median()
#print(med3[1])
#print(Data_nonan.median())
# Creating Median
print('Data Processing Begin')
#med = np.zeros(shape=(1, 26))
#med2 = np.zeros(shape=(1, 26))

#feature_train_nom1 = train_data[train_data[train_data.columns[0:-1]].values != -1 ].median()
#print(feature_train_nom1)

#for i in range(26):
    #med[0][i] = np.median(train_data[train_data.columns[:1800000][i]].values)
    #med2[0][i] = np.median(test_data[test_data.columns[:450000][i]].values)

for row in range(1800000):
    for value in range(26):
        if feature_train[row][value] == -1:
            feature_train[row][value] = med3[value]

for row in range(450000):
    for value in range(26):
        if feature_test[row][value] == -1:
            feature_test[row][value] = med4[value]
print('Data Processing Finish')

# Training
print('Start Training')
model = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=12, subsample=1, reg_lambda=2, n_jobs=4)
model.fit(feature_train, label_train, eval_metric='auc')

# Predicting
print('Start Predicting')
prediction = model.predict(feature_test)

# CSV Writing
print('Start Writing')
file = open('answer4.csv', 'w', newline='')
w = csv.writer(file)
w.writerow(['ID', 'Category'])
q = 0
for i in prediction:
    w.writerow([q+1, prediction[q]])
    q += 1
file.close()
