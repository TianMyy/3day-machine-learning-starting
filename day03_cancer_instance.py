import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# use logistic regression to predict the cancer

# 1.get data
path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
column_name = ['Sample code number', 'CClump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
               'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(path, names=column_name)

# 2.Deal with NaN data
# 1) replace np.nan
data = data.replace(to_replace='?', value=np.nan)
# 2) delete
data.dropna(inplace=True)

# 3.divide dataset
# get the Characteristic value and Target value
x = data.iloc[:, 1:-1]
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# 4.Standardization
transfer = StandardScaler()
x_train = transfer.fit_transform((x_train))
x_test = transfer.transform(x_test)

# 5.Logistic Regression estimator
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
# coef_ and intercept_
print('coef_:', estimator.coef_)
print('intercept_:', estimator.intercept_)

# 6.model assessment
# method 1:compare the true value and predicted value straightly
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("compare true and predicted value:\n", y_test == y_predict)
# method 2:calculate the accuracy rate
score = estimator.score(x_test, y_test)
print("accuracy_score:\n", score)

