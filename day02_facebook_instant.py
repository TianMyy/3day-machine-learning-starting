import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 1. get the data
data=pd.read_csv('./FBlocation/train.csv')

# 2. basic data processing
# 1) reduce the data scope
data=data.query('x > 2 &  x < 2.5 & y > 1 & y < 1.5')
# 2) process timestamp
time_value = pd.to_datatime(data['time'], unit='s')  # type is Series
date = pd.DatetimeIndex(time_value)
data.loc[:, 'day'] = date.day
data.loc[:, 'weekday'] = date.weekday
data.loc[:, 'hour'] = date.hour
# 3) filter places with less marks
place_count = data.groupby('place_id').count()['row_id']
data_processed = data[data['place_id'].isin(place_count[place_count > 3].index.values)]
# 4) filter characteristic values and target values
x = data_processed[['x', 'y', 'accuracy', 'day', 'weekday', 'hour']]
y = data_processed['place_id']
# 5) split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 3. feature engineering:standardization
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. KNN algorithm estimator
estimator = KNeighborsClassifier()
# add grid search and cross validation
param_dict = {"n_neighbors":[3, 5, 7, 9]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

estimator.fit(x_train, y_train)

# 5. model assessment
# method 1:compare the true value and predicted value straightly
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("compare true and predicted value:\n", y_test == y_predict)
# method 2:calculate the accuracy rate
score = estimator.score(x_test, y_test)
print("accuracy_score:\n", score)
# best parameter
print("best param:\n", estimator.best_params_)
# best score
print("best score:\n", estimator.best_score_)
# best estimator
print("best estimator:\n", estimator.best_estimator_)
# cross validation result
print("cross validation result:\n", estimator.cv_results_)