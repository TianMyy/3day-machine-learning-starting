import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def titanic_decision_tree():
    # 1.get data
    path = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt'
    titanic = pd.read_csv(path)

    # 2.get the Characteristic value and Target value
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']

    # 3.data processing
    # deal with NaN
    x['age'].fillna(x['age'].mean, inplace=True)
    # convert to dictionary
    x.to_dict(orient='records')

    # 4.divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 5.dictionary feature extraction
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 6.Decision Tree estimator
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)

    # 7.model assessment
    # method 1:compare the true value and predicted value straightly
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true and predicted value:\n", y_test == y_predict)
    # method 2:calculate the accuracy rate
    score = estimator.score(x_test, y_test)
    print("accuracy_score:\n", score)
    # Decision Tree visualization
    export_graphviz(estimator, out_file='titanic_tree.dot', feature_names=transfer.get_feature_names())

    return None

def titanic_random_forest():
    # 1.get data
    path = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt'
    titanic = pd.read_csv(path)

    # 2.get the Characteristic value and Target value
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']

    # 3.data processing
    # deal with NaN
    x['age'].fillna(x['age'].mean, inplace=True)
    # convert to dictionary
    x.to_dict(orient='records')

    # 4.divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 5.dictionary feature extraction
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 6.Random Forest estimator
    estimator = RandomForestClassifier()
    # add grid search and cross valid
    param_dict = {'n_estimators':[120, 200, 300, 500, 800, 1200],
                  'max_depth':[5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    # 7.model assessment
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

    return None

if __name__ == "__main__":
    # decision tree demo
    #titanic_decision_tree()

    # random forest demo
    titanic_random_forest()