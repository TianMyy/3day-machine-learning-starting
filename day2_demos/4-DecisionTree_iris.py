from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def decision_iris():
    '''
    use Decision tree to classify the iris dataset
    :return:
    '''

    # 1)get data
    iris=load_iris()

    # 2)divide dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

    # 3)Decision Tree estimator
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)

    # 4)model assessment
    # method 1:compare the true value and predicted value straightly
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true and predicted value:\n", y_test == y_predict)
    # method 2:calculate the accuracy rate
    score = estimator.score(x_test, y_test)
    print("accuracy_score:\n", score)

    return None

if __name__ == "__main__":
    # code 4: use Decision tree to classify the iris dataset
    decision_iris()




