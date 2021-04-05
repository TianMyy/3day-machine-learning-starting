from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def linear_ne():
    '''
    use Normal Equation to predict the house price in boston
    :return:
    '''

    # 1)get data
    boston = load_boston()

    # 2)divide dataset
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.taget, random_state=0)

    # 3)feature engineering:standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4)Logistic Regression algorithm estimator
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5)get model
    print('coef_:', estimator.coef_)
    print('intercept_:', estimator.intercept_)

    # 6)model assessment
    # method 1:compare the true value and predicted value straightly
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true and predicted value:\n", y_test == y_predict)
    # method 2:calculate the accuracy rate
    score = estimator.score(x_test, y_test)
    print("accuracy_score:\n", score)

    return None

if __name__ == '__main__':
    # code 1: use Normal Equation to predict the house price in boston
    linear_ne()