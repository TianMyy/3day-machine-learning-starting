from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

def linear_ridge():
    '''
    use Ridge Regression to predict the house price in boston
    :return:
    '''

    # 1)get data
    boston = load_boston()

    # 2)divide dataset
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

    # 3)feature engineering:standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4)Logistic Regression algorithm estimator
    estimator = Ridge(max_iter=10000, alpha=0.5)
    estimator.fit(x_train, y_train)
    # # save model
    #joblib.dump(estimator, 'my_ridge.pkl')
    # # load model
    # estimator = joblib.load('my_ridge.pkl')


    # 5)get model
    print('coef_:', estimator.coef_)
    print('intercept_:', estimator.intercept_)

    # 6)model assessment
    y_pred = estimator.predict(x_test)
    print('predicted house price: ', y_pred)
    error = mean_squared_error(y_test, y_pred)
    print('MSE: ', error)

    return None

if __name__ == '__main__':
    # # code 1: use Normal Equation to predict the house price in boston
    # linear_ne()

    # # code 2: use Gradient Descent to predict the house price in boston
    # linear_gd()

    # code 3: use Ridge Regression to predict the house price in boston
    linear_ridge()