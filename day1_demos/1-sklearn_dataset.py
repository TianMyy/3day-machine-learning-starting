from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def datasets_demo():
    """
    sklearn dataset usage
    :return:
    """
    #get the dataset
    iris = load_iris()
    print("iris dataset:\n", iris)
    print("view description of dataset\n", iris["DESCR"])
    print("view names of feature values\n", iris.feature_names)
    print("view the feature values\n", iris.data, iris.data.shape)

    #split of dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("feature value of trainning dataset:\n", x_train, x_train.shape)

    return None

if __name__ == "__main__":
    # code1: sklearn dataset usage
    datasets_demo()