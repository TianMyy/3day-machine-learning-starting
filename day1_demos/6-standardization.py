from sklearn.preprocessing import StandardScaler
import pandas as pd

def stand_demo():
    """
    standardization
    :return:
    """
    #1.get data
    data = pd.read_csv("dating.txt")
    data.iloc[:, :3]
    print("data:\n", data)

    #2.instantiate a converter
    transfer = StandardScaler()

    #3.use fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None

if __name__ == "__main__":
    # code9: standardization use dating.txt
    stand_demo()