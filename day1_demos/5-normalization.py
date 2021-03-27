from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def minmax_demo():
    """
    normalization
    :return:
    """
    #1.get data
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    #2.instantiate a converter
    transfer = MinMaxScaler()

    #3.use fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None

if __name__ == "__main__":
    # code8: normalization use dating.txt
    minmax_demo()