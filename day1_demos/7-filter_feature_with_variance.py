from scipy.stats import pearsonr
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def variance_demo():
    """
    filter feature with low variance
    :return:
    """
    #1.get data
    data = pd.read_csv("factor_return.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)

    #2.instantiate a converter
    transfer = VarianceThreshold(threshold=10)

    #3.use fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)

    #4.calculate the pearson correalation coefficient
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pearson correlation coefficient1:\n", r1)
    r2 = pearsonr(data["revenue"], data["total_expense"])
    print("pearson correlation coefficient2:\n", r2)

    return None

if __name__ == "__main__":
    # code10: filter feature with low variance use factor_return.csv
    variance_demo()