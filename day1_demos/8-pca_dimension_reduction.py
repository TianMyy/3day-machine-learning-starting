from sklearn.decomposition import PCA

def pca_demo():
    """
    PCA dimension reduction
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]

    #1.instantiate a converter
    transfer = PCA(n_components=0.95)

    #2.use fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None

if __name__ == "__main__":
    # code11: PCA dimension reduction
    pca_demo()
