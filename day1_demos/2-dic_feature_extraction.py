from sklearn.feature_extraction import DictVectorizer

def dict_demo():
    """
    Dict feature extraction
    :return:
    """
    data =[{'city':'Beijing','temperature':100},
           {'city':'Shanghai','temperature':60},
           {'city':'Shenzhen','temperature':30}]
    #1.instantiate a convertor
    transfer = DictVectorizer()

    #2.use fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("feature names:\n", transfer.get_feature_names())

    return None

if __name__ == "__main__":
    # code2: dict feature extraction
    dict_demo()
