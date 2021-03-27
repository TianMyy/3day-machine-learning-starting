from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd

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

def count_demo():
    """
    text feature extraction: CountVectorizer
    :return:
    """
    data =["life is short, i like python",
           "life is too long, i dislike python"]
    #1.instantiate a converter
    transfer = CountVectorizer(stop_words=["is", "too"])

    #2.use fit_trasform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("feature names:\n", transfer.get_feature_names())

    return None

def count_chinese_demo():
    """
    chinese text feature extraction: CountVectorizer
    :return:
    """
    data =["我爱北京天安门",
           "天安门上太阳升"]
    #1.instantiate a converter
    transfer = CountVectorizer()

    #2.use fit_trasform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("feature names:\n", transfer.get_feature_names())

    return None

def cut_word(text):
    """
    cutting chinese text:"我爱北京天安门" -> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    a = " ".join(list(jieba.cut(text)))
    print(a)

    return a

def count_chinese_demo2():
    """
    chinese text feature extraction, with stop_words, divide words automatically: CountVectorizer
    :return:
    """
    data =["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天",
           "我们看到的从很远星系来的光是几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去",
           "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的食物相联系。"]

    #1.cutting chinese words
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)

    #2.instantiate a converter
    transfer = CountVectorizer(stop_words=["一种", "所以"])

    #3.use fit_trasform()
    data_final = transfer.fit_transform(data_new)
    print("data_final:\n", data_final.toarray())
    print("feature names:\n", transfer.get_feature_names())

    return None

def tfidf_demo():
    """
    use TF-IDF method to process text feature extraction
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天",
            "我们看到的从很远星系来的光是几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的食物相联系。"]

    # 1.cutting chinese words
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)

    # 2.instantiate a converter
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    # 3.use fit_trasform()
    data_final = transfer.fit_transform(data_new)
    print("data_final:\n", data_final.toarray())
    print("feature names:\n", transfer.get_feature_names())

    return None

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
    # # code1: sklearn dataset usage
    # datasets_demo()

    # # code2: dict feature extraction
    # dict_demo()
    #
    # # code3: text feature extraction: CountVectorizer
    # count_demo()
    #
    # # code4: chinese text feature extraction: CountVectorizer
    # count_chinese_demo()
    #
    # #code5: chinese text feature extraction, with stop_words, divide words automatically: CountVectorizer
    # count_chinese_demo2()
    #
    # # code6: cutting chinese text
    # cut_word("我爱北京天安门")
    #
    # # code7: use TF-IDF method to process text feature extraction
    # tfidf_demo()

    # # code8: normalization use dating.txt
    # minmax_demo()

    # # code9: standardization use dating.txt
    # stand_demo()

    # # code10: filter feature with low variance use factor_return.csv
    # variance_demo()

    # code11: PCA dimension reduction
    pca_demo()

