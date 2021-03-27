from sklearn.feature_extraction.text import CountVectorizer
import jieba

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

if __name__ == "__main__":
    # code3: text feature extraction: CountVectorizer
    count_demo()

    # code4: chinese text feature extraction: CountVectorizer
    #count_chinese_demo()

    #code5: chinese text feature extraction, with stop_words, divide words automatically: CountVectorizer
    #count_chinese_demo2()

    # code6: cutting chinese text
    #cut_word("我爱北京天安门")