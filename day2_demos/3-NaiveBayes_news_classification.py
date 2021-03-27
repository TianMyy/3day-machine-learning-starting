from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    '''
    use Naive Bayes to classify the news
    :return:
    '''

    # 1)get data
    news = fetch_20newsgroups(subset='all')

    # 2)divide dataset
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, random_state=0)

    # 3)feature engineering: tf-idf text processing
    # instantiate a transformer
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4)Naive Bayes algorithm estimator
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5)model assessment
    # method 1:compare the true value and predicted value straightly
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true and predicted value:\n", y_test == y_predict)
    # method 2:calculate the accuracy rate
    score = estimator.score(x_test, y_test)
    print("accuracy_score:\n", score)

    return None


if __name__ == "__main__":
    # code 3: use Naive Bayes to classify the news
    nb_news()
