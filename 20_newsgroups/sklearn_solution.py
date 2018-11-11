import datetime
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

newsgroups_train = fetch_20newsgroups(subset='all',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'))

train_texts = newsgroups_train['data']
train_labels = newsgroups_train['target']
test_texts = newsgroups_test['data']
test_labels = newsgroups_test['target']
print(len(train_texts), len(test_texts))

stemmer = EnglishStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def do_classify(tag, classifier, vectorizer=TfidfVectorizer()):
    text_clf = Pipeline([('tfidf', vectorizer),
                         (tag, classifier)])
    print(tag + " start...")
    start_time = datetime.datetime.now()
    text_clf = text_clf.fit(train_texts, train_labels)
    end_time = datetime.datetime.now()
    print("   training time: " + str(end_time - start_time))
    predicted = text_clf.predict(test_texts)
    end_time = datetime.datetime.now()
    print("   classification time: " + str(end_time - start_time))
    print("   accuracy: ", np.mean(predicted == test_labels))


vectorizer = TfidfVectorizer(stop_words="english", analyzer=stemmed_words)
do_classify("MultinomialNB", MultinomialNB(), vectorizer)
do_classify("KNeighborsClassifier", KNeighborsClassifier(), vectorizer)

do_classify("RandomForestClassifier", RandomForestClassifier(n_estimators=8), vectorizer)
do_classify("AdaBoostClassifier", AdaBoostClassifier(), vectorizer)
do_classify("DecisionTreeClassifier", DecisionTreeClassifier(), vectorizer)
