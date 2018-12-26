import datetime

import numpy as np
from pandas import DataFrame
from os import listdir
from os.path import join

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier


def read_text(path):
    return open(path, 'r').read()


def read_files(directory):
    for path in [join(directory, f) for f in listdir(directory)]:
        yield path, read_text(path)


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame


POS = 'positive'
NEG = 'negative'

SOURCES = [
    ('aclImdb/train/pos/', POS),
    ('aclImdb/train/neg/', NEG)
]

train = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    train = train.append(build_data_frame(path, classification))

train = train.reindex(np.random.permutation(train.index))
print(train.sample(10))

# k_fold = KFold(n_splits=6)
# scores = []
# confusion = np.array([[0, 0], [0, 0]])
# for train_indices, test_indices in k_fold.split(data):
#     train_text = data.iloc[train_indices]['text'].values
#     train_y = data.iloc[train_indices]['class'].values
#
#     test_text = data.iloc[test_indices]['text'].values
#     test_y = data.iloc[test_indices]['class'].values
#
#     pipeline.fit(train_text, train_y)
#     predictions = pipeline.predict(test_text)
#
#     confusion += confusion_matrix(test_y, predictions)
#     score = f1_score(test_y, predictions, pos_label=POS)
#     scores.append(score)
#
# print('Total review classified:', len(data))
# print('Score:', sum(scores) / len(scores))
# print('Confusion matrix:')
# print(confusion)

TEST_SOURCES = [
    ('aclImdb/test/pos/', POS),
    ('aclImdb/test/neg/', NEG)
]

test = DataFrame({'text': [], 'class': []})
for path, classification in TEST_SOURCES:
    test = test.append(build_data_frame(path, classification))


def do_classify(tag, classifier, vectorizer=TfidfVectorizer()):
    text_clf = Pipeline([('tfidf', vectorizer),
                         (tag, classifier)])
    print(tag + " start...")
    start_time = datetime.datetime.now()
    text_clf = text_clf.fit(train['text'], train['class'])
    end_time = datetime.datetime.now()
    print("   training time: " + str(end_time - start_time))
    predicted = text_clf.predict(test['text'])
    end_time = datetime.datetime.now()
    print("   classification time: " + str(end_time - start_time))
    print("   accuracy: ", np.mean(predicted == test['class']))


# do_classify("MultinomialNB", MultinomialNB())
# do_classify("KNeighborsClassifier", KNeighborsClassifier())
#
# do_classify("RandomForestClassifier", RandomForestClassifier(n_estimators=8))
# do_classify("AdaBoostClassifier", AdaBoostClassifier())
# do_classify("DecisionTreeClassifier", DecisionTreeClassifier())

# stemmer = EnglishStemmer()
# analyzer = TfidfVectorizer().build_analyzer()
#
#
# def stemmed_words(doc):
#     return (stemmer.stem(w) for w in analyzer(doc))
#
#
# stem_vectorizer = TfidfVectorizer(analyzer=stemmed_words, ngram_range=(1, 2))
#
# do_classify("LogisticRegression", LogisticRegression(solver='liblinear'))
# do_classify("LogisticRegression", LogisticRegression(solver='liblinear'), TfidfVectorizer(ngram_range=(1, 2)))
# do_classify("LogisticRegression", LogisticRegression(solver='liblinear'), stem_vectorizer)
#
# Choose the type of classifier.
clf = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(solver='liblinear'))])

# Choose some parameter combinations to try
parameters = {'lr__C': (0.01, 0.1, 1.0, 10.0, 100.0)}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(train['text'], train['class'])

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# 打印最佳的参数
print(grid_obj.best_estimator_.get_params())
# 用最佳方案预测结果
predicted = clf.predict(test['text'])
print("   accuracy: ", np.mean(predicted == test['class']))
