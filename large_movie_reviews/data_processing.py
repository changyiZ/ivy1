from pandas import DataFrame
from os import listdir
from os.path import join
from sklearn.utils import shuffle


def read_text(path):
    return open(path, mode='r', encoding='UTF-8').read()


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
    ('aclImdb/train/neg/', NEG),
    ('aclImdb/test/pos/', POS),
    ('aclImdb/test/neg/', NEG)
]

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = shuffle(data)
data.to_csv('movie_reviews_data_full.csv', sep='\t')
