import matplotlib.pyplot as plt
import pandas as pd
import plotly.offline as py
import seaborn as sns
import numpy as np

py.init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/googleplaystore.csv')
print('Number of apps in the dataset : ', len(df))
df.sample(10)


# 打印某个属性的分布信息
def print_info(feature):
    print(len(df[feature].unique()), feature)
    print("\n", df[feature].unique())


# 打印整个表格的基本信息
df.info()

df['Rating'] = df['Rating'].fillna(df['Rating'].median())
print_info('Rating')

index = df[df['Rating'] == 19.].index
print(df.loc[index])

df = df.drop(index)
print_info('Rating')

# def visualFeatureCounting(feature):
#     g = sns.countplot(x=feature, data=df, palette="Set1")
#     g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
#     g
#     plt.title('Count of app in each ' + feature, size=20)
#
#
# visualFeatureCounting('Rating')

# Removing NaN values
df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]
df.info()

print_info('Content Rating')


def map_content_rating(content_rating):
    if 'Teen' in content_rating:
        return 1
    elif 'Everyone 10+' in content_rating:
        return 2
    elif 'Mature 17+' in content_rating:
        return 3
    elif 'Adults only 18+' in content_rating:
        return 4
    else:
        return 0


# Encode Content Rating features
df['Content Rating'] = df['Content Rating'].map(map_content_rating)
print_info('Content Rating')


def map_reviews(number):
    number = int(number)
    if number < 10:
        return 1
    elif number < 100:
        return 2
    elif number < 1000:
        return 3
    elif number < 10000:
        return 4
    elif number < 100000:
        return 5
    elif number < 1000000:
        return 6
    elif number < 10000000:
        return 7
    else:
        return 8


df['Reviews'] = df['Reviews'].map(map_reviews)

print_info('Reviews')


# visualFeatureCounting('Reviews')


# scaling and cleaning size of installation
def map_size(size):
    if 'M' in size:
        x = float(size[:-1])
        if x < 5.0:
            return 1
        elif x < 10.0:
            return 2
        elif x < 20.0:
            return 3
        elif x < 50.0:
            return 4
        elif x < 100.0:
            return 5
        else:
            return 6
    else:
        return 0


df['Size'] = df['Size'].map(map_size)

print_info('Size')

# visualFeatureCounting('Size')

print_info('Android Ver')


# scaling and cleaning size of installation
def map_version(version):
    version = str(version)
    if version.startswith("1."):
        return 1
    elif version.startswith("2."):
        return 2
    elif version.startswith("3."):
        return 3
    elif version.startswith("4."):
        return 4
    elif version.startswith("5."):
        return 5
    elif version.startswith("6."):
        return 6
    elif version.startswith("7."):
        return 7
    elif version.startswith("8."):
        return 8
    else:
        return 0


df['Android Ver'] = df['Android Ver'].map(map_version)

print_info('Android Ver')
# visualFeatureCounting('Android Ver')

# get_dummies creates a new dataframe which consists of zeros and ones
df['App'] = pd.get_dummies(df['App'])
df['Last Updated'] = pd.get_dummies(df['Last Updated'])

# Encode Category features
le = preprocessing.LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Price cealning
df['Price'] = df['Price'].apply(lambda x: x.strip('$'))


def map_installs(number):
    number = int(number)
    if number < 1000000:
        return 0
    elif number < 10000000:
        return 1
    else:
        return 2


# Installs cealning
df['Installs'] = df['Installs'].apply(lambda x: x.strip('+').replace(',', ''))
df['Installs'] = df['Installs'].map(map_installs)

print_info('Installs')

# visualFeatureCounting('Installs')

X_all = df.drop(['Installs', 'Type', 'Current Ver', 'Genres'], axis=1)
y_all = df['Installs']

print('origin: ')
print(X_all.sample(5))
x_scaled = StandardScaler().fit_transform(X_all.values)
X_all = pd.DataFrame(x_scaled)
print('scale: ')
print(X_all.sample(5))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=10)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

# # Choose the type of classifier.
# clf = RandomForestClassifier()
#
# # Choose some parameter combinations to try
# parameters = {'n_estimators': [4, 6, 9],
#               'max_features': ['log2', 'sqrt', 'auto'],
#               'criterion': ['entropy', 'gini'],
#               'max_depth': [2, 3, 5, 10],
#               'min_samples_split': [2, 3, 5],
#               'min_samples_leaf': [1, 5, 8]
#               }
#
# # Type of scoring used to compare parameter combinations
# acc_scorer = make_scorer(accuracy_score)
#
# # Run the grid search
# grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
# grid_obj = grid_obj.fit(X_train, y_train)
#
# # Set the clf to the best combination of parameters
# clf = grid_obj.best_estimator_
#
# # Fit the best algorithm to the data.
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
# print(accuracy_score(y_test, predictions))
#
# clf = AdaBoostClassifier(n_estimators=100)
#
# # Choose some parameter combinations to try
# parameters = {'n_estimators': [4, 6, 9],
#               'algorithm': ['SAMME', 'SAMME.R']}
#
# # Type of scoring used to compare parameter combinations
# acc_scorer = make_scorer(accuracy_score)
#
# # Run the grid search
# grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
# grid_obj = grid_obj.fit(X_train, y_train)
#
# # Set the clf to the best combination of parameters
# clf = grid_obj.best_estimator_
#
# # Fit the best algorithm to the data.
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
# print(accuracy_score(y_test, predictions))


from sklearn.metrics import classification_report
from sklearn.svm import SVC

# kernels = ['linear', 'rbf', 'poly']
# for kernel in kernels:
#     clf = svm.SVC(kernel=kernel).fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     print(kernel, accuracy_score(y_test, predictions))


# def svc_param_selection(X, y, nfolds):
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     param_grid = {'C': Cs, 'gamma': gammas}
#     grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_


clf = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='rbf', gamma='scale'))])
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

# Set the parameters by cross-validation
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]

tuned_parameters = [
    # {'svm__kernel': ['linear'], 'svm__C': Cs},
    {'kernel': ['rbf'], 'gamma': gammas, 'C': Cs}
    # {'kernel': ['poly'], 'degree': [0, 1, 2, 3, 4, 5, 6]}
]

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring=acc_scorer)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

# Set the clf to the best combination of parameters
clf = clf.best_estimator_

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

degrees = [0, 1, 2, 3, 4, 5, 6]
for degree in degrees:
    clf = svm.SVC(kernel='poly', degree=degree)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Poly with degree:', degree, accuracy_score(y_test, predictions))

# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()
