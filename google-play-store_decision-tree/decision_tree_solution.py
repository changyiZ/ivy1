import numpy as np
import pandas as pd

df = pd.read_csv('data/googleplaystore.csv')
print('Number of apps in the dataset : ', len(df))
print(df.sample(10))

print(len(df['Installs'].unique()), "Installs")
print("\n", df['Installs'].unique())

print(len(df['Rating'].unique()), "Rating")
print("\n", df['Rating'].unique())

# - Installs : Remove + and ,

df = df[df['Installs'] != 'Free']
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int(x))
print(len(df['Installs'].unique()), "Installs")
print("\n", df['Installs'].unique())

df['Size'].replace('Varies with device', np.nan, inplace=True)
df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
           df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
           .fillna(1)
           .replace(['k', 'M'], [10 ** 3, 10 ** 6]).astype(int))
df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'), inplace=True)
# df['Installs'] = df['Installs'].apply(lambda x: float(x))

df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: int(x))

df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).split(' ')[0])
# df['Android Ver'] = df['Android Ver'].apply(lambda x: float(x))

df['Rating'].fillna(df.groupby('Category')['Rating'].transform('mean'), inplace=True)

print(df.sample(10))

from sklearn import preprocessing


def encode_features(features):
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])


encode_features(['Category', 'Last Updated', 'Content Rating', 'Android Ver'])

print(df.sample(10))

from sklearn.model_selection import train_test_split

X_all = df.drop(['Installs', 'App', 'Type', 'Current Ver', 'Genres'], axis=1)
y_all = df['Installs']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier.
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
