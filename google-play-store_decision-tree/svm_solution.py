import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('data/googleplaystore.csv')
print('Number of apps in the dataset : ', len(df))
print(df.sample(3))
df.info()


def print_info(feature):
    print(len(df[feature].unique()), feature)
    print("\n", df[feature].unique())


df['Rating'] = df['Rating'].fillna(df['Rating'].median())
index = df[df['Rating'] == 19.].index
df = df.drop(index)

df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]


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

# - Installs : Remove + and ,
# df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
# df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
# df['Installs'] = df['Installs'].apply(lambda x: int(x))
# df['Installs'] = df['Installs'].apply(lambda x: float(x))

df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))
df['Size'] = df['Size'].fillna(df['Size'].mean())


def map_size(size):
    if size < 5.0:
        return 1
    elif size < 10.0:
        return 2
    elif size < 20.0:
        return 3
    elif size < 50.0:
        return 4
    elif size < 100.0:
        return 5
    else:
        return 6


df['Size'] = df['Size'].map(map_size)


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

df['Price'] = df['Price'].apply(lambda x: x.strip('$'))


def map_price(price):
    price = float(price)
    if price > 10.0:
        return 2
    elif price > 0.0:
        return 1
    else:
        return 0


df['Price'] = df['Price'].map(map_price)


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

# get_dummies creates a new dataframe which consists of zeros and ones
df['App'] = pd.get_dummies(df['App'])
df['Last Updated'] = pd.get_dummies(df['Last Updated'])

# Encode Category features
le = preprocessing.LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])


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

X_all.info()

print('origin: ')
print(X_all.sample(5))
x_scaled = MinMaxScaler().fit_transform(X_all.values)
X_all = pd.DataFrame(x_scaled)
print('scale: ')
print(X_all.sample(5))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=10)


def svm_cv(kernel, params):
    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)
    clf = GridSearchCV(svm.SVC(kernel=kernel), params, scoring=acc_scorer)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)

    # Set the clf to the best combination of parameters
    clf = clf.best_estimator_
    predictions = clf.predict(X_test)
    print(kernel, accuracy_score(y_test, predictions))


# Set the parameters by cross-validation
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
degrees = [0, 1, 2, 3, 4, 5, 6]

svm_cv('linear', [{'C': Cs}])
svm_cv('rbf', [{'gamma': gammas, 'C': Cs}])
svm_cv('poly', [{'degree': degrees}])