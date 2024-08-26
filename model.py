import pandas as pd
transaction = pd.read_csv('data.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = transaction.copy()
target = 'isFraud'
encode = ['cardPresent','transactionType','matchCVV','posEntryMode']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'no':0, 'yes':1}
def target_encode(val):
    return target_mapper[val]

df['isFraud'] = df['isFraud'].apply(target_encode)

# Import library
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Separating X and y
x = df.drop('isFraud', axis=1)
y = df['isFraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

clf = RandomForestClassifier(random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Saving the model
import pickle
pickle.dump(clf, open('transaction_clf.pkl', 'wb'))
