import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


def encode_and_bind(original_dataframe, index_feature_to_encode):
    # pd.get_dummies(X_train[2])
    dummies = pd.get_dummies(original_dataframe[index_feature_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return (res)


# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


def create_confusion_matrix(actual=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1], predicted=[1, 0, 0, 1, 0, 0, 0, 1, 0, 0]):
    # confusion matrix in sklearn
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    # predicted values
    # [1,0,0,1,0,0,1,0,0,1]
    # [1,0,0,1,0,0,0,1,0,0]
    # confusion matrix
    matrix = confusion_matrix(actual, predicted, labels=[1, 0])
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1, 0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(actual, predicted, labels=[1, 0])
    print('Classification report : \n', matrix)


excel_file = 'WikiRef-input.xlsx'
# data = pd.read_excel(excel_file)
# dataset = data
# dataset = pd.read_csv('WikiRef-input.xlsx')
dataset = pd.read_excel('WikiRef-input.xlsx')
dataset = dataset.dropna()
sns.set(style='ticks')
sns.pairplot(dataset.iloc[:, 0:11], hue='isRefQK')
# הפרדה על מנת לאמן את הנתונים
x = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 8].values
# format all fields as string
X = x.astype(str)
# reshape target to be a 2d array
y = y.reshape((len(y), 1))
# קבלת הסוגים

species_names = np.unique(np.array(y[np.logical_not(np.isnan(y))]))

print(species_names)
species_names_dict = {k: v for v, k in enumerate(species_names)}
print(species_names_dict)
s = pd.DataFrame(y)
y_cat = pd.get_dummies(s)
print(y_cat.sample(5))

X = pd.DataFrame(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=42)
# אימון המודל

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

clf = RandomForestClassifier()
clf.fit(X_train_enc, y_train_enc)
print(clf.predict(np.nan_to_num(X_test_enc)))
print(y_test)
print(clf.score(np.nan_to_num(X_test_enc), y_test_enc))
create_confusion_matrix(y_train_enc, y_test_enc)
#create_confusion_matrix(y_test_enc, np.nan_to_num(X_test_enc))
# clf.score(np.nan_to_num(X_test_enc), y_test_enc)
#### you need tensorflow 2.2 or 3.3 to run it###
import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import adam_v2

# model = Sequential()
# model.add(Dense(2, input_shape=(9,), activation='softmax'))
# model.compile(adam_v2.Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# from keras.callbacks import EarlyStopping

# es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

# model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es], epochs=500)
# y_pred = model.predict(X_test)
# print(y_pred[:5])
# y_pred_class = np.argmax(y_pred, axis=1)
# print(y_pred_class)

###end tnsoflow 2.2 -3.3 v##
