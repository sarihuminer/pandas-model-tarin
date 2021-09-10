import pandas as pd
import numpy as np

import seaborn as sns

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

#קבלת הסוגים

species_names = np.unique(np.array(y[np.logical_not(np.isnan(y))]))

print(species_names)
species_names_dict = {k: v for v, k in enumerate(species_names)}
print(species_names_dict)
s = pd.DataFrame(y)
y_cat = pd.get_dummies(s)
print(y_cat.sample(5))

X = pd.DataFrame(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.25,random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=42)
# אימון המודל
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2

model = Sequential()
model.add(Dense(2, input_shape=(9,), activation='softmax'))
model.compile(adam_v2.Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es], epochs=500)
y_pred = model.predict(X_test)
print(y_pred[:5])
y_pred_class = np.argmax(y_pred, axis=1)
print(y_pred_class)
