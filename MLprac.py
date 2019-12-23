import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
# %matplotlib inline
df = pd.read_csv('yourdata.csv')

# extract the columns required
data = df[['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14']]

# data normalization using standard scalar
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
data = scaler.transform(data)
print(data)

# pyplot after normalization of data
pyplot.hist(data)
pyplot.show()

# Feature Extraction using Mutual Information
array = df.values
X = array[:,0:14]
# target data
Y = array[:,13]
mi = mutual_info_regression(X, Y)
mi = pd.Series(mi)
print(mi.sort_values(ascending=False))
print(mi.sort_values(ascending=False).plot.bar(figsize=(2.5, 1)))


 # code for KNN Algorithm using all channels
A = df.iloc[:, 0:14].values
B = df.iloc[:, 14].values
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.20)
scaler = StandardScaler()
scaler.fit(A_train)

A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
from sklearn.neighbors import KNeighborsClassifier

# with 5 nearest neigbours
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(A_train, B_train)
B_pred = classifier.predict(A_test)
print(confusion_matrix(B_test, B_pred))
print(classification_report(B_test, B_pred))
acc = accuracy_score(B_test, B_pred)
print(acc)

