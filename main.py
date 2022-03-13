import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("iris.data")

print("Before replacing:\n", df.head())
df["class"] = df["class"].replace("Iris-setosa", 0)
df["class"] = df["class"].replace("Iris-versicolor", 1)
df["class"] = df["class"].replace("Iris-virginica", 2)
print("After replacing:\n", df.head())

predict = "class"

x = np.array(df.drop(columns = [predict]))
y = np.array(df[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)



# Use SVC without any parameters
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
#96.666667%
print("SVM(without kernel):", format(acc, "%"))


# use one kernel: linear with soft/hard margin or poly with degree
clf = svm.SVC(kernel = "linear", C=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
# 100.000000%
print("SVM(with kernel):", format(accuracy, "%"))