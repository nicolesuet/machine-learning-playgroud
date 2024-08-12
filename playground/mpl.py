from sklearn import metrics, datasets, neural_network
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_features=5, n_classes=2, n_samples=100)
x_train, x_test, y_train, y_test = train_test_split(X, y)

print("Shape x_train, x_test:", x_train.shape, x_test.shape)

clf = neural_network.MLPClassifier(max_iter=5000)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion matrix:")
print(confusion_matrix)