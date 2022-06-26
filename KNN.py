from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from preprocessing import X_train, X_test, y_train, y_test, y_test_dict, _accuracy

N_NEIGHBOR = 3

#train model KNN, k =3
model = KNeighborsClassifier(n_neighbors = N_NEIGHBOR)
model.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=N_NEIGHBOR, p=2,
           weights='uniform')

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

uniq_labels = list(y_test_dict.values())
# print(classification_report(y_test, y_test_pred, target_names = uniq_labels))
print("accuracy train : ",_accuracy(y_train_pred, y_train), "%")

print("predict test :")
print(y_test_pred)
print ("label test : ")
print(y_test)

print("accuracy test : ", _accuracy(y_test_pred, y_test), "%")

