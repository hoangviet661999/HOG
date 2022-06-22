from sklearn import svm
from preprocessing import X_train, X_test, y_train, y_test, _accuracy

KERNEL = "linear"
model = svm.SVC(kernel = KERNEL)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("accuracy train : ",_accuracy(y_train_pred, y_train), "%")

print("predict test :")
print(y_test_pred)
print ("label test : ")
print(y_test)

print("accuracy test : ", _accuracy(y_test_pred, y_test), "%")
