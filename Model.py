from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from numpy.random import seed


class SVM:
    def __init__(self, X, y, X_test, y_test):
        seed(2021)
        self.model = SVC(C=1.0,
                         kernel='linear',
                         degree=4,
                         gamma='scale',
                         coef0=0.0,
                         shrinking=True,
                         probability=True,
                         tol=0.001,
                         cache_size=200,
                         class_weight={1: 1, 2: 2.7},
                         verbose=False,
                         max_iter=-1,
                         decision_function_shape='ovr',
                         break_ties=False,
                         random_state=15)
        std = StandardScaler()
        self.X = std.fit_transform(X)
        self.y = y
        self.X_test = std.transform(X_test)
        self.y_test = y_test

    def svm_train(self):
        self.model.fit(self.X, self.y)
        return self.model.coef_

    def svm_predict(self):
        return self.model.predict(self.X_test)

    def svm_predict_confidence(self):
        return self.model.predict_proba(self.X_test)

    def svm_score(self):
        return self.model.score(self.X_test, self.y_test)