from Exercise3 import *

from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=10, scoring=None):
    if type(train_sizes) == int:
        train_sizes = np.linspace(.1, 1.0, train_sizes)

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    if cv is not None:
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    if cv is not None:
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

    plt.legend(loc="best")
    return plt


from sklearn.base import BaseEstimator


class KernelEstimator(BaseEstimator):

    def __init__(self, k, lbda):
        self.k = k
        self.lbda = lbda

    def fit(self, X, y):
        self._X_train = X
        self._alpha, _ = learn_reg_kernel_ERM(X, y, lbda=self.lbda, k=self.k, max_iter=20000, eta=1, tol=1e-3)
        return self

    def predict(self, X):
        return predict(self._alpha, self._X_train, X, self.k)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


"""<b>Exercise 4:</b>

Vary the choices of the kernel functions, regularization parameters and kernel smoothing parameters (the $\lambda$ in the definition of $k_{\text{DTW}}$). 
"""

estimator = KernelEstimator(k2_hyp(1.0), 1.0)   # MODIFY
estimator.fit(X_train, y_train)
print("Accuracy {}".format(estimator.score(X_train, y_train)))
plot_learning_curve(estimator, 'Euclidean distance DTW, lambda = 2.0', X, y, cv=3, scoring="accuracy", train_sizes=[0.01,0.1,0.3,0.5,0.6,0.7,0.8,0.9,1.0]);
