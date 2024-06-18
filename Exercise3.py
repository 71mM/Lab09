from Exercise2 import *

"""<b>Exercise 3</b>:

Adjust the old code (Lab 06) to actually learn the kernel linear regression. Note that there is a new parameter $k$ that encodes the kernel function. Note that lbda is not the $\lambda$ used in the definition of $k$, but the regularization coefficient (as before). Note also that the learning rate $\alpha$ has been renamed to $\eta$, because $\alpha$ coincides with the dual coefficients (see lecture).
Also make sure to return the Gram matrix $K$ together with the weight vector $w$ (or $\alpha$), as it is costly to compute and needed for the inference.
"""


def learn_reg_kernel_ERM(X, y, lbda, k, loss=hinge_loss, reg=L2_reg, max_iter=200, tol=0.001, eta=1., verbose=False):
    """Kernel Linear Regression (default: kernelized L_2 SVM)
    X -- data, each row = instance
    y -- vector of labels, n_rows(X) == y.shape[0]
    lbda -- regularization coefficient lambda
    k -- the kernel function
    loss -- loss function, returns vector of losses (for each instance) AND the gradient
    reg -- regularization function, returns reg-loss and gradient
    max_iter -- max. number of iterations of gradient descent
    tol -- stop if norm(gradient) < tol
    eta -- learning rate
    """
    num_features = X.shape[0]

    g_old = None

    K = build_dtw_gram_matrix(X, X, k)
    w = np.random.randn(num_features)  # modify; hint: The matrix K should be used and w has as many entries as training examples.

    for t in range(max_iter):
        h = np.dot(K, w)
        l, lg = loss(h, y)

        if verbose:
            print('training loss: ' + str(np.mean(l)))

        r, rg = reg(w, lbda)
        g = lg + rg

        if g_old is not None:
            eta = 0.9 ** t

        w = w - eta * g
        if (np.linalg.norm(eta * g) < tol):
            break
        g_old = g

    return w, K


"""The adjusted inference function is given as (for binary classification):"""


def predict(alpha, X, X_train, k):
    K = build_dtw_gram_matrix(X_train, X, k)
    y_pred = np.dot(K, alpha)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    return y_pred


"""## 3. DTW Kernel SVM in Action

Now we put our results from section $1$ and $2$ together to use a kernelized SVM for a classification task on sequence data.
"""
from sklearn.model_selection import train_test_split
from scipy.io import loadmat  # for matlab *.mat format, for modern once need to install hdf5

file_path = "laser_small.mat"  # file path for multi os support
mat = loadmat(file_path)

X = mat['X']
y = mat['Y'].reshape(50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print(X.shape, y.shape)

"""We have only 50 training instances and thus only go for a simple train-test-split (we cannot afford a simple train-val-test-split). If we try several kernels, we are actually tuning a hyperparameter and thus are fitting on the test set. The solution to this problem would be the nested cross-validation procedure, which we learn in the evaluation lecture."""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape, X_test.shape)

alpha, K = learn_reg_kernel_ERM(X_train, y_train, lbda=1, k=k2, max_iter=20000, eta=1, tol=1e-3, verbose=True)

"""And evaluation of the model."""

y_pred = predict(alpha, X_train, X_train, k2)
print("Training Accuracy: {}".format(np.mean(y_train == y_pred)))
print("Test Accuracy: {}".format(np.mean(y_test == predict(alpha, X_train, X_test, k2))))
print("Shape of alpha {}".format(alpha.shape))

"""We see that the training accuracy is far better than the test accuracy. This *could* - but does not have to - mean that we are overfitting.

Vary the choices of the kernel functions, regularization parameters and kernel smoothing parameters (the $\lambda$ in the definition of $k_{\text{DTW}}$). In the rest of the notebook you learn how you can draw learning curves we have discussed in the tutorial. To be able to use the helper function, the estimator needs to be wrapped in a scikit-learn conform way. You can find and use the example class KernelEstimator.
"""


