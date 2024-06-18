from Exercise1 import *
"""
<b>Exercise 2</b>:

Implement the missing distance metrics.
"""

def d1(x, x2):
    return 1 if x != x2 else 0

def d2(x, x2):
    return (x - x2) ** 2

def d3(x, x2):
    return abs(x - x2)

"""The following code lifts the distance metrics to maps that map a given hyperparameter $\lambda$ return the corresponding kernel function $k_{\text{DTW}}$."""

k1_hyp, k2_hyp, k3_hyp = [lambda lmbd: (lambda x, x2: np.exp(-lmbd * d_DTW(x, x2, d))) for d in [d1, d2, d3]]

k1 = k1_hyp(2.0)
k2 = k2_hyp(2.0)
k3 = k3_hyp(2.0)

"""The following code computes the Gram matrix $K$ with respect to the kernel $k$ (a parameter) and the data $xs$ (another parameter), see slide 28 and 29 in Kernel Methods lecture."""


def build_dtw_gram_matrix(xs, x2s, k):
    """
    xs: collection of sequences (vectors of possibly varying length)
    x2s: the same, needed for prediction
    k: a kernel function that maps two sequences of possibly different length to a real
    The function returns the Gram matrix with respect to k of the data xs.
    """
    t1, t2 = len(xs), len(x2s)
    K = np.empty((t1, t2))

    for i in range(t1):
        for j in range(i, t2):
            K[i, j] = k(xs[i], x2s[j])
            if i < t2 and j < t1:
                K[j, i] = K[i, j]

    return K


build_dtw_gram_matrix([[1, 2], [2, 3]], [[1, 2, 3], [4]], k1)

"""## 2. Kernel SVM

Now we implement the training algorithm for kernel SVMs. We adjust the ERM learning algorithm from the linear classification lab. First we are reusing the code for the $\mathcal{L}_2$-regularizer and the hinge loss.
"""


def L2_reg(w, lbda):
    return 0.5 * lbda * (np.dot(w.T, w)), lbda * w


def hinge_loss(h, y):
    n = len(h)
    l = np.maximum(0, np.ones(n) - y * h)
    g = -y * (h > 0)
    return l, g
