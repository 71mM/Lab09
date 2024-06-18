import numpy as np
from matplotlib import pyplot as plt

"""
<b>Exercise 1</b>:

Implement the function *d_DTW(x, x2, dist)*. The inputs x and x2 are the sequences to be compared and the parameter dist is a function on a pairs of points of the input space $X$ that outputs a real number (the distance between the pairs of points). Some code is given to help you dealing with the edge cases. The function is supposed to return the value of $d_{\text{DTW}}$ (The distance between x and x2) with the specified parameters, *not* the $k_{\text{DTW}}$.
"""


def d_DTW(x, x2, dist):
    t1, t2 = len(x), len(x2)

    # Edge Cases

    if t1 == 0 and t2 == 0:
        return 0.0
    elif (t1 == 0) or (t2 == 0):
        return np.inf

    dp = np.empty((t1 + 1, t2 + 1))
    dp[0, 0] = 0

    for i in range(1, t1 + 1):
        dp[i, 0] = np.inf

    for j in range(1, t2 + 1):
        dp[0, j] = np.inf

    for i in range(1, t1 + 1):
        for j in range(1, t2 + 1):
            cost = dist(x[i - 1], x2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j],  # Insertion
                                  dp[i, j - 1],  # Deletion
                                  dp[i - 1, j - 1])  # Match

    return dp[t1, t2]


"""Check your solution:"""
if __name__ == '__main__':
    try:
        assert d_DTW([1, 2, 3, 3], [1, 2, 3], lambda x, y: 1 if x != y else 0) == 0.0
        assert d_DTW([1, 2, 3, 4], [1, 2, 3], lambda x, y: 1 if x != y else 0) == 1.0
        assert d_DTW([1, 2, 3, 2], [1, 2], lambda x, y: 1 if x != y else 0) == 1.0
        assert d_DTW([], [1, 2], lambda x, y: 1 if x != y else 0) == np.inf
        assert d_DTW([], [], lambda x, y: 1 if x != y else 0) == 0.0
        print("There is no error in your function!")
    except AssertionError:
        print("There is an error in your function!")

"""We define three distance functions on two values $x, x' \in X$:

$d_1(x_2, x_2) = \mathbb{1}[x_1 != x_2]$, (a boolean Value should be returned)

$d_2(x_1, x_2) = (x_1 - x_2)^2$,

$d_3(x_1, x_2) = |x_1 - x_2|$,

Optional: $d_4(\Delta x_i, \Delta x'_i) = (\Delta x_i - \Delta x'_i)^2$, with
$$ \Delta x_i = \frac{1}{2}\left( x_i - x_{i-1} + \frac{x_{i+1} - x_{i-1}}{2}\right) $$
as *approximate derivates of order 2*. Note that the edge cases are $\Delta x_1 = 0$ and $\Delta x_{|x|} = x_{|x|} - x_{|x|-1}$. 

*Hint*: It's best to map the sequences $x = (x_1, \dots, x_{|x|})$ to $\Delta x = \left(\Delta x_1, \dots, \Delta x_{|x|}\right)$ and then apply $d_2$.
"""