from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W


    ソフトマックス損失関数、素朴な実装（ループあり）

    入力はD次元で、C個のクラスがあり、N個の例のミニバッチを操作する。

    入力：
    - W: 重みを含む形状 (D, C) の numpy 配列。
    - X: データのミニバッチを含む (N, D) 形式の numpy 配列。
    - y: 訓練ラベルを含む Numpy 形式の配列 (N,); y[i]=cは、X[i]のラベルがcであることを意味する。
    - reg: (float) 正則化の強さ

    出力(タプル):
    - 単一フロートとしての損失
    - 重みWに対する勾配
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # ===========================================================================#
    # TODO: ソフトマックス損失とその勾配を明示的ループを使って計算する。
    # 損失はlossに、勾配はdWに格納する。ここで注意しないと、
    # 数値が不安定になりやすい。正則化を忘れないでください！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]  # num samples

    for i in range(num_train):
        y_hat = X[i].dot(W)  # raw scores vector
        y_exp = np.exp(y_hat - y_hat.max())  # numerically stable exponent vector
        softmax = y_exp / y_exp.sum()  # pure softmax for each score
        loss -= np.log(softmax[y[i]])  # append cross-entropy
        softmax[y[i]] -= 1  # update for gradient
        dW += np.outer(X[i], softmax)  # gradient

    loss = loss / num_train + reg * np.sum(W**2)  # average loss and regularize
    dW = dW / num_train + 2 * reg * W  # finish calculating gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # ========================================================================= #
    # TODO: 明示的なループを使わずに、ソフトマックス損失とその勾配を計算する
    # 損失はlossに、勾配はdWに格納する。ここで注意しないと、数値が不安定になりやすい。
    # 正則化を忘れないでください！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    Y_hat = X @ W  # raw scores matrix

    P = np.exp(Y_hat - Y_hat.max())  # numerically stable exponents
    P /= P.sum(axis=1, keepdims=True)  # row-wise probabilities (softmax)

    loss = -np.log(P[range(num_train), y]).sum()  # sum cross entropies as loss
    loss = loss / num_train + reg * np.sum(W**2)  # average loss and regularize

    P[range(num_train), y] -= 1  # update P for gradient
    dW = X.T @ P / num_train + 2 * reg * W  # calculate gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
