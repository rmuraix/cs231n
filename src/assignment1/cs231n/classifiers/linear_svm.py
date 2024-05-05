from builtins import range
from random import shuffle

import numpy as np
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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

    ----

    構造化SVM損失関数、素朴な実装（ループあり）。

    入力はD次元で、C個のクラスがあり、N個の例のミニバッチを操作する。

    入力
    - W: 重みを含む形状 (D, C) の numpy 配列。
    - X: データのミニバッチを含む shape (N, D) の numpy 配列。
    - y: y: 学習ラベルを含む shape (N,) の numpy 配列; y[i] = c は
      X[i] がラベル c を持つことを意味する.
    - reg: (float) 正則化の強さ

    出力（タプル）：
    - 単一 float としての loss
    - 重み W に対する勾配; W と同じ形の配列
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # i 番目のデータと重みを計算しスコアを出す
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # すべてのクラスに対して損失を計算
        for j in range(num_classes):
            # 正しいクラスのスコアはスキップ
            if j == y[i]:
                continue
            # 正しいクラスのスコアと他のクラスのスコアとの差を計算
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # ================================================================
                # 間違ったクラスの勾配を増加させ、正しいクラスの勾配を減少させることで、
                # 損失を最小化する方向にパラメータを更新
                dW[:, j] += X[i]  # 間違ったクラスの勾配を更新(増加)
                dW[:, y[i]] -= X[i]  # 正しいクラスの勾配を更新（減少）
                # ================================================================

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # 現在、損失は全トレーニング例の合計であるが、
    # 代わりに平均にしたいので、num_trainで割る。
    loss /= num_train

    # Add regularization to the loss.
    # 損失に正則化を加える。
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # ------------------------------------------------------------------------- #
    # 損失関数の勾配を計算し、dWに格納する。
    # 最初に損失を計算してから導関数を計算するよりも
    # 損失計算と同時に導関数を計算する方が簡単かもしれません。その結果、
    # 勾配を計算するために上記のコードの一部を修正する必要があるかもしれません。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 平均勾配を取得
    dW /= num_train

    # 正則化項の勾配を加える
    # L2正則化（L2正則化項の微分が2 * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # ------------------------------------------------------------------------- #
    # 構造化SVM損失のベクトル化バージョンを実装し、結果をlossに格納する。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # サンプル数
    N = len(y)

    scores = X.dot(W)

    # 正解ラベルのスコア
    scores_true = scores[range(N), y][:, np.newaxis]
    # 各スコアに対するマージン
    margins = np.maximum(0, scores - scores_true + 1)
    # L2正則化された損失
    # N-1 は正解ラベルのスコアを除外するため
    loss = margins.sum() / N - 1 + reg * np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    # ------------------------------------------------------------------------- #
    # 構造化SVM損失の勾配のベクトル化バージョンを実装し、結果をdWに格納する。
    #
    # ヒント：ゼロから勾配を計算する代わりに、損失を計算するために
    # 使用したいくつかの中間値を再利用すると、勾配の計算が簡単になるかもしれません。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # スコアに対する初期勾配
    # マージンが0より大きい場合は1、それ以外は0
    dW = (margins > 0).astype(int)

    # 正解ラベルを含むように勾配を更新

    # 各サンプルの正解クラスに対応するdWの要素から、
    # そのサンプルの全クラスにわたる勾配の合計を引く
    # これにより、正解クラスのスコアが他のクラスのスコアよりも大きくなるように
    # 勾配が調整される。
    dW[range(N), y] -= dW.sum(axis=1)
    # Wに対する勾配
    dW = X.T @ dW / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
