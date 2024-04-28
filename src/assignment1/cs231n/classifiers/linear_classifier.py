from __future__ import print_function

from builtins import object, range

import numpy as np
from past.builtins import xrange

from ..classifiers.linear_svm import *
from ..classifiers.softmax import *


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.

        確率的勾配降下法を用いて，この線形分類器を学習する．

        入力
        - X: 入力： X: 訓練データを含む (N, D) 形式の numpy 配列．
          個の訓練サンプルがある．
        - y: 訓練ラベルを含む shape (N,) の numpy 配列; y[i] = c
          は、X[i] が C 個のクラスに対して 0 <= c < C のラベルを持つことを意味する。
        - learning_rate: (float) 最適化のための学習率。
        - reg: (float) 正則化の強さ。
        - num_iters: (integer) 最適化のステップ数。
        - batch_size: (integer) 各ステップで使用する学習例数。
        - verbose: (boolean) true の場合、最適化の進行状況を表示します。

        出力：
        各トレーニング反復における損失関数の値を含むリスト。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            # --------------------------------------------------------------------- #
            # この勾配降下のラウンドで使用するために、                                  #
            # 訓練データとそれに対応するラベルからbatch_sizeの要素をサンプリングする。    #
            # データをX_batchに、対応するラベルをy_batchに格納する。                    #
            # サンプリング後、X_batchは形状(batch_size, dim)を持ち、                   #
            # y_batchは形状(batch_size,)を持つ。                                      #
            #                                                                       #
            # ヒント：インデックスを生成するために np.random.choice を使用します。       #
            # サンプリングは、置換なしでサンプリングするよりも、                         #
            # 置換でサンプリングする方が速いです。                                     #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # num_trainの範囲からbatch_sizeの数だけランダムに整数を選択
            index = np.random.choice(num_train, batch_size)
            X_batch = X[index]
            y_batch = y[index]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            # --------------------------------------------------------------------- #
            # 勾配と学習率を用いて重みを更新する。                                     #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 勾配の負の方向に更新することで損失関数を最小化する
            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.

        この線形分類器の学習された重みを使って、データ点のラベルを予測する。
        を予測する。

        入力
        - X: 入力： X: 訓練データを含む (N, D) 形式の numpy 配列。
          個の訓練サンプルが存在する。

        戻り値
        - y_pred: y_pred: X のデータに対する予測ラベル。
          長さ N の 1 次元配列であり、各要素は予測されるクラスである。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        # ======================================================================= #
        # このメソッドを実装する。予測されたラベルを y_pred に格納します。             #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 計算されたスコアの中から最も高いスコアを持つクラスのインデックスを選択
        y_pred = np.argmax(X.dot(self.W), axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
