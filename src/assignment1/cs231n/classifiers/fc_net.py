from builtins import object, range

import numpy as np

from ..layer_utils import *
from ..layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.

    ReLU非線形性とsoftmax損失を持つ2層全結合ニューラルネットワーク。
    softmax損失を持つ2層全結合ニューラルネットワークである。入力次元
    D、隠れ次元をHとし、C個のクラスに対して分類を行う。

    アーキテクチャはアフィン - relu - アフィン - ソフトマックスとする。

    このクラスは勾配降下を実装しないことに注意してください。
    を実行する別のソルバー・オブジェクトと相互作用する。
    最適化を実行する。

    モデルの学習可能なパラメータは、パラメータ名をマップした辞書
    self.paramsに格納され、パラメータ名をnumpyの配列にマップします。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.

        新しいネットワークを初期化する。

        入力：
        - input_dim: 入力の大きさを表す整数。
        - hidden_dim: 隠れ層のサイズを表す整数。
        - num_classes: 分類するクラスの数を指定する整数。
        - weight_scale: ランダムな重みの標準偏差を与えるスカラー。
          の標準偏差を与えるスカラー。
        - reg: L2 正則化の強さを与えるスカラー。
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        #                                                                          #
        # 層ネットの重みとバイアスを初期化する。
        # 重みはweight_scaleに等しい標準偏差を持つ0.0を中心とするガウスから初期化され、
        # バイアスはゼロに初期化されるべきである。
        # すべての重みとバイアスは辞書 self.params に格納する。
        # 第1層の重みとバイアスは 'W1' と 'b1' をキーに、
        # 第2層の重みとバイアスは 'W2' と 'b2' をキーにする。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params = {
            "W1": np.random.randn(input_dim, hidden_dim) * weight_scale,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, num_classes) * weight_scale,
            "b2": np.zeros(num_classes),
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.

        ミニバッチデータの損失と勾配を計算する。

        入力
        - X: 形状 (N, d_1, ..., d_k) の入力データの配列.
        - y: y[i] は X[i] のラベルを与える。

        戻り値：
        y が None の場合、モデルのテストタイムフォワードパスを実行し、リターンする：
        - スコアを返す： 分類スコアを与える shape (N, C) の配列。
          scores[i, c] は X[i] とクラス c の分類スコアである。

        y が None でない場合、学習時のフォワードパスとバックワードパスを実行し
        のタプルを返す：
        - loss: 損失を与えるスカラー値
        - grads: self.paramsと同じキーを持つ辞書。
          パラメータ名を、それらのパラメータに対する損失の勾配にマッピングする。
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        #                                                                          #
        # Xのクラス・スコアを計算し、変数scoresに格納する。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1, W2, b2 = self.params.values()

        out1, cache1 = affine_forward(X, W1, b1)
        out2, cache2 = relu_forward(out1)
        scores, cache3 = affine_forward(out2, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #                                                                          #
        # TODO: 2層ネットのバックワードパスを実装する。
        # 損失をloss変数に、勾配をgrads辞書に格納する。
        # softmaxを使ってデータ損失を計算し、
        # grads[k]がself.params[k]の勾配を保持していることを確認する。
        # L2正則化を追加することを忘れないでください！
        #
        # NOTE: あなたの実装が我々の実装と一致し、自動テストに合格するように、
        # L2正則化に係数0.5が含まれ、勾配の式が単純化されていることを確認してください。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dout3, dW2, db2 = affine_backward(dloss, cache3)
        dout2 = relu_backward(dout3, cache2)
        dout1, dW1, db1 = affine_backward(dout2, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
