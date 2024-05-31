from builtins import object, range

import numpy as np

from ..layer_utils import *
from ..layers import *


class FullyConnectedNet(object):
    """
    Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.

    多層完全連結ニューラルネットワークのクラス。

    ネットワークは任意の数の隠れ層、ReLU非線形性を含む、
    およびソフトマックス損失関数があります。また、オプションとしてドロップアウトとバッチ/レイヤー
    正規化も実装される。L層のネットワークの場合、アーキテクチャは次のようになる。

    {アフィン - [バッチ/レイヤノルム] - リリュー - [ドロップアウト]} x (L - 1) - アフィン - ソフトマックス

    ここでバッチ/レイヤーの正規化とドロップアウトは任意であり、{...}ブロックはL - 1回繰り返される。
    をL - 1回繰り返す。

    学習可能なパラメータはself.params辞書に格納され、ソルバークラスを使って学習されます。
    ソルバークラスを使って学習されます。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.

        新しい FullyConnectedNet を初期化します。

        入力：
        - hidden_dims: 各隠れ層のサイズを示す整数のリスト。
        - input_dim: 入力のサイズを示す整数。
        - num_classes: クラスの数： 分類するクラスの数を指定する整数。
        - dropout_keep_ratio: ドロップアウトの強さを表す 0 から 1 までのスカラー。
            dropout_keep_ratio=1 の場合、ネットワークはドロップアウトをまったく使用しない。
        - normalization: ネットワークが使用する正規化の種類。有効な値は 「batchnorm」、「layernorm」、または正規化なし（デフォルト）の 「None」。
        - reg: L2正則化の強さを与えるスカラー。
        - weight_scale: 重みのランダム初期化の標準偏差を指定するスカラー。
        - dtype: すべての計算はこのデータ型を使って行われます。
            すべての計算はこのデータ型を使って行われます。
            を使うべきである。
        - seed: None でない場合、このランダムシードをドロップアウトレイヤーに渡す。
            これにより、ドロップアウトレイヤーをデテリミスティックにし、モデルのグラデーションチェックを行うことができます。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        #                                                                          #
        # ネットワークのパラメータを初期化し、すべての値をself.params辞書に格納する。
        # 第1層の重みとバイアスをW1とb1に格納し、第2層はW2とb2を使用する。
        # 重みは0を中心とする正規分布から初期化し、標準偏差はweight_scaleに等しくする。
        # バイアスはゼロに初期化する。
        # バッチ正規化を使用する場合、
        # 最初のレイヤーのスケールとシフトのパラメータをガンマ1とベータ1に格納し、
        # 2番目のレイヤーにはガンマ2とベータ2を使用する。スケール・パラメータは1に、
        # シフト・パラメータは0に初期化する。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for l, (i, j) in enumerate(
            zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])
        ):
            # 重みとバイアスを初期化
            # 重みは正規分布からランダムに選んで初期化し、
            # バイアスはゼロで初期化する
            self.params[f"W{l+1}"] = np.random.randn(i, j) * weight_scale
            self.params[f"b{l+1}"] = np.zeros(j)

            # 正規化が有効で、最後のレイヤーでない場合
            if self.normalization and l < self.num_layers - 1:
                self.params[f"gamma{l+1}"] = np.ones(j)
                self.params[f"beta{l+1}"] = np.zeros(j)

        # del self.params[f'gamma{l+1}'], self.params[f'beta{l+1}'] # no batchnorm after last FC

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 各ドロップアウトレイヤーにdropout_param辞書を渡す必要があります。
        # これにより、レイヤーはドロップアウト確率と
        # モード（train / test）を知ることができます。
        # 各ドロップアウト層に同じdropout_paramを渡すことができます。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # バッチ正規化では、実行平均と分散を追跡する必要があるので、
        # 各バッチ正規化レイヤに特別なbn_paramオブジェクトを渡す必要があります。
        # self.bn_params[0]を最初のバッチ正規化層のフォワード・パスに渡し、
        # self.bn_params[1]を2番目のバッチ正規化層のフォワード・パスに渡す。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # すべてのパラメータを正しいデータ型にキャストする。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully connected net.

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

        完全連結ネットの損失と勾配を計算する。

        入力
        - X: 入力データの配列 (N, d_1, ..., d_k)
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # batchnormパラメータとdropoutパラメータは、トレーニング時とテスト時で挙動が異なるため、
        # train/testモードを設定する。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        #                                                                          #
        # 完全連結ネットのフォワードパスを実装し、Xのクラススコアを計算し、
        # 変数scoresに格納する。
        # ドロップアウトを使用する場合、各ドロップアウト・フォワードパスに
        # self.dropout_paramを渡す必要がある。
        #
        # バッチ正規化を使用する場合、最初のバッチ正規化レイヤーのフォワード・パスに
        # self.bn_params[0]を渡し、2番目のバッチ正規化レイヤーのフォワード・パスに
        # self.bn_params[1]を渡す必要がある。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache = {}

        for l in range(self.num_layers):
            # パラメーターのリスト
            keys = [f"W{l+1}", f"b{l+1}", f"gamma{l+1}", f"beta{l+1}"]
            # パラメーターを取得する
            w, b, gamma, beta = (self.params.get(k, None) for k in keys)

            bn = self.bn_params[l] if gamma is not None else None
            do = self.dropout_param if self.use_dropout else None

            # ジェネリックフォワードパス
            X, cache[l] = generic_forward(
                X, w, b, gamma, beta, bn, do, l == self.num_layers - 1
            )

        scores = X

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #                                                                          #
        # 完全に接続されたネットに対してバックワードパスを実施する。
        # 損失をloss変数に、勾配をgrads辞書に格納する。
        # softmaxを使ってデータ損失を計算し、
        # grads[k]がself.params[k]の勾配を保持していることを確認する。
        # L2正則化を追加することを忘れないでください！
        # バッチ／レイヤー正規化を使用する場合、
        # スケールとシフトのパラメータを正規化する必要はない。
        # 注：あなたの実装が我々の実装と一致し、
        # 自動テストに合格することを保証するために、L2正則化に係数0.5が含まれ、
        # 勾配の式が単純化されていることを確認してください。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        # L2正則化
        loss += (
            0.5
            * self.reg
            * np.sum([np.sum(W**2) for k, W in self.params.items() if "W" in k])
        )

        for l in reversed(range(self.num_layers)):
            dout, dW, db, dgamma, dbeta = generic_backward(dout, cache[l])

            grads[f"W{l+1}"] = dW + self.reg * self.params[f"W{l+1}"]
            grads[f"b{l+1}"] = db

            if dgamma is not None and l < self.num_layers - 1:
                grads[f"gamma{l+1}"] = dgamma
                grads[f"beta{l+1}"] = dbeta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
