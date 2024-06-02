from builtins import object

import numpy as np

from ..fast_layers import *
from ..layer_utils import *
from ..layers import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    次のアーキテクチャを持つ 3 層の畳み込みネットワーク:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    ネットワークは、高さ H、幅 W の N 個の画像と C 個の入力チャネルで構成される、形状 (N、C、H、W) のデータのミニバッチで動作します。
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.

        新しいネットワークを初期化します。

        入力:
        - input_dim: 入力データのサイズを示すタプル (C、H、W)
        - num_filters: 畳み込み層で使用するフィルターの数
        - filter_size: 畳み込み層で使用するフィルターの幅/高さ
        - hidden_dim: 完全接続の隠し層で使用するユニットの数
        - num_classes: 最終アフィン層から生成するスコアの数。
        - weight_scale: 重みのランダム初期化の標準偏差を示すスカラー。
        - reg: L2 正則化の強度を示すスカラー
        - dtype: 計算に使用する numpy データ型。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        #                                                                          #
        # 3 層畳み込みネットワークの重みとバイアスを初期化します。
        # 重みは、標準偏差が weight_scale に等しい 0.0 を中心とする
        # ガウス分布から初期化する必要があります。
        # バイアスは 0 に初期化する必要があります。
        # すべての重みとバイアスは、辞書 self.params に格納する必要があります。
        # 畳み込み層の重みとバイアスは、キー 'W1' と 'b1' を使用して格納します。
        # 隠しアフィン層の重みとバイアスにはキー 'W2' と 'b2' を使用し、
        # 出力アフィン層の重みとバイアスにはキー 'W3' と 'b3' を使用します。
        #
        # 重要: この課題では、**入力の幅と高さが保持される** ように
        # 最初の畳み込み層のパディングとストライドが選択されていると想定できます。
        # loss() 関数の先頭を見て、それがどのように行われるかを確認してください。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 次元サイズの取得
        F, (C, H, W) = num_filters, input_dim

        self.params.update(
            {
                "W1": np.random.randn(F, C, filter_size, filter_size)
                * weight_scale,  # すべてのフィルタ形状を考慮する
                "b1": np.zeros(num_filters),  # フィルターの数を考慮する
                "W2": np.random.randn(F * H * W // 4, hidden_dim)
                * weight_scale,  # プール出力の削減を検討する
                "b2": np.zeros(hidden_dim),  # 隠れノードの数を考慮する
                "W3": np.random.randn(hidden_dim, num_classes)
                * weight_scale,  # 隠しノードと出力ノードを考慮する
                "b3": np.zeros(num_classes),  # 出力ノードを考慮する
            }
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.

        3 層畳み込みネットワークの損失と勾配を評価します。

        入力 / 出力: fc_net.py の TwoLayerNet と同じ API。
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # conv_paramを畳み込み層のフォワードパスに渡す
        # 入力空間サイズを維持するために選択されたパディングとストライド
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        #                                                                          #
        # 3 層畳み込みネットワークのフォワードパスを実装し、X のクラススコアを計算し、
        # scores 変数に格納します。
        #
        # 実装には、cs231n/fast_layers.py と cs231n/layer_utils.py で定義された
        # 関数を使用できることを覚えておいてください（すでにインポートされています）。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 畳み込みフォワードパス
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # 全結合フォワードパス
        out, cache2 = affine_relu_forward(out, W2, b2)
        # 出力層フォワードパス
        scores, cache3 = generic_forward(out, W3, b3, last=True)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #                                                                          #
        # 3 層畳み込みネットワークのバックワードパスを実装し、
        # 損失と勾配を loss と grads 変数に格納します。
        # softmax を使用してデータ損失を計算し、
        # grads[k] が self.params[k] の勾配を保持するようにします。
        # L2 正則化を追加することを忘れないでください！
        #
        # 注: 実装が私たちのものと一致し、自動テストに合格するようにするには、
        # L2 正則化に 0.5 の係数を含めるようにして、
        # 勾配の式を簡略化してください。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        # 正規化された損失
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        # 最初のバックワードパス
        dout, grads["W3"], grads["b3"], _, _ = generic_backward(dout, cache3)
        # 全結合バックワードパス
        (
            dout,
            grads["W2"],
            grads["b2"],
        ) = affine_relu_backward(dout, cache2)
        # 畳み込みバックワードパス
        (
            dout,
            grads["W1"],
            grads["b1"],
        ) = conv_relu_pool_backward(dout, cache1)

        # L2 正則化
        grads["W3"] += self.reg * W3
        grads["W2"] += self.reg * W2
        grads["W1"] += self.reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
