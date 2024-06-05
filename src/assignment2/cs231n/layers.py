from builtins import range

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)

    アフィン（完全接続）層のフォワードパスを計算する。

    入力 x は形状 (N, d_1, ..., d_k) を持ち、N 個のミニバッチを含む。
    のミニバッチを含み、各例 x[i]は形状(d_1, ..., d_k)を持つ。我々は
    各入力をD = d_1 * ... * d_kの次元のベクトルに再形成する。* d_k、そして
    次元の出力ベクトルに変換する。

    入力
    - x: (N, d_1, ..., d_k)の入力データを含むnumpy配列。
    - w: 重みを格納するnumpy配列。
    - b: バイアスを表すnumpy配列,形状は (M,)

    のタプルを返す：
    - out: 形状 (N, M) の出力。
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape(len(x), -1) @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)

    アフィン（完全接続）レイヤーのバックワードパスを計算する。

    入力
    - dout: 上流導関数, 形状は (N, M)
    - キャッシュ： のタプル：
      - x: 入力データ、形状は (N, d_1, ... d_k)
      - w: 重み、形状は (D, M)
      - b: 形状 (M,) のバイアス。

    のタプルを返す：
    - dx: dx: xに対する勾配, 形状 (N, d1, ..., d_k)
    - dw: 形状 (D, M) の w に関する勾配
    - db: 形状(M,)のbに関する勾配
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape(len(x), -1).T @ dout
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x

    整流された線形ユニット(ReLUs)のレイヤーのフォワードパスを計算する。

    入力
    - x: 任意の形状の入力

    のタプルを返す：
    - out: x と同じ形の出力。
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x

    整流された線形ユニット(ReLUs)のレイヤーのバックワードパスを計算する。

    入力
    - dout: 任意の形状の上流導関数
    - cache: 入力 x, dout と同じ形状

    戻り値
    - dx: x に対する勾配
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x

    ソフトマックス分類の損失と勾配を計算する．

    入力
    - x: 入力: x: 入力データ(N, C)。
      クラスのスコアである。
    - y: y[i]はx[i]に対するラベルであり
      0 <= y[i] < C

    のタプルを返す：
    - loss: 損失を与えるスカラー
    - dx: x に対する損失の勾配
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = len(y)  # number of samples

    # numerically stable exponents
    P = np.exp(x - x.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)  # row-wise probabilities (softmax)

    loss = -np.log(P[range(N), y]).sum() / N  # sum cross entropies as loss

    P[range(N), y] -= 1
    dx = P / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass

    バッチ正規化のためのフォワードパス。

    トレーニング中、サンプル平均と（未補正の）サンプル分散がミニバッチ統計から計算され、入力データの正規化に使用される。
    ミニバッチ統計から計算され、入力データの正規化に使用される。
    学習中、各特徴の平均と分散の指数減衰する実行平均も保持します。
    これらの平均はテスト時のデータの正規化に使用される。

    各タイムステップで、運動量パラメータに基づく指数減衰を用いて、平均と分散の実行平均を更新する：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    バッチ正規化論文では、テスト時の動作が異なっていることに注意してください。
    この実装では、ランニングアベレージを使用することにしました。
    バッチ正規化の実装でも、実行平均を使用しています。

    入力
    - x: 形状 (N, D) のデータ
    - gamma: 形状のスケールパラメータ (D,)
    - beta: 形状のシフトパラメータ (D,)
    - bn_param: 以下のキーを持つ辞書：
      - mode: 'train' または 'test'; 必須
      - eps: 数値安定性のための定数
      - momentum: 実行平均/分散を表す定数.
      - running_mean: 特徴量の走行平均を与える shape (D,) の配列．
      - running_var 特徴量の実行分散を与える shape (D,) の配列．

    のタプルを返す：
    - out: shape (N, D) のタプル
    - cache: バックワードパスで必要な値のタプル
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #                                                                     #
        # バッチ正規化のトレーニング時のフォワードパスを実装する。
        # ミニバッチ統計を使用して平均と分散を計算し、
        # これらの統計を使用して入力データを正規化し、
        # ガンマとベータを使用して正規化されたデータをスケーリングおよびシフトする。
        #
        # 出力は out 変数に格納する必要があります。
        # バックワードパスに必要な中間値は cache 変数に格納する必要があります。
        #
        # また、サンプル平均と分散を計算し、モーメンタム変数と共に
        # 実行平均と実行分散を更新する必要があります。
        # 結果を running_mean と running_var 変数に格納してください。
        #
        # 実行分散を追跡する必要がありますが、
        # データを標準偏差（分散の平方根）に基づいて正規化する必要があります！
        # 元の論文 (https://arxiv.org/abs/1502.03167)
        # を参照すると役立つかもしれません。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 各特徴のバッチ平均
        mu = x.mean(axis=0)
        # 各特徴のバッチ分散
        var = x.var(axis=0)
        # 各特徴のバッチ標準偏差
        std = np.sqrt(var + eps)
        # 標準化された x
        x_hat = (x - mu) / std
        # スケーリングとシフトされた x_hat
        out = gamma * x_hat + beta

        # バックプロップで使用されるリシェイプ
        shape = bn_param.get("shape", (N, D))
        # バックプロップで使用される合計軸
        axis = bn_param.get("axis", 0)
        # キャッシュに保存
        cache = x, mu, var, std, gamma, x_hat, shape, axis

        # モーメンタムを使用して実行平均と実行分散を更新
        # バッチ正規化でない場合
        if axis == 0:
            running_mean = (
                # 全体平均の更新
                momentum * running_mean + (1 - momentum) * mu
            )
            running_var = (
                # 全体分散の更新
                momentum * running_var + (1 - momentum) * var
            )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #                                                                     #
        # バッチ正規化のテスト時のフォワードパスを実装する。
        # 実行平均と実行分散を使用して入力データを正規化し、
        # ガンマとベータを使用して正規化されたデータをスケーリングおよびシフトする。
        # 結果を out 変数に格納してください。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 更新された実行中のパラメータをbn_paramに戻す
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)

    バッチ正規化のためのバックワードパス。

    この実装では、バッチ正規化のための計算グラフを紙に書き出して
    この実装では、バッチ正規化のための計算グラフを紙に書き出し、勾配を中間ノードを介して後方に伝搬させる必要がある
    中間ノードを介して後方に勾配を伝播させる。

    入力
    - dout: 形状 (N, D) の上流導関数
    - キャッシュ： batchnorm_forward からの中間ノードの変数。

    のタプルを返す：
    - dx: 入力 x に対する勾配, 形状は (N, D).
    - dgamma: スケールパラメータ gamma に関する勾配。
    - dbeta: 形状(D,)のシフトパラメータβに関する勾配
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    #                                                                         #
    # バッチ正規化のためのバックワードパスを実装する。
    # dx, dgamma, dbeta 変数に結果を格納してください。
    # 元の論文 (https://arxiv.org/abs/1502.03167)
    # を参照すると役立つかもしれません。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    x, mu, var, std, gamma, x_hat, shape, axis = cache  # キャッシュを展開する

    # betaに関する導関数
    dbeta = dout.reshape(shape, order="F").sum(axis)
    # gammaに関する導関数
    dgamma = (dout * x_hat).reshape(shape, order="F").sum(axis)

    # x_hatに関する導関数
    dx_hat = dout * gamma
    # stdに関する導関数
    dstd = -np.sum(dx_hat * (x - mu), axis=0) / (std**2)
    # varに関する導関数
    dvar = 0.5 * dstd / std
    # dxに関する偏導関数
    dx1 = dx_hat / std + 2 * (x - mu) * dvar / len(dout)
    # muに関する導関数
    dmu = -np.sum(dx1, axis=0)
    # dxに関する偏導関数
    dx2 = dmu / len(dout)
    # xに関する完全な導関数
    dx = dx1 + dx2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward

    バッチ正規化のための代替バックワードパス。

    この実装では、バッチ正規化バックワードパスの導関数を紙の上で計算し、可能な限り単純化する必要がある。
    そして後方パスの簡単な式を導くことができるはずです。
    より多くのヒントはjupyter notebookを参照してください。

    注意: この実装では、batchnorm_backwardと同じキャッシュ変数
    を受け取るはずですが、キャッシュ内のすべての値を使うとは限りません。

    入出力： batchnorm_backwardと同じ
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    #                                                                         #
    # バッチ正規化のためのバックワードパスを実装する。
    # dx, dgamma, dbeta 変数に結果を格納してください。
    #
    # 中心化された入力に関する勾配を計算した後、
    # 1つのステートメントで入力に関する勾配を計算できるはずです。
    # 私たちの実装は80文字の1行に収まります。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    _, _, _, std, gamma, x_hat, shape, axis = cache

    # ヘルパー関数
    # numpyのsum関数をSとして定義
    def S(x):
        return x.sum(axis=0)

    # betaに関する導関数
    dbeta = dout.reshape(shape, order="F").sum(axis)
    # gammaに関する導関数
    dgamma = (dout * x_hat).reshape(shape, order="F").sum(axis)

    # スケール値を一時的に初期化する
    dx = dout * gamma / (len(dout) * std)
    # 非正規化されたxに関する導関数
    dx = len(dout) * dx - S(dx * x_hat) * x_hat - S(dx)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass

    レイヤー正規化のためのフォワードパス。

    トレーニング時とテスト時の両方で、入力データはデータポイントごとに正規化される、
    バッチ正規化と同じガンマとベータパラメータでスケーリングされる前に。

    バッチ正規化とは対照的に、学習時とテスト時のレイヤ正規化の動作は同じであることに注意。
    層正規化の訓練時とテスト時の動作は同一であり、実行平均を追跡する必要がないことに注意する。
    を追跡する必要はない。

    入力
    - x: データ形状 (N, D)
    - gamma: 形状のスケールパラメータ (D,)
    - beta: 形状のシフトパラメータ (D,)
    - ln_param: 以下のキーを持つ辞書：
        - eps: 数値安定のための定数

    のタプルを返す：
    - out: 形状 (N, D) の
    - キャッシュ： 後方パスで必要な値のタプル
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    #                                                                         #
    # レイヤ正規化のためのトレーニング時のフォワードパスを実装する。
    # ガンマとベータを使用して入力データを正規化し、
    # 正規化されたデータをスケーリングおよびシフトする
    # ヒント：これは、バッチ正規化のトレーニング時の実装を
    # わずかに変更することで行うことができます。
    # そして、適切な場所に1行または2行のコードを挿入します。
    # 特に、どのような行列変換を行うことができるか考えることができますか？
    # これにより、バッチ正規化コードをコピーしてほぼ変更せずに残すことができますか？
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # trainモードでのbatchnormと同じ + gradをどの軸で合計するか
    bn_param = {
        "mode": "train",
        "axis": 1,
        **ln_param,
    }
    # 転置を行う2Dを保証する
    [gamma, beta] = np.atleast_2d(gamma, beta)

    out, cache = batchnorm_forward(x.T, gamma.T, beta.T, bn_param)
    # 転置をもとに戻す
    out = out.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)

    レイヤー正規化のためのバックワードパス。

    この実装では、バッチ正規化ですでに行った作業を大いに利用することができる。
    を大いに利用することができる。

    入力：
    - dout: 形状（N, D）のアップストリーム微分。
    - キャッシュ： layernorm_forward からの中間値の変数。

    のタプルを返す：
    - dx: 入力 x に対する勾配, 形状は (N, D).
    - dgamma: スケールパラメータ gamma に対する勾配。
    - dbeta: 形状(D,)のシフトパラメータβに関する勾配
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    #                                                                         #
    # レイヤ正規化のためのバックワードパスを実装する。
    #
    # ヒント：これは、バッチ正規化のトレーニング時の実装を
    # わずかに変更することで行うことができます。
    # フォワードパスへのヒントは引き続き適用されます！
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # バッチノームバックプロップと同じ
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    # # 転置をもとに戻す
    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    逆ドロップアウトのフォワードパス。

    これはバニラ版のドロップアウトとは異なることに注意。
    ここで、pはニューロン出力を維持する確率であり、ニューロン出力をドロップする確率とは対照的である。
    とは対照的である。
    詳細は http://cs231n.github.io/neural-networks-2/#reg を参照。

    入力
    - x: 入力データ。
    - dropout_param: 以下のキーを持つ辞書：
      - p: ドロップアウトパラメータ。各ニューロンの出力を確率pで保持する。
      - mode: 'test' または 'train'. モードがtrainの場合、ドロップアウトを実行する
        (モードがtestの場合、入力を返す。)
      - seed: 乱数生成器のシード。seedを渡すと
        これは勾配チェックには必要だが、実際のネットワークでは必要ない。

    出力：
    - out: x と同じ形の配列。
    - cache: タプル (dropout_param, mask)。訓練モードでは、mask は入力に乗算するために使用された
      テストモードでは、mask は None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #                                                                     #
        # 逆ドロップアウトのためにトレーニングフェーズのフォワードパスを導入する。
        # マスク変数にドロップアウトマスクを格納する。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # p未満で1、p以上で0のマスクを適用
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #                                                                     #
        # 逆ドロップアウトのテストフェーズのフォワードパスを実装する。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # テスト時はドロップアウトしない
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.

    逆ドロップアウトのためのバックワードパス。

    入力
    - dout: 任意の形状のアップストリーム微分
    - cache:  dropout_forwardからの(dropout_param, mask).
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #                                                                     #
        # 逆ドロップアウトのためのトレーニングフェーズのバックワードパスを実装する
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # ドロップアウトを適用した場合の微分
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)

    畳み込み層のフォワードパスの素朴な実装。

    入力はN個のデータ点からなり、それぞれ高さH、幅WのC個のチャンネルを持つ。
    各入力をF個の異なるフィルターで畳み込む。
    はすべてのC個のチャンネルにまたがり、高さHH、幅WWを持つ。

    入力
    - x: 形状の入力データ (N, C, H, W)
    - w: 形状(F, C, HH, WW)のフィルター重み
    - b: 形状 (F,) のバイアス。
    - conv_param: 以下のキーを持つ辞書：
      - stride'： 水平方向と垂直方向で隣接する受容野間のピクセル数。
        水平方向と垂直方向の隣接する受容野間のピクセル数。
      - pad'： pad': 入力をゼロパディングするために使用されるピクセル数。

    パディングの際、'pad'ゼロは、入力の縦軸と横軸に沿って対称に(つまり左右均等に)配置されなければならない。
    入力の縦軸と横軸に沿って対称に（つまり左右均等に）配置されなければならない。元の
    を直接修正しないように注意すること。

    のタプルを返す：
    - out： 出力データ。形状は (N, F, H', W') で、H' と W' は次式で与えられる。
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - キャッシュ: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    #                                                                         #
    # 畳み込みのフォワードパスを実装する。
    # ヒント：パディングには np.pad 関数を使用できます。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # パディング：上＝右＝下＝左
    PADDING_TOP = PADDING_RIGHT = PADDING_BOTTOM = PADDING_LEFT = conv_param["pad"]
    # ストライド：上＝下
    STRIDE_TOP = STRIDE_BOTTOM = conv_param["stride"]
    # 入力の次元
    N, C, HI, WI = x.shape
    # フィルタの次元
    F, _, HF, WF = w.shape
    # 出力の高さ
    HO = 1 + (HI + PADDING_TOP + PADDING_BOTTOM - HF) // STRIDE_TOP
    # 出力の幅
    WO = 1 + (WI + PADDING_RIGHT + PADDING_LEFT - WF) // STRIDE_BOTTOM

    # ヘルパー関数 (警告: 使用には numpy バージョン 1.20 以上が必要)
    def to_fields(x):
        return np.lib.stride_tricks.sliding_window_view(x, (WF, HF, C, N))

    # ウェイトを行に
    w_row = w.reshape(F, -1)
    # パディングされた入力
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (PADDING_TOP, PADDING_BOTTOM), (PADDING_RIGHT, PADDING_LEFT)),
        "constant",
    )
    # 入力を列に
    x_col = (
        to_fields(x_pad.T)
        .T[..., ::STRIDE_TOP, ::STRIDE_BOTTOM]
        .reshape(N, C * HF * WF, -1)
    )

    out = (w_row @ x_col).reshape(N, F, HO, WO) + np.expand_dims(b, axis=(2, 1))

    # バックプロパゲーションでは、パディングされたバージョンも使用する。
    x = x_pad

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b

    畳み込み層の後方パスの素朴な実装。

    入力
    - dout: 上流の微分。
    - cache: conv_forward_naive と同様、 (x, w, b, conv_param) のタプル。

    のタプルを返す：
    - dx: x に対する勾配
    - dw: w に対する勾配
    - db: bに対する勾配
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    #                                                                         #
    # 畳み込みバックワードパスを実装する。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ヘルパー関数 (警告: numpy 1.20 以上が必要)
    to_fields = np.lib.stride_tricks.sliding_window_view

    # キャッシュからパラメータを抽出
    x_pad, w, b, conv_param = cache
    # ストライド：上＝下
    STRIDE_TOP = STRIDE_BOTTOM = conv_param["stride"]
    # パディング：上＝右＝下＝左
    PADDING_TOP = PADDING_RIGHT = PADDING_BOTTOM = PADDING_LEFT = conv_param["pad"]
    # フィルタの次元
    F, C, HF, WF = w.shape
    # 出力の次元
    N, _, HO, WO = dout.shape

    # 「行方不明」の行
    dout = np.insert(dout, [*range(1, HO)] * (STRIDE_TOP - 1), 0, axis=2)
    # 「行方不明」の列
    dout = np.insert(dout, [*range(1, WO)] * (STRIDE_BOTTOM - 1), 0, axis=3)
    # 完全畳み込み用
    dout_pad = np.pad(dout, ((0,), (0,), (HF - 1,), (WF - 1,)), "constant")

    # doutに関する入力のフィールド
    x_fields = to_fields(x_pad, (N, C, dout.shape[2], dout.shape[3]))
    # filterに関するdoutのフィールド
    dout_fields = to_fields(dout_pad, (N, F, HF, WF))
    # 回転カーネル（畳み込み用）
    w_rot = np.rot90(w, 2, axes=(2, 3))

    # 総和
    db = np.einsum("ijkl->j", dout)
    # 関連付ける
    dw = np.einsum("ijkl,mnopiqkl->jqop", dout, x_fields)
    # 畳み込み
    dx = np.einsum("ijkl,mnopqikl->qjop", w_rot, dout_fields)[
        ..., PADDING_TOP:-PADDING_BOTTOM, PADDING_RIGHT:-PADDING_LEFT
    ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)

    マックスプーリング層のフォワードパスの素朴な実装。

    入力
    - x: 入力データ、形状は (N, C, H, W)
    - pool_param: 以下のキーを持つ辞書：
      - pool_height': 各プーリング領域の高さ
      - pool_width': 各プーリング領域の幅
      - 'stride': 隣接するプーリング領域間の距離

    ここではパディングは必要ない：
      - (H - pool_height) % stride == 0 と仮定できます。
      - (W - pool_width) % stride == 0 と仮定します。

    のタプルを返す：
    - 出力： 出力データ。形状は (N, C, H', W') で、H' と W' は次式で与えられる。
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride。
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    #                                                                         #
    # マックスプーリング・フォワードパスの実装
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # パラメータを展開
    # ストライド：上＝下
    S1 = S2 = pool_param["stride"]
    # プールの高さ
    HP = pool_param["pool_height"]
    # プールの幅
    WP = pool_param["pool_width"]
    # 入力の次元
    N, C, HI, WI = x.shape
    # 出力の高さ
    HO = 1 + (HI - HP) // S1
    # 出力の幅
    WO = 1 + (WI - WP) // S2

    # ヘルパー関数 (警告: 使用には numpy バージョン 1.20 以上が必要です)
    def to_fields(x):
        return np.lib.stride_tricks.sliding_window_view(x, (WP, HP, C, N))

    # 入力のローカル領域
    x_fields = to_fields(x.T).T[..., ::S1, ::S2].reshape(N, C, HP * WP, -1)
    # プールされた出力
    out = x_fields.max(axis=2).reshape(N, C, HO, WO)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x

    マックスプーリング層のバックワードパスの素朴な実装。

    入力
    - dout: アップストリーム微分
    - cache: (x,pool_param)のタプル。

    戻り値
    - dx: x に対する勾配
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    #                                                                         #
    # マックスプーリングのバックワードパスを実装する
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # パラメータを展開
    x, pool_param = cache
    N, C, HO, WO = dout.shape
    # 初期の導関数
    dx = np.zeros_like(x)

    # ストライド：上＝下
    S1 = S2 = pool_param["stride"]
    # プールの高さ
    HP = pool_param["pool_height"]
    # プールの幅
    WP = pool_param["pool_width"]

    for i in range(HO):
        for j in range(WO):
            [ns, cs], h, w = (
                np.indices(
                    # コンパクトなインデックス
                    (N, C)
                ),
                i * S1,
                j * S2,
            )
            # 入力ローカルフィールド
            f = x[:, :, h : (h + HP), w : (w + WP)].reshape(N, C, -1)
            # 最大値のオフセット
            k, l = np.unravel_index(np.argmax(f, 2), (HP, WP))
            # 更新するエリアを選択する
            dx[ns, cs, h + k, w + l] += dout[ns, cs, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass

    空間バッチ正規化のフォワード パスを計算します。

    入力:
    - x: 形状 (N, C, H, W) の入力データ
    - gamma: 形状 (C,) のスケール パラメータ
    - beta: 形状 (C,) のシフト パラメータ
    - bn_param: 次のキーを持つ辞書:
    - mode: 'train' または 'test'; 必須
    - eps: 数値安定性の定数
    - momentum: 実行平均 / 分散の定数。
      momentum=0 は、古い情報が各時間ステップで完全に破棄されることを意味し、
      momentum=1 は新しい情報が組み込まれないことを意味します。
      momentum=0.9 のデフォルトは、ほとんどの状況で適切に機能します。
    - running_mean: フィーチャの実行平均を示す形状 (D,) の配列
    - running_var フィーチャの実行分散を示す形状 (D,) の配列

    次のタプルを返します:
    - out: 形状 (N、C、H、W) の出力データ
    - cache: バックワード パスに必要な値
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    #                                                                         #
    # 空間バッチ正規化のフォワード パスを実装します。
    #
    # ヒント: 上記で実装したバッチ正規化のバニラバージョンを呼び出すことで、
    # 空間バッチ正規化を実装できます。
    # あなたの実装は非常に短くなるはずです。私たちの実装は5行未満です。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 入力の次元
    N, C, H, W = x.shape
    # 軸を交換してバニラバッチノルムを使用する
    x = np.moveaxis(x, 1, -1).reshape(-1, C)
    # バニラバッチノルムを実行する
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    # 出力の軸を入れ替える
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)

    空間バッチ正規化の逆方向パスを計算します。

    入力:
    - dout: 上流導関数、形状 (N, C, H, W)
    - cache: 順方向パスからの値

    次のタプルを返します:
    - dx: 入力に関する勾配、形状 (N, C, H, W)
    - dgamma: スケール パラメータに関する勾配、形状 (C,)
    - dbeta: シフト パラメータに関する勾配、形状 (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    #                                                                         #
    # 空間バッチ正規化の逆方向パスを計算します。
    #
    # ヒント: 上記で実装したバッチ正規化のバニラバージョンを呼び出すことで、
    # 空間バッチ正規化を実装できます。
    # あなたの実装は非常に短くなるはずです。私たちの実装は5行未満です。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # アップストリームの次元
    N, C, H, W = dout.shape
    # 軸を交換してバニラのバッチノルムバックプロパゲーションを使用する
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)
    # バニラバッチノルムバックプロパゲーションを実行する
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    # dxの勾配の軸を入れ替える
    dx = np.moveaxis(dx.reshape(N, H, W, C), -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass

    空間グループ正規化のフォワード パスを計算します。

    レイヤー正規化とは対照的に、グループ正規化では、データの各エントリを G 個の連続した部分に分割し、
    個別に正規化します。次に、バッチ正規化およびレイヤー正規化と同じ方法で、
    フィーチャごとのシフトとスケーリングがデータに適用されます。

    入力:
    - x: 形状 (N、C、H、W) の入力データ
    - gamma: 形状 (1、C、1、1) のスケール パラメーター
    - beta: 形状 (1、C、1、1) のシフト パラメーター
    - G: 分割するグループの整数数。C の約数である必要があります
    - gn_param: 次のキーを持つ辞書:
    - eps: 数値安定性のための定数

    次のタプルを返します:
    - out: 形状 (N、C、H、W) の出力データ
    - cache: バックワード パスに必要な値
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    #                                                                         #
    # 空間グループ正規化のフォワード パスを実装します。
    # これはレイヤー正規化の実装と非常に似ています。
    # 特に、どのようにして行列を変換して、
    # コードの大部分がトレーニング時のバッチ正規化とレイヤー正規化の
    # 両方と似ているかについて考えてみてください！
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 入力の次元
    N, C, H, W = x.shape
    # バッチ正規化メソッドを再利用するためのパラメータ
    ln_param = {
        "shape": (W, H, C, N),
        "axis": (0, 1, 3),
        **gn_param,
    }

    # x をバニラレイヤーノルムを使用するために再形成する
    x = x.reshape(N * G, -1)

    # ガンマとベータを再形成してバニラレイヤーノルムを使用する
    gamma = np.tile(gamma, (N, 1, H, W)).reshape(N * G, -1)
    beta = np.tile(beta, (N, 1, H, W)).reshape(N * G, -1)
    out, cache = layernorm_forward(x, gamma, beta, ln_param)

    # 出力の形状を元に戻す
    out = out.reshape(N, C, H, W)
    # キャッシュにはGが含まれる
    cache = (G, cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)

    空間グループ正規化の逆方向パスを計算します。

    入力:
    - dout: 上流導関数、形状 (N、C、H、W)
    - cache: 順方向パスからの値

    次のタプルを返します:
    - dx: 入力に関する勾配、形状 (N、C、H、W)
    - dgamma: スケール パラメータに関する勾配、形状 (1、C、1、1)
    - dbeta: シフト パラメータに関する勾配、形状 (1、C、1、1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    #                                                                         #
    # 空間グループ正規化の逆方向パスを計算します。
    # これはレイヤー正規化の実装と非常に似ています。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # キャッシュを展開
    G, cache = cache
    # アップストリームの次元
    N, C, H, W = dout.shape
    # バニラレイヤーノルムバックプロパゲーションを使用するために再形成する
    dout = dout.reshape(N * G, -1)
    # バニラレイヤーノルムバックプロパゲーションを実行する
    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    # dx、dbeta、dgammaの形状を元に戻す
    dx = dx.reshape(N, C, H, W)
    dbeta = dbeta[None, :, None, None]
    dgamma = dgamma[None, :, None, None]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
