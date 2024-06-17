from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass

    アフィン変換の後にReLUを実行する便利なレイヤ。

    入力
    - x: アフィン層への入力
    - w, b: アフィン層の重み

    のタプルを返す：
    - out: ReLUからの出力
    - キャッシュ： バックワードパスに渡すオブジェクト
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer."""
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def generic_forward(
    x, w, b, gamma=None, beta=None, bn_param=None, dropout_param=None, last=False
):
    """
    Convenience layer that performs an affine transform, a batch/layer normalization
    if needed, a ReLU, and dropout if needed.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Scale and shift params for the batch normalization
    - bn_param: Dictionary of required BN parameters
    - dropout_param: Dictionary of required Dropout parameters
    - last: Indicates wether to perform just affine forward

    Returns a tuple of:
    - out: Output from the ReLU or Dropout
    - cache: Object to give to the backward pass

    アフィン変換、必要に応じてバッチ/レイヤー正規化、ReLU、必要に応じてドロップアウトを行う便利なレイヤー。
    ReLU、必要に応じてドロップアウトを行う。

    入力
    - x: アフィン層への入力
    - w, b: アフィン層の重み
    - gamma, beta: バッチ正規化のスケールとシフトパラメータ
    - bn_param: 必要なBNパラメータの辞書。
    - dropout_param: 必要なDropoutパラメータの辞書
    - last:アフィンフォワードのみを行うかどうかを示す。

    のタプルを返す：
    - out: ReLUまたはDropoutからの出力
    - キャッシュ： バックワードパスに渡すオブジェクト
    """
    # オプションのキャッシュを「なし」に初期化
    bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None

    # Affine forward is a must
    out, fc_cache = affine_forward(x, w, b)

    # レイヤーが最後でない場合
    if not last:
        # もし正規化レイヤーがあれば、出力を正規化する：
        # もしbn_paramがモード(train | test)であれば、バッチノルム(batchnorm)、
        # そうでなければレイノルム(layernorm)
        if bn_param is not None:
            if "mode" in bn_param:
                out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
            else:
                out, ln_cache = layernorm_forward(out, gamma, beta, bn_param)

        # 出力をアクティベーションに通す
        out, relu_cache = relu_forward(out)  # reluの実行

        # パラメータが与えられたら、ドロップアウトを使う
        if dropout_param is not None:
            out, dropout_cache = dropout_forward(out, dropout_param)

    # バックワードパス用にキャッシュを準備する
    cache = fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache

    return out, cache


def generic_backward(dout, cache):
    """
    Backward pass for the affine-bn/ln?-relu-dropout? convenience layer.
    """
    # normのパラメータを「なし」に初期化
    dgamma, dbeta = None, None

    # フォワード・パスからプラパード・キャッシュを取得
    fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache = cache

    # ドロップアウトが行われた場合
    if dropout_cache is not None:
        dout = dropout_backward(dout, dropout_cache)

    # relu が行われた場合
    if relu_cache is not None:
        dout = relu_backward(dout, relu_cache)

    # normが行われた場合
    if bn_cache is not None:
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    elif ln_cache is not None:
        dout, dgamma, dbeta = layernorm_backward(dout, ln_cache)

    # アフィン・バックワードは必須
    dx, dw, db = affine_backward(dout, fc_cache)

    return dx, dw, db, dgamma, dbeta


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass

    コンボリューションの後にReLUを実行する便利なレイヤ。

    入力
    - x: 畳み込み層への入力
    - w, b, conv_param: 畳み込み層の重みとパラメータ。

    のタプルを返す：
    - out: ReLUからの出力
    - キャッシュ： バックワードパスに渡すオブジェクト
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    conv-relu convenience layerのバックワードパス。
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """
    Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass

    コンボリューション、バッチ正規化、ReLUを実行する便利なレイヤ。

    入力
    - x: 畳み込み層への入力
    - w, b, conv_param: 畳み込み層の重みとパラメータ。
    - pool_param: プーリング層のパラメータ
    - gamma, beta: スケールとシフトを与える形状 (D2,) と (D2,) の配列。
      パラメータを与える配列．
    - bn_param: バッチ正規化のパラメータ辞書．

    のタプルを返す：
    - out: プーリング層からの出力
    - キャッシュ： バックワードパスに渡すオブジェクト
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """
    Backward pass for the conv-bn-relu convenience layer.
    conv-relu convenience layerのバックワードパス。
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass

    コンボリューション、ReLU、プールを実行する便利なレイヤ。

    入力
    - x: 畳み込み層への入力
    - w, b, conv_param: 畳み込み層の重みとパラメータ。
    - pool_param: プーリング層のパラメータ

    のタプルを返す：
    - out: プーリング層からの出力
    - cache: バックワードパスに渡すオブジェクト
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer.
    conv-relu-pool convenience layerのバクワードパス。
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
