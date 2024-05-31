import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.

このファイルは、ニューラルネットワークのトレーニングによく使われるさまざまな一次更新ルールを実装している。
を実装している。各更新ルールは現在の重みと
を受け取り、次の重みのセットを生成する。
を生成する。各更新ルールは同じインターフェースを持っている：

def update(w, dw, config=None):

入力
  - w: 現在の重みを与えるnumpy配列。
  - dw: wと同じ形のnumpy配列。
    の勾配を与える。
  - config: 学習速度、運動量などのハイパーパラメータ値を含む辞書。
    率、運動量などのハイパーパラメータ値を含む辞書。更新ルールが、何回にもわたって値をキャッシュする必要がある場合
    の繰り返しで値をキャッシュする必要がある場合、 config はこれらのキャッシュ値も保持する。

戻り値
  - next_w: 更新後の次のポイント。
  - config: 更新ルールの次の繰り返しに渡されるconfig辞書。
    に渡される設定辞書。

注意: ほとんどの更新ルールでは、デフォルトの学習率ではおそらくうまくいかないだろう。
しかし、他のハイパーパラメータのデフォルト値は、さまざまな問題でうまく機能するはずである。

効率化のために、更新ルールはインプレース更新を行い、wを変異させ
next_wをwに等しく設定する。
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.

    バニラ確率勾配降下を実行する。

    設定フォーマット
    - learning_rate: スカラー学習率
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.

    運動量を用いた確率的勾配降下を行う。

    設定フォーマット
    - learning_rate: スカラー学習速度。
    - 運動量： 0から1の間のスカラーで、運動量の値を指定する。
      momentum = 0にするとsgdになる。
    - velocity: 速度： wとdwと同じ形のnumpy配列。
      勾配の移動平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    #                                                                         #
    # 運動量の更新式を実装する。更新された値をnext_w変数に格納する。
    # 速度 v も使用して更新する。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 更新速度 = 運動量 * 前回の更新速度 - 学習率 * 勾配
    v = config["momentum"] * v - config["learning_rate"] * dw
    # 位置を更新
    next_w = w + v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.

    RMSProp更新ルールを使用する。
    勾配値の移動平均を使用するRMSProp更新規則を使用する。

    設定フォーマット
    - learning_rate: スカラー学習率。
    - decay_rate: 2乗勾配キャッシュの減衰率を0から1の間のスカラーで指定する。
      勾配キャッシュの減衰率を与える。
    - epsilon: ゼロ除算を避けるための平滑化に使用する小さなスカラー。
    - キャッシュ： 勾配の二次モーメントの移動平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    #                                                                         #
    # RMSpropの更新式を実装し、変数next_wにwの次の値を格納する。
    # config['cache']に格納されているキャッシュ値を更新することを忘れないでください。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # キーで値を取得
    keys = ["learning_rate", "decay_rate", "epsilon", "cache"]
    lr, dr, eps, cache = (config.get(key) for key in keys)

    # cacheを更新
    config["cache"] = dr * cache + (1 - dr) * dw**2

    next_w = w - lr * dw / (np.sqrt(config["cache"]) + eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.

    アダム更新ルールを使用。
    勾配とその2乗の移動平均とバイアス補正項を組み込んだアダム更新ルールを使用する。

    設定フォーマット
    - learning_rate: スカラー学習率。
    - beta1: 勾配の最初のモーメントの移動平均の減衰率。
    - beta2: 勾配の第2モーメントの移動平均の減衰率。
    - epsilon: ゼロ除算を避けるための平滑化に使用される小さなスカラー。
    - m: 勾配の移動平均。
    - v: 2乗勾配の移動平均。
    - t: 繰り返し数.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    #                                                                         #
    # アダム更新式を実装し、変数 next_w に w の次の値を格納する。
    # configに格納されているm、v、t変数の更新も忘れずに。
    # 注：基準出力と一致させるため、計算に使用する前にtを修正してください。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # キーで値を取得
    keys = [
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "m",
        "v",
        "t",
    ]
    lr, beta1, beta2, eps, m, v, t = (config.get(k) for k in keys)

    # 反復カウンタ
    config["t"] = t = t + 1

    # 勾配平滑化（モーメンタム）
    config["m"] = m = beta1 * m + (1 - beta1) * dw

    # mのバイアス補正
    mt = m / (1 - beta1**t)

    # 勾配平滑化 (RMSprop)
    config["v"] = v = beta2 * v + (1 - beta2) * (dw**2)

    # vのバイアス補正
    vt = v / (1 - beta2**t)

    next_w = w - lr * mt / (np.sqrt(vt) + eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
