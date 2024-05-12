from __future__ import division, print_function

from future import standard_library

standard_library.install_aliases()
import os
import pickle as pickle
from builtins import object, range

import numpy as np

from cs231n import optim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.

    ソルバーは、分類モデルの学習に必要なすべてのロジックをカプセル化する。
    をカプセル化する。ソルバーは確率的勾配降下を実行します。
    を実行します。

    ソルバーは訓練データと検証データの両方とラベルを受け取ることができます。
    定期的に分類精度をチェックすることができます。

    モデルを学習するには、まずソルバーのインスタンスを作成します。
    モデル、データセット、さまざまなオプション（学習率、バッチサイズなど）をコンストラクタに渡す。
    コンストラクタに渡す。その後、train()メソッドを呼び出して最適化手順を実行し、モデルを学習します。。

    train()メソッドが返された後、model.paramsには検証セットで最も良い結果を出したパラメータが格納されます。
    が含まれます。
    さらに、インスタンス変数solver.loss_historyには、トレーニング中に発生したすべての損失と
    インスタンス変数solver.train_acc_historyとsolver.val_acc_historyには、トレーニング中に発生したすべての損失のリストが格納されます。
    インスタンス変数solver.train_acc_historyとsolver.val_acc_historyは、各エポックにおける訓練セットと検証セットでのモデルの精度のリストになります。

    例えば、次のようになります:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()

    ソルバーは、以下のAPIに準拠しなければならないモデルオブジェクト上で動作する:

    - model.paramsは、文字列のパラメータ名をパラメータ値を含むnumpy
      配列にマッピングした辞書でなければならない。

    - model.loss(X,y)は、訓練時間の損失と勾配を計算する関数でなければなりません。
      勾配を計算する関数でなければなりません。
      を計算する関数でなければなりません：

      入力
      - X: 入力： X: 形状(N, d_1, ..., d_k)の入力データのミニバッチを与える配列。
      - y: y[i]はX[i]のラベルである。
        X[i]に対するラベル。

      戻り値：
      y が None の場合、テストタイムフォワードパスを実行し、リターンする：
      - のスコアを返す： X の分類スコアを与える shape (N, C) の配列。
        scores[i, c] は X[i] に対するクラス c のスコアを与える。

      y が None でない場合、訓練時間のフォワードパスとバックワードパスを実行し、次のタプルを返す。
      のタプルを返す：
      - loss: 損失を与えるスカラー
      - grads: パラメータ
        の名前を、それらのパラメータに対する損失の勾配にマッピングする。
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.

        新しいソルバー・インスタンスを構築する。

        必要な引数
        - モデル： 上記のAPIに準拠したモデルオブジェクト。
        - データ： トレーニングデータと検証データの辞書：
          'X_train': 'X_train': 訓練画像の配列、形状 (N_train, d_1, ..., d_k)
          'X_val': 配列，検証画像の形状 (N_val, d_1, ..., d_k).
          'y_train': 配列，形状 (N_train,).
          'y_val': 配列，検証画像のラベルの形状 (N_val,)

        オプションの引数：
        - update_rule: optim.pyの更新ルールの名前を指定する文字列。
          デフォルトは 'sgd' です。
        - optim_config: 選択された更新ルールに渡されるハイパーパラメータを含む辞書。
          ハイパーパラメータを含む辞書です。各更新ルールには異なる
          (optim.pyを参照)ですが、全ての更新ルールに'learning_rate'パラメータが必要です。
        - lr_decay: 学習率の減衰を表すスカラーです。
          各エポックの後、学習率にこの値が掛けられる。
        - batch_size: 学習時の損失と勾配の計算に使われるミニバッチのサイズ。
          の計算に使用するミニバッチのサイズ。
        - num_epochs: 学習中に実行するエポック数。
        - print_every: 整数。
          print_every反復ごとにトレーニングの損失が表示される。
        - verbose: ブール値; falseに設定すると、トレーニング中の出力は表示されない。
          を出力しない。
        - num_train_samples: トレーニングの精度をチェックするために使用するトレーニングサンプルの数。
          デフォルトは 1000; None に設定するとトレーニングセット全体を使用する。
        - num_val_samples: 値
          デフォルトは None で、検証セット全体を使用します。
        - checkpoint_name: Noneでない場合、エポック毎にモデルのチェックポイントを保存します。
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.

        提供されたデータでモデルの精度をチェックする。

        入力
        - X: データの配列、形状は (N, d_1, ..., d_k)
        - y: ラベルの配列, 形状は (N,).
        - num_samples: Noneでない場合、データをサブサンプリングし、num_samples個のデータに対してのみモデルをテストする
          num_samples データポイントに対してのみモデルをテストする。
        - batch_size: Xとyをこのサイズのバッチに分割する。
          メモリを使いすぎないようにする。

        戻り値
        - acc: モデルによって正しく分類されたインスタンスの割合を示すスカラー。
          インスタンスの割合を示すスカラー。
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
