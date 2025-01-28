"""
 - Blockchain for Federated Learning -
      Federated Learning Script
"""
import tensorflow as tf
import numpy as np
import pickle


def reset():
    # 这在 TF1.x 环境下用于清理默认图，避免多次运行冲突
    tf.reset_default_graph()


class NNWorker:
    def __init__(self, X=None, Y=None, tX=None, tY=None, size=0, id="nn0", steps=10):
        """
        :param X: 训练数据 features
        :param Y: 训练数据 labels (one-hot)
        :param tX: 测试数据 features
        :param tY: 测试数据 labels (one-hot)
        :param size: 数据量大小(可选，用于记录)
        :param id: 节点/客户端 ID
        :param steps: 这里指训练的 epoch 数，默认=10
        """
        self.id = id
        self.train_x = X
        self.train_y = Y
        self.test_x = tX
        self.test_y = tY
        self.size = size

        # 超参数
        self.learning_rate = 0.01      # Adam 的学习率，酌情设置
        self.num_steps = steps         # 训练epoch数
        self.batch_size = 64          # mini-batch 大小，可灵活调整

        # 网络结构相关
        self.n_hidden_1 = 256
        self.n_hidden_2 = 256
        self.num_input = 784
        self.num_classes = 10

        # 构建 TF 会话 (在 TF1.x)
        self.sess = tf.Session()

    def build(self, base):
        """
        使用给定的 base 参数(来自区块链等)初始化网络。
        base 包含 'w1','w2','wo','b1','b2','bo' 等张量
        """
        self.X = tf.placeholder("float", [None, self.num_input], name="X")
        self.Y = tf.placeholder("float", [None, self.num_classes], name="Y")

        self.weights = {
            'w1': tf.Variable(base['w1'], name="w1"),
            'w2': tf.Variable(base['w2'], name="w2"),
            'wo': tf.Variable(base['wo'], name="wo")
        }
        self.biases = {
            'b1': tf.Variable(base['b1'], name="b1"),
            'b2': tf.Variable(base['b2'], name="b2"),
            'bo': tf.Variable(base['bo'], name="bo")
        }

        # 前向传播
        layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['w2']), self.biases['b2'])
        self.logits = tf.matmul(layer_2, self.weights['wo']) + self.biases['bo']

        # 准确率
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 初始化
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def build_base(self):
        """
        初始化随机的网络参数，用于创世区块或第一次无模型时。
        """
        self.X = tf.placeholder("float", [None, self.num_input], name="X")
        self.Y = tf.placeholder("float", [None, self.num_classes], name="Y")

        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1]), name="w1"),
            'w2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name="w2"),
            'wo': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]), name="wo")
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name="b1"),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name="b2"),
            'bo': tf.Variable(tf.random_normal([self.num_classes]), name="bo")
        }

        # 前向传播
        layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['w2']), self.biases['b2'])
        self.logits = tf.matmul(layer_2, self.weights['wo']) + self.biases['bo']

        # 准确率
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 初始化
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        """
        实现真正的 mini-batch 训练:
          1) 每个 epoch 开头对 train_x, train_y 进行 shuffle
          2) 分批 (batch_size) 训练
        """
        # 定义损失和优化器
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.Y
            )
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # 重新初始化一下参数(必要时，看你是否需要保留之前build的值)
        # 如果你希望保留原始 build() 中的变量值，可以注释掉这行
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        num_samples = self.train_x.shape[0]

        for epoch in range(1, self.num_steps + 1):
            # 1. 打乱索引
            perm = np.random.permutation(num_samples)
            train_x_shuff = self.train_x[perm]
            train_y_shuff = self.train_y[perm]

            # 2. mini-batch训练
            num_batches = num_samples // self.batch_size
            avg_loss = 0.0
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = start_idx + self.batch_size

                bx = train_x_shuff[start_idx:end_idx]
                by = train_y_shuff[start_idx:end_idx]

                _, batch_loss = self.sess.run(
                    [self.train_op, self.loss_op],
                    feed_dict={self.X: bx, self.Y: by}
                )
                avg_loss += batch_loss / num_batches

            # 3. 可在每个 epoch 结束后，查看一下训练集准确率
            train_acc = self.sess.run(
                self.accuracy,
                feed_dict={self.X: self.train_x, self.Y: self.train_y}
            )
            print(f"Epoch {epoch}, Loss= {avg_loss:.4f}, Train_Accuracy= {train_acc:.3f}")

    def evaluate(self):
        """
        在测试数据集上计算准确率
        """
        return self.sess.run(self.accuracy, feed_dict={
            self.X: self.test_x,
            self.Y: self.test_y
        })

    def get_model(self):
        """
        获取当前网络的可训练参数
        """
        # 这里将每个 Variable 的前2字符('w1','b1',...)当key, 也可以更灵活
        varsk = {}
        tv = tf.trainable_variables()
        for v in tv:
            # v.name 形如 "w1:0", 取前2字符
            key = v.name[:2]
            varsk[key] = v.eval(self.sess)
        varsk["size"] = self.size
        return varsk

    def close(self):
        """
        关闭会话
        """
        self.sess.close()
