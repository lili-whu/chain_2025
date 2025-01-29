# NNWorker.py  (CIFAR + CNN 示例)
import tensorflow as tf
import numpy as np

def reset():
    tf.reset_default_graph()

class NNWorker:
    def __init__(self, X=None, Y=None, tX=None, tY=None, size=0, id="nn0", steps=2):
        self.id = id
        self.train_x = X
        self.train_y = Y
        self.test_x  = tX
        self.test_y  = tY
        self.size    = size

        self.learning_rate = 0.001
        self.num_steps     = steps  # epoch
        self.batch_size    = 64

        # CIFAR-10: shape=(32,32,3)
        self.image_height  = 32
        self.image_width   = 32
        self.num_channels  = 3
        self.num_classes   = 10

        self.sess = tf.Session()

    def build(self, base):
        """
        假设 base = { 'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b', 'fc_w', 'fc_b' } ...
        可以根据实际网络深度添加更多层
        """
        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.num_channels], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="Y")

        # 把 base 里的卷积核和偏置加载到 TF Variable
        self.weights = {
            'conv1_w': tf.Variable(base['c1w'], name="conv1_w"),
            'conv2_w': tf.Variable(base['c2w'], name="conv2_w"),
            'fc_w':    tf.Variable(base['fcw'], name="fc_w")
        }
        self.biases = {
            'conv1_b': tf.Variable(base['c1b'], name="conv1_b"),
            'conv2_b': tf.Variable(base['c2b'], name="conv2_b"),
            'fc_b':    tf.Variable(base['fcb'], name="fc_b")
        }

        # 构建 CNN
        conv1 = tf.nn.conv2d(self.X, self.weights['conv1_w'], strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, self.biases['conv1_b'])
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, self.weights['conv2_w'], strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, self.biases['conv2_b'])
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # shape变为 [None,8,8,64], 我们假设 conv2_w 输出通道=64
        flatten = tf.reshape(pool2, [-1, 8*8*64])

        # 全连接
        fc = tf.matmul(flatten, self.weights['fc_w']) + self.biases['fc_b']
        self.logits = fc  # 这里没再加一层激活, 可以加Relu + fc2

        pred = tf.argmax(self.logits, 1)
        labl = tf.argmax(self.Y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labl), tf.float32))

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def build_base(self):
        """
        初始化随机CNN参数，用于创世区块时
        """
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
        self.Y = tf.placeholder(tf.float32, [None, 10], name="Y")

        # conv1: kernel shape [3,3,3,64] => out channels=64
        conv1_w = tf.random_normal([3,3,3,64], stddev=0.01)
        conv1_b = tf.random_normal([64], stddev=0.01)
        conv2_w = tf.random_normal([3,3,64,64], stddev=0.01)
        conv2_b = tf.random_normal([64], stddev=0.01)
        fc_w    = tf.random_normal([8*8*64, 10], stddev=0.01)
        fc_b    = tf.random_normal([10], stddev=0.01)

        self.weights = {
            'conv1_w': tf.Variable(conv1_w, name='conv1_w'),
            'conv2_w': tf.Variable(conv2_w, name='conv2_w'),
            'fc_w':    tf.Variable(fc_w,    name='fc_w')
        }
        self.biases = {
            'conv1_b': tf.Variable(conv1_b, name='conv1_b'),
            'conv2_b': tf.Variable(conv2_b, name='conv2_b'),
            'fc_b':    tf.Variable(fc_b,    name='fc_b')
        }

        # CNN forward
        conv1 = tf.nn.conv2d(self.X, self.weights['conv1_w'], strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, self.biases['conv1_b'])
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, [1,2,2,1],[1,2,2,1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, self.weights['conv2_w'], strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, self.biases['conv2_b'])
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, [1,2,2,1],[1,2,2,1], padding='SAME')

        flatten = tf.reshape(pool2, [-1, 8*8*64])
        fc = tf.matmul(flatten, self.weights['fc_w']) + self.biases['fc_b']
        self.logits = fc

        pred = tf.argmax(self.logits,1)
        labl = tf.argmax(self.Y,1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,labl), tf.float32))

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        """
        mini-batch 训练
        """
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.Y
            )
        )
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        if self.train_x is None or len(self.train_x)==0:
            print("No training data!")
            return

        num_samples = self.train_x.shape[0]
        for epoch in range(self.num_steps):
            perm = np.random.permutation(num_samples)
            train_x_shuff = self.train_x[perm]
            train_y_shuff = self.train_y[perm]

            batch_count = num_samples // self.batch_size
            avg_loss = 0.0
            for b in range(batch_count):
                start = b*self.batch_size
                end   = start + self.batch_size
                bx = train_x_shuff[start:end]
                by = train_y_shuff[start:end]

                _, loss_val = self.sess.run([self.train_op, self.loss_op],
                                            feed_dict={self.X: bx, self.Y: by})
                avg_loss += loss_val / batch_count

            # 每 epoch 打印一次
            acc_val = self.sess.run(self.accuracy,
                feed_dict={self.X: self.train_x, self.Y: self.train_y})
            print(f"Epoch {epoch+1}/{self.num_steps}, loss={avg_loss:.4f}, train_acc={acc_val:.3f}")

    def evaluate(self):
        """在测试集上计算准确率"""
        if self.test_x is None or len(self.test_x)==0:
            return 0.0
        return self.sess.run(self.accuracy, feed_dict={self.X: self.test_x, self.Y: self.test_y})

    def get_model(self):
        """
        获取参数并打包成字典:
        {
          'c1w': ...,
          'c1b': ...,
          'c2w': ...,
          'c2b': ...,
          'fcw': ...,
          'fcb': ...,
          'size': ...
        }
        """
        var_dict = {}
        for v in tf.trainable_variables():
            v_name = v.name
            # v.name 可能是 'conv1_w:0', 'conv1_b:0', ...
            if 'conv1_w' in v_name:
                var_dict['c1w'] = v.eval(self.sess)
            elif 'conv1_b' in v_name:
                var_dict['c1b'] = v.eval(self.sess)
            elif 'conv2_w' in v_name:
                var_dict['c2w'] = v.eval(self.sess)
            elif 'conv2_b' in v_name:
                var_dict['c2b'] = v.eval(self.sess)
            elif 'fc_w' in v_name:
                var_dict['fcw'] = v.eval(self.sess)
            elif 'fc_b' in v_name:
                var_dict['fcb'] = v.eval(self.sess)

        var_dict["size"] = self.size
        return var_dict

    def close(self):
        self.sess.close()
