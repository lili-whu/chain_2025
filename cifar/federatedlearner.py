import tensorflow as tf
import numpy as np


test_acc_list = []
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
        假设 base = { 'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b', 'conv3_w', 'conv3_b', 'fc_w', 'fc_b' } ...
        可以根据实际网络深度添加更多层
        """
        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.num_channels], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="Y")

        # 把 base 里的卷积核和偏置加载到 TF Variable
        self.weights = {
            'conv1_w': tf.Variable(base['c1w'], name="conv1_w"),
            'conv2_w': tf.Variable(base['c2w'], name="conv2_w"),
            'conv3_w': tf.Variable(base['c3w'], name="conv3_w"),
            'fc_w':    tf.Variable(base['fcw'], name="fc_w")
        }
        self.biases = {
            'conv1_b': tf.Variable(base['c1b'], name="conv1_b"),
            'conv2_b': tf.Variable(base['c2b'], name="conv2_b"),
            'conv3_b': tf.Variable(base['c3b'], name="conv3_b"),
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

        conv3 = tf.nn.conv2d(pool2, self.weights['conv3_w'], strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, self.biases['conv3_b'])
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # shape变为 [None,4,4,256], 我们假设 conv3_w 输出通道=256
        flatten = tf.reshape(pool3, [-1, 4*4*256])

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
        conv2_w = tf.random_normal([3,3,64,128], stddev=0.01)
        conv2_b = tf.random_normal([128], stddev=0.01)
        conv3_w = tf.random_normal([3,3,128,256], stddev=0.01)
        conv3_b = tf.random_normal([256], stddev=0.01)
        fc_w    = tf.random_normal([4*4*256, 10], stddev=0.01)
        fc_b    = tf.random_normal([10], stddev=0.01)

        self.weights = {
            'conv1_w': tf.Variable(conv1_w, name='conv1_w'),
            'conv2_w': tf.Variable(conv2_w, name='conv2_w'),
            'conv3_w': tf.Variable(conv3_w, name='conv3_w'),
            'fc_w':    tf.Variable(fc_w,    name='fc_w')
        }
        self.biases = {
            'conv1_b': tf.Variable(conv1_b, name='conv1_b'),
            'conv2_b': tf.Variable(conv2_b, name='conv2_b'),
            'conv3_b': tf.Variable(conv3_b, name='conv3_b'),
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

        conv3 = tf.nn.conv2d(pool2, self.weights['conv3_w'], strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, self.biases['conv3_b'])
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, [1,2,2,1],[1,2,2,1], padding='SAME')

        flatten = tf.reshape(pool3, [-1, 4*4*256])
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
            acc_val = self.evaluate(self.train_x, self.train_y, batch_size=512)
            print(f"Epoch {epoch+1}/{self.num_steps}, loss={avg_loss:.4f}, train_acc={acc_val:.3f}")

            test_acc = self.evaluate()
            test_acc_list.append(test_acc)
            print("test_acc_list:", test_acc_list)
            print(f"epoch:{epoch:.4f}, \n测试集准确率: {test_acc:.4f}")


    def evaluate(self, data_x=None, data_y=None, batch_size=512):
        if data_x is None or data_y is None:
            data_x, data_y = self.test_x, self.test_y

        """分批计算准确率"""
        total_acc = 0.0
        num_samples = data_x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = data_x[start:end]
            batch_y = data_y[start:end]

            acc = self.sess.run(self.accuracy,
                                feed_dict={self.X: batch_x, self.Y: batch_y})
            total_acc += acc * batch_x.shape[0]

        return total_acc / num_samples


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
            elif 'conv3_w' in v_name:
                var_dict['c3w'] = v.eval(self.sess)
            elif 'conv3_b' in v_name:
                var_dict['c3b'] = v.eval(self.sess)
            elif 'fc_w' in v_name:
                var_dict['fcw'] = v.eval(self.sess)
            elif 'fc_b' in v_name:
                var_dict['fcb'] = v.eval(self.sess)

        var_dict["size"] = self.size
        return var_dict

    def close(self):
        self.sess.close()

def load_and_preprocess_data():
    # 加载CIFAR-10数据集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # 归一化像素值到[0,1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # 将标签转换为one-hot编码
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return train_images, train_labels, test_images, test_labels

def main():
    # 加载和预处理数据
    train_x, train_y, test_x, test_y = load_and_preprocess_data()

    # 初始化神经网络工作节点
    worker = NNWorker(
        X=train_x,
        Y=train_y,
        tX=test_x,
        tY=test_y,
        size=len(train_x),  # 训练集样本数
        id="cifar_cnn",
        steps=10  # 设置训练轮数
    )
    print(2)
    # 构建初始模型（随机初始化参数）
    worker.build_base()

    # 开始训练
    worker.train()

    # 在测试集上评估模型
    test_acc = worker.evaluate(test_x, test_y, batch_size=512)
    print(f"\n测试集准确率: {test_acc:.4f}")

    # 关闭会话
    worker.close()

if __name__ == "__main__":
    main()