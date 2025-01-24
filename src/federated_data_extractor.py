import tensorflow as tf

import pickle


def get_mnist():
    '''
    func to get mnist images dataset from tensorflow site
    '''
    # from tensorflow.examples.tutorials.mnist import input_data # type: ignore
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    # dataset = dict()
    # dataset["train_images"] = mnist.train.images
    # dataset["train_labels"] = mnist.train.labels
    # dataset["test_images"] = mnist.test.images
    # dataset["test_labels"] = mnist.test.labels
    # return dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    dataset = dict()
    # Normalize the images
    dataset["train_images"] = train_images.reshape(train_images.shape[0], 28 * 28).astype('float32') / 255
    dataset["train_labels"] = tf.keras.utils.to_categorical(train_labels, 10)
    dataset["test_images"] = test_images.reshape(test_images.shape[0], 28 * 28).astype('float32') / 255
    dataset["test_labels"] = tf.keras.utils.to_categorical(test_labels, 10)
    return dataset


def save_data(dataset, name="mnist.d"):
    '''
    Func to save mnist data in binary mode(its good to use binary mode)
    '''
    with open(name, "wb") as f:
        pickle.dump(dataset, f)


def load_data(name="mnist.d"):
    '''
    Func to load mnist data in binary mode(for reading also binary mode is important)
    '''
    with open(name, "rb") as f:
        return pickle.load(f)


def get_dataset_details(dataset):
    '''
    Func to display information on data
    '''
    # print(dataset)
    for k in dataset.keys():
        print(k, dataset[k].shape)
    print("get_dataset_details return")

def split_dataset(dataset, split_count):
    '''
    Function to split dataset to federated data slices as per specified count so as to try federated learning
    '''
    '''
    功能: 将数据集拆分为指定数量的部分，以便模拟联邦学习中的客户端数据分布。
    步骤:
    1.计算每个拆分部分的训练数据长度: split_data_length。
    2.对训练数据进行切片，分配给每个客户端。
    3.保持测试集在所有客户端相同，而训练集则根据指定数量进行分割。
    4.返回一个包含多个数据片段的列表。
    '''
    datasets = []
    split_data_length = len(dataset["train_images"]) // split_count
    for i in range(split_count):
        d = dict()
        d["test_images"] = dataset["test_images"][:]
        d["test_labels"] = dataset["test_labels"][:]
        d["train_images"] = dataset["train_images"][i * split_data_length:(i + 1) * split_data_length]
        d["train_labels"] = dataset["train_labels"][i * split_data_length:(i + 1) * split_data_length]
        datasets.append(d)
    return datasets


if __name__ == '__main__':
    save_data(get_mnist())
    dataset = load_data()
    get_dataset_details(dataset)
    for n, d in enumerate(split_dataset(dataset, 10)):
        save_data(d, "federated_data_" + str(n) + ".d")
        dk = load_data("federated_data_" + str(n) + ".d")
        get_dataset_details(dk)
        print()
