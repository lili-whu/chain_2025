import tensorflow as tf
import numpy as np
import pickle
import os
import shutil

def get_cifar10():
    (train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.cifar10.load_data()
    train_imgs = train_imgs.astype("float32") / 255.0
    test_imgs  = test_imgs.astype("float32")  / 255.0
    # one-hot
    train_lbls = tf.keras.utils.to_categorical(train_lbls, 10)
    test_lbls  = tf.keras.utils.to_categorical(test_lbls, 10)
    return train_imgs, train_lbls, test_imgs, test_lbls

def poison_all_labels(labels):
    """随机将标签改成其他类别，用于恶意节点。"""
    num_samples = labels.shape[0]
    new_labels = np.zeros_like(labels)
    rnd = np.random.randint(0, 10, size=num_samples)
    new_labels[np.arange(num_samples), rnd] = 1
    return new_labels

def split_cifar_iid(train_x, train_y, test_x, test_y, split_sizes, malicious_indices):
    """按split_sizes分配train集, malicious_indices为恶意节点索引."""
    assert sum(split_sizes)==50000, "总和必须是50000"

    all_indices = np.arange(50000)
    np.random.shuffle(all_indices)

    ptr = 0
    client_datasets = []
    for cid, size in enumerate(split_sizes):
        idx = all_indices[ptr:ptr+size]
        ptr += size

        c_data = {
            "train_images": train_x[idx],
            "train_labels": train_y[idx],
            "test_images": test_x,   # 每个客户端可共享同一份测试集
            "test_labels": test_y,
        }
        # 若是恶意节点，就污染其训练标签
        if cid in malicious_indices:
            c_data["train_labels"] = poison_all_labels(c_data["train_labels"])

        client_datasets.append(c_data)
    return client_datasets

def save_experiment_data(client_datasets, outdir):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    for i, data in enumerate(client_datasets):
        with open(os.path.join(outdir, f"client_{i}.pkl"), "wb") as f:
            pickle.dump(data, f)

if __name__=="__main__":
    train_x, train_y, test_x, test_y = get_cifar10()
    # 4:4:2:2:1:1:1:1 + 2:2
    # => 10节点
    # => 8节点: [10000,10000,5000,5000,2500,2500,2500,2500], 2节点(恶意): [5000,5000]
    split_sizes = [10000,10000,5000,5000,2500,2500,2500,2500,5000,5000]
    # 最后俩(8,9)恶意
    malicious_indices = [8,9]

    clients = split_cifar_iid(train_x, train_y,
                              test_x, test_y,
                              split_sizes=split_sizes,
                              malicious_indices=malicious_indices)
    save_experiment_data(clients, "experiments/cifar_custom")
    print("已生成 10 个客户端, 存储在 experiments/cifar_custom 下.")
