import tensorflow as tf
import numpy as np
import pickle
import os

def get_mnist():
    """
    加载MNIST数据集并预处理
    Returns:
        dict: 包含归一化图像和one-hot标签的字典
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    dataset = dict()
    # 归一化 + 扁平化 + one-hot编码
    dataset["train_images"] = train_images.reshape(-1, 28*28).astype('float32') / 255.0
    dataset["train_labels"] = tf.keras.utils.to_categorical(train_labels, 10)
    dataset["test_images"] = test_images.reshape(-1, 28*28).astype('float32') / 255.0
    dataset["test_labels"] = tf.keras.utils.to_categorical(test_labels, 10)
    return dataset

def poison_all_labels(labels):
    """
    完全污染所有标签（随机生成新标签）
    Args:
        labels (np.ndarray): 原始one-hot标签，形状为(num_samples, 10)
    Returns:
        np.ndarray: 污染后的标签
    """
    num_samples = labels.shape[0]
    poisoned_labels = np.zeros_like(labels)
    # 为每个样本生成随机标签
    random_labels = np.random.randint(0, 10, size=num_samples)
    # 转换为one-hot格式
    poisoned_labels[np.arange(num_samples), random_labels] = 1
    return poisoned_labels

def split_iid_with_malicious(dataset, split_sizes=[12000, 12000, 6000, 6000, 3000, 3000, 3000, 3000, 6000, 6000], malicious_indices=[8, 9]):
    """
    IID数据分配（完全随机划分）并标记恶意节点
    Args:
        dataset (dict): MNIST数据集字典
        split_sizes (list): 每个客户端的数据量分配
        malicious_indices (list): 恶意节点的索引（从0开始）
    Returns:
        list: 包含10个客户端数据的列表
    """
    # 验证总数据量
    total = sum(split_sizes)
    assert total == 60000, f"总数据量应为60000，当前分配总和为{total}"

    # 随机打乱所有训练数据的索引
    all_indices = np.arange(len(dataset["train_images"]))
    np.random.shuffle(all_indices)

    # 按split_sizes切割索引
    split_indices = []
    ptr = 0
    for size in split_sizes:
        split_indices.append(all_indices[ptr:ptr+size])
        ptr += size

    # 构建客户端数据集
    client_datasets = []
    for i, indices in enumerate(split_indices):
        client_data = {
            "test_images": dataset["test_images"],
            "test_labels": dataset["test_labels"],
            "train_images": dataset["train_images"][indices],
            "train_labels": dataset["train_labels"][indices],
            "original_indices": indices  # 新增字段，保存原始索引
        }
        # 如果是恶意节点，污染所有标签
        if i in malicious_indices:
            client_data["train_labels"] = poison_all_labels(client_data["train_labels"])
        client_datasets.append(client_data)

    return client_datasets

def save_data(data, filename):
    """
    保存数据到文件
    Args:
        data (dict): 客户端数据字典
        filename (str): 保存路径
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # 加载完整MNIST数据集
    dataset = get_mnist()

    # 定义客户端数据分配（10个节点）
    split_sizes = [12000, 12000, 6000, 6000, 3000, 3000, 3000, 3000, 6000, 6000]
    malicious_indices = [8, 9]  # 最后两个节点为恶意节点

    # 分割数据集
    client_datasets = split_iid_with_malicious(dataset, split_sizes, malicious_indices)

    # 保存分片数据
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    for idx, client_data in enumerate(client_datasets):
        filename = os.path.join(output_dir, f"client_{idx}.pkl")
        save_data(client_data, filename)

        # 打印客户端信息
        print(f"Client {idx}:")
        print(f"  Train size: {len(client_data['train_images'])}")
        print(f"  Malicious: {idx in malicious_indices}")

        # 验证恶意节点标签污染
        if idx in malicious_indices:
            # 使用保存的原始索引获取原始标签
            original_labels = np.argmax(dataset["train_labels"][client_data["original_indices"]], axis=1)
            poisoned_labels = np.argmax(client_data["train_labels"], axis=1)
            # 计算标签匹配率（应该接近10%随机概率）
            match_rate = np.mean(original_labels == poisoned_labels)
            print(f"  Label Match Rate: {match_rate:.4f} (expected ~0.1)")
        print("-" * 50)