import tensorflow as tf
import numpy as np
import pickle
import os
import shutil

def get_mnist():
    """加载并预处理MNIST数据集"""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    dataset = {
        "train_images": train_images.reshape(-1, 28*28).astype("float32") / 255.0,
        "train_labels": tf.keras.utils.to_categorical(train_labels, 10),
        "test_images": test_images.reshape(-1, 28*28).astype("float32") / 255.0,
        "test_labels": tf.keras.utils.to_categorical(test_labels, 10)
    }
    return dataset

def poison_all_labels(labels):
    """完全污染标签（随机生成新标签）"""
    num_samples = labels.shape[0]
    poisoned_labels = np.zeros_like(labels)
    random_labels = np.random.randint(0, 10, size=num_samples)
    poisoned_labels[np.arange(num_samples), random_labels] = 1
    return poisoned_labels

def split_iid_with_malicious(dataset, split_sizes, malicious_indices):
    """IID数据划分并标记恶意节点"""
    assert sum(split_sizes) == 60000, "总数据量必须为60000"

    all_indices = np.arange(60000)
    np.random.shuffle(all_indices)

    client_datasets = []
    ptr = 0
    for size in split_sizes:
        indices = all_indices[ptr:ptr+size]
        ptr += size

        client_data = {
            "test_images": dataset["test_images"],
            "test_labels": dataset["test_labels"],
            "train_images": dataset["train_images"][indices],
            "train_labels": dataset["train_labels"][indices],
            "original_indices": indices
        }

        if len(client_datasets) in malicious_indices:
            client_data["train_labels"] = poison_all_labels(client_data["train_labels"])

        client_datasets.append(client_data)

    return client_datasets

def save_experiment_data(client_datasets, output_dir):
    """保存实验数据到指定文件夹"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for idx, data in enumerate(client_datasets):
        with open(os.path.join(output_dir, f"client_{idx}.pkl"), "wb") as f:
            pickle.dump(data, f)

# 实验配置
experiments = {
    "centralized": {
        "split_sizes": [60000],  # 单节点包含全部数据
        "malicious_indices": []
    },
    "federated_normal": {
        "split_sizes": [6000]*10,  # 10个均等节点
        "malicious_indices": []
    },
    "federated_20malicious": {
        "split_sizes": [6000]*10,
        "malicious_indices": [8, 9]  # 最后2个为恶意节点
    },
    "federated_50malicious": {
        "split_sizes": [6000]*10,
        "malicious_indices": [5,6,7,8,9]  # 最后5个为恶意节点
    }
}

# 主流程
dataset = get_mnist()

for exp_name, config in experiments.items():
    print(f"\n=== 正在生成实验组: {exp_name} ===")
    clients = split_iid_with_malicious(
        dataset,
        split_sizes=config["split_sizes"],
        malicious_indices=config["malicious_indices"]
    )

    # 保存数据
    save_experiment_data(clients, os.path.join("experiments", exp_name))

    # 验证输出
    print(f"已保存到: experiments/{exp_name}")
    print(f"客户端数量: {len(clients)}")
    print(f"恶意节点索引: {config['malicious_indices']}")
    # 验证恶意节点标签
    if config["malicious_indices"]:
        mal_client = clients[config["malicious_indices"][0]]
        original_labels = np.argmax(dataset["train_labels"][mal_client["original_indices"]], axis=1)
        poisoned_labels = np.argmax(mal_client["train_labels"], axis=1)
        match_rate = np.mean(original_labels == poisoned_labels)
        print(f"恶意节点标签匹配率: {match_rate:.4f} (应接近0.1)")

print("\n=== 所有实验数据生成完成 ===")