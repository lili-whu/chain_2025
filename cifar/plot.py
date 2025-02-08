import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_over_epochs(data, legend_labels=None, title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy", colors=None):
    """
    绘制多个实验结果的折线图
    :param data: list of lists/arrays，包含多个折线的数据，每个列表代表一个实验的准确率随训练轮次的变化
    :param legend_labels: 可选，折线图的图例标签
    :param title: 图表标题
    :param xlabel: 横坐标标签
    :param ylabel: 纵坐标标签
    :param colors: 可选，指定每条折线的颜色
    """
    epoch = np.arange(1, len(data[0]) + 1)  # 默认10轮

    plt.figure(figsize=(10, 6))

    for i, acc in enumerate(data):
        label = legend_labels[i] if legend_labels else f"Experiment {i+1}"
        color = colors[i] if colors else None
        plt.plot(epoch, acc, label=label, color=color, linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(epoch)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()

    # 显示图表
    plt.show()

# 示例数据：假设你有3组实验数据
# 每组实验的准确率随训练轮次（epoch）的变化，10轮数据

data_example = [
    [0.45, 0.50, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84],  # 组1：普通聚合
    [0.40, 0.55, 0.60, 0.67, 0.71, 0.73, 0.76, 0.79, 0.81, 0.83],  # 组2：MQI聚合
    [0.43, 0.50, 0.57, 0.63, 0.67, 0.71, 0.74, 0.77, 0.80, 0.82]   # 组3：恶意节点20%
]

legend_labels = ["FedAvg", "AccWeight", "FedAvg with 20% Malicious"]
colors = ["#FF6347", "#4682B4", "#32CD32"]  # 分别为红色、蓝色、绿色

# 调用绘图函数
plot_accuracy_over_epochs(data_example, legend_labels=legend_labels, title="Different Aggregation Methods", colors=colors)
