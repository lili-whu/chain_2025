import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pylab import mpl

from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,

    'font.family': 'SimSun'  # 更改为中文字体
})

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_style(rc={'font.sans-serif': "SimSun"})
epochs = np.arange(0, 11)

# 数据定义（保持原始精度）
data = {
    # 集中学习（取前10轮）
    "central": [0.1000, 0.4846,0.5556,0.6370,0.6683,0.6777,0.7052,0.7222,0.7081,0.7218,0.7205], 
    # 普通联邦
    "fed_normal": [0.1000,0.1309,0.5604,0.6406,0.6744,0.6875,0.6943,0.6987,0.6996,0.6992,0.7015],
    # Acc 联邦
    "acc_normal": [0.1000,0.1047,0.5639,0.6458,0.6761,0.6881,0.6992,0.7011,0.7057,0.7078,0.7082],
    # FedAvg + 20%恶意
    "fed_20": [0.1000,0.1004,0.4500,0.5700,0.6040,0.6176,0.6172,0.6143,0.6161,0.6171,0.6172],
    # FedAvg + 50%恶意
    "fed_50": [0.1000,0.1000,0.1086,0.3623,0.4273,0.4102,0.4043,0.4026,0.3967,0.3936,0.3878],
    # AccWeight + 20%恶意
    "acc_20": [0.1000,0.1376,0.5849,0.6479,0.6724,0.6841,0.6910,0.6934,0.6956,0.6982,0.6981],
    # AccWeight + 50%恶意 
    "acc_50": [0.1000,0.1871,0.5727,0.6352,0.6574,0.6634,0.6652,0.6650,0.6639,0.6663,0.6651],
    # 非均等划分数据
    "fed_20_uneven": [0.1000,0.1087,0.5004,0.5757,0.6068,0.6348,0.6337,0.6328,0.6329,0.6341,0.6287],
    "acc_20_uneven": [0.1000,0.1187,0.5676,0.6555,0.6851,0.6916,0.6931,0.6958,0.6977,0.7003,0.7009]
}

# ============= 图1：FedAvg对比 =============
plt.figure(figsize=(8,5))
plt.plot(epochs, data["central"], 'k--', marker='s', markersize=6, linewidth=2, label='集中学习')
plt.plot(epochs, data["fed_normal"], 'b-', marker='o', markersize=6, linewidth=2, label='FedAvg（正常节点）')
plt.plot(epochs, data["fed_20"], 'r-.', marker='^', markersize=6, linewidth=2, label='FedAvg（20%恶意节点）')
plt.plot(epochs, data["fed_50"], 'g:', marker='D', markersize=6, linewidth=2, label='FedAvg（50%恶意节点）')

plt.title('FedAvg聚合方法在不同恶意节点比例下的性能对比（均等划分）')
plt.xlabel('训练轮次')
plt.ylabel('测试集准确率')
plt.xticks(epochs)
plt.ylim(0.0, 0.8)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()

# ============= 图2：AccWeight对比 =============
plt.figure(figsize=(8,5))
plt.plot(epochs, data["central"], 'k--', marker='s', markersize=6, linewidth=2, label='集中学习')
plt.plot(epochs, data["acc_normal"], 'b-', marker='o', markersize=6, linewidth=2, label='AccWeight（正常节点）')
plt.plot(epochs, data["acc_20"], 'r-.', marker='^', markersize=6, linewidth=2, label='AccWeight（20%恶意节点）')
plt.plot(epochs, data["acc_50"],'g:', marker='D', markersize=6, linewidth=2, label='AccWeight（50%恶意节点）')

plt.title('AccWeight聚合方法在不同恶意节点比例下的性能对比（均等划分）')
plt.xlabel('训练轮次')
plt.ylabel('测试集准确率')
plt.xticks(epochs)
plt.ylim(0.0, 0.8)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()

# ============= 图3：均等划分对比 =============
plt.figure(figsize=(8,5))
plt.plot(epochs, data["fed_20"], 'k--', marker='s', markersize=6, linewidth=2, label='FedAvg（20%恶意节点）')
plt.plot(epochs, data["acc_20"],  'b-', marker='o', markersize=6, linewidth=2,  label='AccWeight（20%恶意节点）')
plt.plot(epochs, data["fed_50"],'r-.', marker='^', markersize=6, linewidth=2, label='FedAvg（50%恶意节点）')
plt.plot(epochs, data["acc_50"], 'g:', marker='D', markersize=6, linewidth=2, label='AccWeight（50%恶意节点）')

plt.title('两种聚合方法在恶意节点影响下的模型性能对比（均等划分）')
plt.xlabel('训练轮次')
plt.ylabel('测试集准确率')
plt.xticks(epochs)
plt.ylim(0.0, 0.8)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()

# ============= 图4：非均等划分对比 =============
plt.figure(figsize=(8,5))
plt.plot(epochs, data["fed_20_uneven"], 'k--', marker='s', markersize=6, linewidth=2,  label='FedAvg（20%恶意节点）')
plt.plot(epochs, data["acc_20_uneven"], 'b-', marker='o', markersize=6, linewidth=2,  label='AccWeight（20%恶意节点）')

plt.title('两种聚合方法在恶意节点影响下的模型性能对比（非均等划分）')
plt.xlabel('训练轮次')
plt.ylabel('测试集准确率')
plt.xticks(epochs)
plt.ylim(0.0, 0.75)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()


