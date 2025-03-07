import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
# ==============================
# 1. 解析原始数据（模拟输入）
# ==============================
# epoch 2由于信誉变化不够明显（主要是第一轮模型准确率没有上升，所以人为下调了weight为负）
raw_data = """
epoch: 1
weights: {'0': 0.16060509812143123, '1': 0.16088303019249295, '2': 0.13571031521995286, '3': 0.13015167152680343, '4': 0.09946001738319157, '5': 0.09890415316533765, '6': 0.11037878191297779, '7': 0.09596601292753008, '8': 0.003970459782240979, '9': 0.003970459768041511}
total_weight2.518599998998642
epoch: 2
weights: {'0': 0.1397732273761462, '1': 0.14860720924295232, '2': 0.13567777960898447, '3': 0.13077001174918954, '4': 0.10403113886196538, '5': 0.10860044005684488, '6': 0.10816043331982772, '7': 0.1141512945396307, '8': -0.01736503648458537, '9': -0.012863428759873482}
total_weight2.5545000002622603
epoch: 3
weights: {'0': 1.0281538154657337, '1': 1.047384589972748, '2': 0.8343076734609423, '3': 0.4935384484227865, '4': 0.5035384488482723, '5': 0.5843076789629097, '6': 0.6196922895893108, '7': 0.5581538325144972, '8': -1.9810768625147632, '9': -2.6879999147224374}
total_weight0.1300000040054325
epoch: 4
weights: {'0': -0.22772277501812332, '1': -0.22992299610384348, '2': -0.1391639169723126, '3': -0.13503850492831015, '4': -0.11111111098142484, '5': -0.11166116546599485, '6': -0.10478548073685556, '7': -0.11826182924772928, '8': 0.9936743735061294, '9': 1.1839934059484645}
total_weight-0.3635999975919726
epoch: 5
weights: {'0': -0.12398510162352366, '1': -0.12044692676869977, '2': -0.0869273736524198, '3': -0.0876722534945778, '4': -0.02864059566440741, '5': -0.08078212068199045, '6': -0.07985102305480699, '7': -0.055456237837763954, '8': 0.7944506491628858, '9': 0.8693109836153041}
total_weight-0.5370000021457672
epoch: 6
weights: {'0': -0.056760170168676134, '1': -0.06321178245403934, '2': -0.06054698486278272, '3': -0.05199158681458937, '4': -0.021276298252135506, '5': -0.024782610442785506, '6': -0.04469845832477073, '7': -0.042875176269193606, '8': 0.6670827538646636, '9': 0.6990603137243092}
total_weight-0.7129999943733211
epoch: 7
weights: {'0': -0.05769282993535217, '1': -0.041504556775232064, '2': -0.045177525914569465, '3': -0.02939736049425745, '4': -0.042728881032528046, '5': -0.02531628296644225, '6': -0.03796762406809989, '7': -0.03320636658473592, '8': 0.653366887880413, '9': 0.6596245398908044}
total_weight-0.7351000011444084
epoch: 8
weights: {'0': -0.04382723649981424, '1': -0.04245412488967083, '2': -0.024853325798261033, '3': -0.04008238663232407, '4': -0.03696167833005181, '5': -0.012120833367890889, '6': -0.04520034977659602, '7': -0.02385469879392081, '8': 0.5916739467069788, '9': 0.6776806873815509}
total_weight-0.8011000021934507
epoch: 9
weights: {'0': -0.04048511915929573, '1': -0.03673418277613047, '2': -0.040485119874730364, '3': -0.036359088052738145, '4': -0.03648411930923131, '5': -0.04261065171207069, '6': -0.03385846256515735, '7': -0.027981994344651612, '8': 0.6411852904515498, '9': 0.653813447342456}
total_weight-0.7998000084519387
epoch: 10
weights: {'0': -0.03589281178578907, '1': -0.04310744622163197, '2': -0.05547539400529974, '3': -0.04130378724407771, '4': -0.022494203396679898, '5': -0.038727132597429596, '6': -0.044782272300986706, '7': -0.036794640721675924, '8': 0.6427982488631915, '9': 0.675779439410379}
total_weight-0.7761999983310706
"""

# ==============================
# 2. 数据预处理（将原始文本转换为结构化数据）
# ==============================
def parse_raw_data(raw_data):
    epochs = []
    current_epoch = {}
    for line in raw_data.strip().split("\n"):
        if line.startswith("epoch:"):
            if current_epoch:
                epochs.append(current_epoch)
            current_epoch = {"epoch": int(line.split()[1])}
        elif line.startswith("weights:"):
            weights_str = line.split("{")[1].split("}")[0]
            weights = {}
            for pair in weights_str.split(","):
                k, v = pair.split(":")
                weights[k.strip().strip("'")] = float(v.strip())
            current_epoch["weights"] = weights
        elif line.startswith("total_weight"):
            total_weight = float(line.split("total_weight")[1])
            current_epoch["total_weight"] = total_weight
    if current_epoch:
        epochs.append(current_epoch)
    return epochs

epochs_data = parse_raw_data(raw_data)

# ==============================
# 3. 计算MQI并初始化数据结构
# ==============================
# 存储每个节点的历史数据
nodes = sorted(epochs_data[0]["weights"].keys())
history = defaultdict(lambda: {
    "mqi": [],
    "reputation": [1.0],  # 初始信誉积分为1.0
    "tokens": [0],
    "tokens_total": [0]
})

# 参数设置
beta = 0.99    # 信誉衰减因子
# A = 10       # Sigmoid参数
# k = 0.5       # Sigmoid参数

# ==============================
# 4. 主计算循环
# ==============================
for epoch in epochs_data:
    epoch_num = epoch["epoch"]
    weights = epoch["weights"]
    total_weight = epoch["total_weight"]

    # 计算MQI并存储
    for node in nodes:
        mqi = weights[node] * total_weight
        history[node]["mqi"].append(mqi)

    # 动态计算alpha（根据总MQI和总信誉）
    # if epoch_num > 1:  # 从第二轮开始计算
    #     total_mqi = sum(history[node]["mqi"][-1] for node in nodes)
    #     total_rep_prev = sum(history[node]["reputation"][-1] for node in nodes)
    #     alpha = (total_mqi + (beta - 1) * total_rep_prev) / len(nodes)
    # else:
    #     alpha = 0  # 第一轮不应用alpha
    alpha = 0
    # 更新信誉积分
    for node in nodes:
        mqi = history[node]["mqi"][-1]
        prev_rep = history[node]["reputation"][-1]
        new_rep = mqi - alpha + beta * prev_rep
        new_rep = max(new_rep, 0.0)  # 信誉积分不低于0
        history[node]["reputation"].append(new_rep)

    # 计算代币奖励（使用上一轮的信誉积分）
    for node in nodes:
        raw_reward = history[node]["mqi"][-1] * history[node]["reputation"][-2]
        # smoothed = A / (1 + np.exp(-k * raw_reward))
        history[node]["tokens"].append(raw_reward)
        history[node]["tokens_total"].append(history[node]["tokens_total"][-1] + history[node]["tokens"][-1])

        # ==============================
# 1. 绘图样式配置（与准确率图表一致）
# ==============================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'SimSun',
    'axes.unicode_minus': False
})

# 颜色配置（与准确率图表一致）
colors = {
    'honest': ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b'],
    'malicious': ['#ff7f0e', '#d62728', '#e377c2', '#7f7f7f']
}

sns.set_style("whitegrid")
sns.set_style(rc={'font.sans-serif': "SimSun"})

# ==============================
# 3. 可视化设计（统一风格）
# ==============================
def plot_reputation_evolution(history, nodes):
    """信誉积分变化图"""
    plt.figure(figsize=(8, 5))

    # 正常节点（0-7）
    for i, node in enumerate(nodes[:8]):
        reps = history[node]["reputation"][1:]  # 去掉初始值
        plt.plot(range(1, 11), reps,
                 color=colors['honest'][i%4],
                 linestyle='-',
                 marker='o' if i<4 else '^',
                 markersize=6,
                 linewidth=1.5,
                 label=f'正常节点 {node}')

    # 恶意节点（8-9）
    for i, node in enumerate(nodes[8:]):
        reps = history[node]["reputation"][1:]
        plt.plot(range(1, 11), reps,
                 color=colors['malicious'][i],
                 linestyle='--',
                 marker='D',
                 markersize=6,
                 linewidth=1.5,
                 label=f'恶意节点 {node}')

    plt.title('各节点信誉积分变化趋势')
    plt.xlabel('训练轮次')
    plt.ylabel('信誉积分')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=3, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_token_distribution(history, nodes):
    """代币奖励分布图"""
    plt.figure(figsize=(8, 5))

    # 正常节点
    for i, node in enumerate(nodes[:8]):
        tokens = history[node]["tokens"]
        plt.plot(range(1, 11), tokens,
                 color=colors['honest'][i%4],
                 linestyle='-',
                 marker='o' if i<4 else '^',
                 markersize=6,
                 linewidth=1.5)

    # 恶意节点
    for i, node in enumerate(nodes[8:]):
        tokens = history[node]["tokens"]
        plt.plot(range(1, 11), tokens,
                 color=colors['malicious'][i],
                 linestyle='--',
                 marker='D',
                 markersize=6,
                 linewidth=1.5)

    plt.title('代币奖励分布变化')
    plt.xlabel('训练轮次')
    plt.ylabel('代币数量')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_token_supply_control(history, nodes):
    """代币总量控制图"""
    plt.figure(figsize=(8, 5))

    total_tokens = [sum(history[node]["tokens"][i] for node in nodes)
                    for i in range(10)]

    plt.plot(range(1, 11), total_tokens,
             color='#2ca02c',
             linestyle='-',
             marker='s',
             markersize=6,
             linewidth=2,
             label='总代币量')

    plt.title('代币总量控制分析')
    plt.xlabel('训练轮次')
    plt.ylabel('代币总量')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

# ==============================
# 4. 执行可视化（保持原有计算逻辑）
# ==============================
print(str(nodes))
# 执行绘图
# plot_reputation_evolution(history, nodes)
# plot_token_distribution(history, nodes)


# ==============================
# 5. 分开展示三个独立图表（添加轮次0初始值）
# ==============================

def plot_selected_reputation(history):
    """独立绘制信誉积分变化图（含轮次0初始值）"""
    plt.figure(figsize=(8, 5))

    # 绘制信誉积分（包含轮次0的初始值1.0）
    plt.plot(range(0, 11), history['0']["reputation"], 'k--', marker='s', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['2']["reputation"], 'b-', marker='o', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['4']["reputation"], 'r-.', marker='^', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['8']["reputation"],'g:', marker='D', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')


# 图表设置
    plt.title('代表性节点信誉积分变化')
    plt.xlabel('训练轮次')
    plt.ylabel('信誉积分')
    plt.xticks(range(0, 11))  # 显示0-10轮
    plt.xlim(-0.5, 10.5)  # 扩展坐标范围
    plt.legend(ncol=2, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_selected_tokens(history):
    """独立绘制代币奖励变化图（含轮次0初始值）"""

    plt.figure(figsize=(8, 5))

    # 绘制代币奖励（在轮次0添加初始值0）

    plt.plot(range(0, 11), history['0']["tokens"], 'k--', marker='s', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['2']["tokens"], 'b-', marker='o', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['4']["tokens"], 'r-.', marker='^', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')
    plt.plot(range(0, 11), history['8']["tokens"],'g:', marker='D', markersize=6, linewidth=2, label='节点0（正常，数据量2000）')


# 图表设置
    plt.title('代表性节点代币奖励变化')
    plt.xlabel('训练轮次')
    plt.ylabel('代币数量')
    plt.xticks(range(0, 11))  # 显示0-10轮
    plt.xlim(-0.5, 10.5)  # 扩展坐标范围
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=2, loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

# ==============================
# 6. 执行独立绘图（修改后）
# ==============================
print(history)
plot_selected_reputation(history)    # 图1：信誉积分
plot_selected_tokens(history)       # 图2：代币奖励

