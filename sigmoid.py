# Parameters for Sigmoid function
import numpy as np
from matplotlib import pyplot as plt

A = 100  # 最大奖励
k_values = [0.01, 0.05, 0.1, 0.5]  # 更小的增长速率以更明显地显示差异

# Generate x values (original reward before applying Sigmoid)
x = np.linspace(0, 200, 1000)  # 更广的范围，从0开始

plt.figure(figsize=(10, 6))

# Plot Sigmoid functions for different k values
for k in k_values:
    y = A / (1 + np.exp(-k * (x - 100)))  # 偏移以使中心位于特定值
    plt.plot(x, y, label=f'Sigmoid函数，k={k}')

plt.axhline(y=A, color='r', linestyle='--', label='最大奖励 A')
plt.xlabel('原始奖励 $R_{val,i}^{(t)}$')
plt.ylabel('平滑后奖励 $R\_Smooth_{val,i}^{(t)}$')
plt.title('不同增长速率 (k) 下的 Sigmoid 函数（最大奖励 A）')
plt.legend()
plt.grid(True)
plt.show()
