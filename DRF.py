
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'neighbor.csv'  # 请替换为你的 CSV 文件路径
data = pd.read_csv(file_path)
data.columns = ['x', 'y','Vx','Vy','Ax','Ay']


plt.figure(figsize=(10, 6))

plt.scatter(data['x'], data['y'], label='Data')



plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.legend()

plt.show()
#%%
# 创建势能场的坐标网格
x_min, x_max = data['x'].min(), data['x'].max()
y_min, y_max = data['y'].min(), data['y'].max()

# 将网格扩大10%
x_min, x_max = x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)
y_min, y_max = y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)

# 创建扩大后的坐标网格
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)

# 定义简单的高斯势能场函数
def gaussian_potential(x, y, center_x, center_y, sigma):
    return np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

# 计算势能场的值，假设每个数据点都是一个障碍物
sigma = 1.0  # 调整高斯函数的标准差
Z = np.zeros_like(X)

for index, row in data.iterrows():
    Z += gaussian_potential(X, Y, row['x'], row['y'], sigma)

# 绘制等高线图
plt.contour(X, Y, Z, levels=20, cmap='viridis')

# 添加标题和标签
plt.title('Potential Field with Obstacles')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# 显示图表
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# 定义椭圆场强函数
def ellipse_field_strength(x, y, x0, y0, Vx, Vy, a, kappa, kappa1, kappa2, lambda_val, gamma_val, m_val, rotation_angle):
    if a == 0:
        a = 0.0001
    
    v = np.sqrt(Vx**2 + Vy**2)
    
    # Rotate coordinates (x, y) by the specified angle
    x_rotated = (x - x0) * np.cos(rotation_angle) - (y - y0) * np.sin(rotation_angle)
    y_rotated = (x - x0) * np.sin(rotation_angle) + (y - y0) * np.cos(rotation_angle)
    
    theta = abs(abs(np.arctan2(y_rotated, x_rotated)))
    print(theta)
    phi1 = (kappa * v)**kappa1
    phi2 = lambda_val * gamma_val * m_val
    
    term_x = (((x_rotated) - np.sqrt(1 - kappa2**2) * np.exp(phi1) * phi2) / ((-a / np.abs(a)) * phi2 * np.exp(phi1)))**2
    term_y = ((y_rotated) / (((np.cos(theta / 2)**np.abs(a / kappa2)) * kappa2 * phi2 * np.exp(phi1))))**2
    
    strength = np.sqrt(1 / (term_x + term_y))
    strength = np.where((term_x + term_y) <= 1, strength, np.nan)
    return np.clip(strength, None, 2.5)

# 设置参数
x0, y0 = 0.0, 0.0
Vx,Vy= 20,20
a = 3.0  # 请根据实际情况调整

kappa = 1
kappa1,kappa2 = 0.2, 0.4

lambda_val,gamma_val,m_val=2,0.5,2

rotation_angle = np.pi / 4

# 生成坐标网格
x = np.linspace(-30, 30, 500)
y = np.linspace(-30, 30, 500)
X, Y = np.meshgrid(x, y)

# 计算椭圆场强度值
field_strength = ellipse_field_strength(X, Y, x0, y0, Vx,Vy, a, kappa,kappa1,kappa2,lambda_val,gamma_val,m_val,rotation_angle)

# 绘制场强度图
plt.contourf(X, Y, field_strength, levels=100, cmap='viridis')
plt.colorbar(label='Field Strength')
plt.title('Elliptical Field Strength')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().set_aspect('equal', adjustable='box')  # 保持横纵坐标比例一致
plt.grid(True)
plt.show()


