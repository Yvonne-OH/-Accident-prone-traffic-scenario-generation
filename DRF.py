import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def DRF_strength(x, y, x0, y0, Vx, Vy, ax,ay, kappa, kappa1, kappa2, lambda_val, gamma_val, m_val):
    """
    Calculate the strength of an elliptical field at a given point (x, y).

    Parameters:
    - x (float): X-coordinate.
    - y (float): Y-coordinate.
    - x0 (float): X-coordinate of the Ego Vehicle center.
    - y0 (float): Y-coordinate of the Ego Vehicle center.
    - Vx (float): X-component of the velocity vector.
    - Vy (float): Y-component of the velocity vector.
    - a (float): Parameter defining the ellipse semi-minor axis.
    
    - kappa (float): Parameter influencing the field strength.
    - kappa1 (float): Exponent applied to the velocity term.
    - kappa2 (float): Parameter related to the ellipse eccentricity.
    - lambda_val (float): Parameter affecting the field strength.
    - gamma_val (float): Parameter influencing the field strength.
    - m_val (float): Parameter affecting the field strength.
    

    Returns:
    - strength (float): Strength of the DRF at the specified point.
    """
    
    rotation_angle=-np.arctan2(Vy, Vx)
    v = np.sqrt(Vx**2 + Vy**2)
    a = np.sqrt(ax**2 + ay**2)
    
    if a == 0:
        a = 0.0001
    
    # Rotate coordinates (x, y) by the specified angle
    x_rotated = (x - x0) * np.cos(rotation_angle) - (y - y0) * np.sin(rotation_angle)
    y_rotated = (x - x0) * np.sin(rotation_angle) + (y - y0) * np.cos(rotation_angle)
    
    theta = abs(abs(np.arctan2(y_rotated, x_rotated)))
   
    phi1 = (kappa * v)**kappa1
    phi2 = lambda_val * gamma_val * m_val
    
    term_x = (((x_rotated) - np.sqrt(1 - kappa2**2) * np.exp(phi1) * phi2) / ((-a / np.abs(a)) * phi2 * np.exp(phi1)))**2
    term_y = ((y_rotated) / (((np.cos(theta / 2)**np.abs(a / kappa2)) * kappa2 * phi2 * np.exp(phi1))))**2
    
    strength = np.sqrt(1 / (term_x + term_y))
    print(strength)
    # Clip the strength values to a specified range, replacing values outside the range with 'nan'
    #strength = np.where(strength <=0 , strength, np.nan)
    return np.clip(strength, None, 2.5)


file_path = 'neighbor.csv'  # 请替换为你的 CSV 文件路径
data = pd.read_csv(file_path)
data.columns = ['x', 'y','Vx','Vy','ax','ay']

#%%
# 创建势能场的坐标网格
x_min, x_max = data['x'].min(), data['x'].max()
y_min, y_max = data['y'].min(), data['y'].max()

# 将网格扩大10%
x_min, x_max = x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min)
y_min, y_max = y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)

# 创建扩大后的坐标网格
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)

kappa = 1
kappa1,kappa2 = 0.2, 0.4

lambda_val,gamma_val,m_val=2,0.5,2

for index, row in data.iterrows():
    
    x0=row['x']
    y0=row['y']
    Vx=row['Vx']*1
    Vy=row['Vy']*1
    ax=row['ax']
    ay=row['ay']
    if index==0:  
        field_strength = DRF_strength(X, Y, x0, y0, Vx,Vy, ax,ay, kappa,kappa1,kappa2,lambda_val,gamma_val,m_val)
        print(field_strength)
    else:
        field_strength += DRF_strength(X, Y, x0, y0, Vx,Vy, ax,ay, kappa,kappa1,kappa2,lambda_val,gamma_val,m_val)
   
        print(field_strength)
        
field_strength = np.where(field_strength >=1 , field_strength, np.nan)   
field_strength = np.clip(field_strength, None,5)
    
plt.contourf(X, Y, field_strength, levels=100, cmap='viridis')
    
plt.colorbar(label='Field Strength')
plt.title('Elliptical Field Strength')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().set_aspect('equal', adjustable='box')  # 保持横纵坐标比例一致
plt.grid(True)
plt.show()


