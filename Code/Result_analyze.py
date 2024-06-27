

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



His_folder_path = os.getcwd()+"\\Log\\"
# List all files in the directory
all_files = os.listdir(His_folder_path)
# Filter for files that end with '.csv'
csv_files = [file for file in all_files if file.endswith('.csv')]

# Now let's read each CSV file into a pandas DataFrame
csv_dataframes = {}
for csv_file in csv_files:
    file_path = os.path.join(His_folder_path, csv_file)
    data = pd.read_csv(file_path)
    # Calculate the moving average with a window size of N
    N = 25  # Window size, adjust based on your data
    data['Smoothed'] = data['Value'].rolling(window=N).mean()
    
    # Plot original and smoothed data
    plt.figure(figsize=(10, 6))
    #plt.plot(data['Step'], data['Value'], marker='o', label='Original')
    plt.plot(data['Step'], data['Smoothed'], marker='o', label='Smoothed')
    
    coefficients = np.polyfit(data['Step'], data['Value'], 2)
    trend_line = np.polyval(coefficients, data['Step'])
    plt.plot(data['Step'], trend_line, label='Trend Line', color='red')
    
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(csv_file.split("-")[-1].split(".")[0])
    plt.legend()
    plt.grid()
    plt.show()


def simple_moving_average_np(data, window_size=3):
    """Smooth data using a simple moving average with NumPy's convolve."""
    window = np.ones(int(window_size))/float(window_size)
    return np.array([np.convolve(v, window, 'same') for v in data])

window_size = 8  # Adjust based on your preference
folder_path = os.getcwd()+"\\Output\\"
# List all files in the directory
all_files = os.listdir(folder_path)
# Filter for files that end with '.csv'
npy_files = [file for file in all_files if file.endswith('.npy')]

Gen_data = [item for item in npy_files if "Gen" in item]
Tru_data = [item for item in npy_files if "Tru" in item]

Gen_v=[]
Gen_a=[]
Gen_omega=[]

for file in Gen_data:
    temp=np.load(folder_path+file)
    differences = np.diff(temp.squeeze(), axis=1)
    
    # Calculate velocity (magnitude) for each vehicle and each time step
    velocities =  simple_moving_average_np(np.linalg.norm(differences, axis=2)*10*3.6, window_size)
    accelerations = simple_moving_average_np(np.diff(velocities, axis=1)*10/3.6, window_size)
    angles = np.arctan2(differences[..., 1], differences[..., 0])
    angular_velocities = simple_moving_average_np(np.diff(np.unwrap(angles, axis=1), axis=1), window_size)
    
    Gen_v.append(velocities)
    Gen_a.append(accelerations)
    Gen_omega.append(angular_velocities)

Tru_v=[]
Tru_a=[]
Tru_omega=[]

for file in Tru_data:
    temp=np.load(folder_path+file).squeeze()
    differences = np.diff(temp, axis=0)
    velocities = simple_moving_average_np(np.linalg.norm(differences, axis=1)*10*3.6)
    accelerations = simple_moving_average_np(np.diff(velocities, axis=0)*100/3.6, window_size)
    angles = np.arctan2(differences[:, 1], differences[:, 0])

    angular_velocities = np.diff(angles, axis=0)
    
    Tru_v.append(velocities)
    Tru_a.append(accelerations)
    Tru_omega.append(angular_velocities)

# Prepare data for boxplot
data_for_boxplot = [np.array(Tru_v).flatten()]

# Plot
plt.figure(figsize=(5, 6))
plt.violinplot(data_for_boxplot, showmeans=False, showmedians=True)
plt.xticks([1], ['Velocities'])
plt.title('Violin Plot of Velocities, Accelerations, and Angular Velocities')
plt.ylabel('Value')
plt.grid(True)
plt.show()
# Prepare data for boxplot
data_for_boxplot = [np.array(Tru_omega).flatten(),np.array(Gen_omega).flatten()]

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(data_for_boxplot, labels=['Velocities','2'])
plt.title('Boxplot of Velocities, Accelerations, and Angular Velocities')
plt.ylabel('Value')
plt.grid(True)
plt.show()