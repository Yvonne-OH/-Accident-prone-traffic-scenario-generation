

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



