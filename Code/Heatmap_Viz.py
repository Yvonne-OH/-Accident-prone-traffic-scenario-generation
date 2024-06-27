import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate some example data (replace this with your actual data)
data = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], size=1000)

# Fit a 2D Gaussian distribution to the data
mean, cov = np.mean(data, axis=0), np.cov(data, rowvar=False)
gaussian = multivariate_normal(mean=mean, cov=cov)

# Create a meshgrid for the heatmap
x, y = np.meshgrid(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100),
                   np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100))

# Evaluate the Gaussian PDF on the meshgrid
pdf_values = gaussian.pdf(np.column_stack((x.flatten(), y.flatten())))

# Reshape PDF values to match the shape of the meshgrid
pdf_values = pdf_values.reshape(x.shape)

# Set values below a threshold to NaN
threshold = 0.01
pdf_values[pdf_values < threshold] = np.nan

# Visualize the original data
#plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')

# Plot the heatmap of the fitted Gaussian distribution with the 'plasma' colormap
heatmap = plt.imshow(pdf_values, extent=[np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])],
           origin='lower', cmap='YlGnBu', alpha=0.85, aspect='auto')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap of Fitted Gaussian Distribution')
plt.legend()

# Add colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label('Probability Density')

# Show the plot
plt.show()
