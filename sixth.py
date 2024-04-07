import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv("d7_StudentsPerformance_cleaned.csv")

# Calculate the average score as the common attribute
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# Selecting numerical features for outlier detection
numerical_features = ['math score', 'reading score', 'writing score', 'average_score']
X = data[numerical_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Outlier detection using DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=20) # Adjust parameters as needed
outlier_label_dbscan = dbscan.fit_predict(X_scaled)

# Define colors for density-based points
colors = {-1: 'red', 0: 'blue', 1: 'green'}

# Create custom colormap for legend
legend_cmap = ListedColormap(['red', 'blue', 'green'])

# Visualize the DBSCAN clustering
plt.figure(figsize=(12, 6))

# Plot points
scatter = plt.scatter(data['average_score'], data['math score'], c=outlier_label_dbscan, cmap=legend_cmap)
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outliers'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Core Points'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Border Points')]
plt.legend(handles=legend_elements, title="Density")
plt.annotate("Clusters detected by DBSCAN with circles representing cluster boundary.",
             xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='center',
             bbox=dict(boxstyle="round", fc="w"))

# Plot circles around clusters
unique_labels = np.unique(outlier_label_dbscan)
for label in unique_labels:
    if label != -1:
        cluster_points = X[outlier_label_dbscan == label]
        circle_radius = 4  # Adjust the circle radius as needed
        circle = plt.Circle((cluster_points['average_score'].mean(), cluster_points['math score'].mean()),
                            circle_radius, color='black', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        plt.annotate('Cluster {}'.format(label),
                     (cluster_points['average_score'].mean(), cluster_points['math score'].mean()),
                     textcoords="offset points", xytext=(-15,10), ha='center', fontsize=8)

# Add annotations for density-based points in a box
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", linewidth=1, facecolor="white")
plt.annotate("Outliers: Points with low density\nCore Points: Densely packed points\nBorder Points: Points near dense regions",
             xy=(0.99, 0.01), xycoords='axes fraction', fontsize=10, ha="right", va="bottom", bbox=bbox_props)

plt.title('DBSCAN Outlier Detection')
plt.xlabel('Average Score')
plt.ylabel('Math Score')
plt.show()

# Print the details of detected clusters
cluster_details = pd.DataFrame(columns=['Cluster', 'Number of Points'])
for label in unique_labels:
    if label != -1:
        num_points = len(X[outlier_label_dbscan == label])
        cluster_details = pd.concat([cluster_details, pd.DataFrame({'Cluster': [label], 'Number of Points': [num_points]})])

print("Details of Detected Clusters:")
print(cluster_details)
