import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('seattle-weather.csv')

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Histogram of precipitation
plt.figure(figsize=(10, 6))
sns.histplot(df['precipitation'], bins=30, kde=True)
plt.title('Distribution of Precipitation')
plt.xlabel('Precipitation')
plt.ylabel('Frequency')
plt.show()

# Boxplot of temperature
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['temp_max', 'temp_min']])
plt.title('Temperature Distribution')
plt.ylabel('Temperature (°C)')
plt.show()

# Bar plot of weather types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='weather')
plt.title('Weather Types')
plt.xlabel('Weather')
plt.ylabel('Frequency')
plt.show()

# Inferential Analysis

# Correlation matrix (excluding non-numeric columns)
correlation_matrix = df.drop(columns=['date', 'weather']).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Hypothesis testing - t-test for mean temperature difference
from scipy.stats import ttest_ind

# Example: Test if there's a significant difference in mean temperature between rainy and non-rainy days
rainy_temps = df[df['weather'] == 'rain']['temp_max']
non_rainy_temps = df[df['weather'] != 'rain']['temp_max']

t_stat, p_value = ttest_ind(rainy_temps, non_rainy_temps)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference in mean temperature between rainy and non-rainy days.")
else:
    print("Fail to reject null hypothesis: There is no significant difference in mean temperature between rainy and non-rainy days.")
