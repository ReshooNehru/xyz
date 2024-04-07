import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/ADS/d7_StudentsPerformance_cleaned.csv")  # Replace "your_dataset.csv" with the path to your dataset file

from collections import Counter

# Categorical features to check for imbalance
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Calculate the frequency of each category in each feature
for feature in categorical_features:
    feature_counts = Counter(data[feature])
    print(f"Feature: {feature}")
    print(feature_counts)
    # Check for imbalance (if any category has significantly higher frequency than others)
    total_samples = len(data)
    for category, count in feature_counts.items():
        percentage = count / total_samples * 100
        print(f"{category}: {percentage:.2f}%")
    print()


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Apply label encoding to categorical columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Apply SMOTE to imbalance columns only
smote = SMOTE(random_state=42)
imbalance_columns = ['parental level of education']
counter_report_before_smote = {}
for column in imbalance_columns:
    counter_report_before_smote[column] = Counter(data[column])

# Print counter report before SMOTE
print("Counter report before SMOTE:")
for column, counter in counter_report_before_smote.items():
    print(f"Feature: {column}")
    print(counter)
    total = sum(counter.values())
    for category, count in counter.items():
        print(f"{category}: {count / total * 100:.2f}%")

# Apply SMOTE to imbalance columns
for column in imbalance_columns:
    X = data.drop(column, axis=1)
    y = data[column]
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Update the original dataset with the balanced data
    data_balanced = pd.concat([X_resampled, pd.DataFrame({column: y_resampled})], axis=1)
    data = data_balanced

# Counter report after SMOTE
counter_report_after_smote = {}
for column in imbalance_columns:
    counter_report_after_smote[column] = Counter(data[column])

# Print counter report after SMOTE
print("\nCounter report after SMOTE:")
for column, counter in counter_report_after_smote.items():
    print(f"Feature: {column}")
    print(counter)
    total = sum(counter.values())
    for category, count in counter.items():
        print(f"{category}: {count / total * 100:.2f}%")

