import pandas as pd
data = pd.read_csv('D7_Student_Performance.csv')
print(data.head())


missing_values = data.isnull().sum()

print(missing_values)

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data)

# Display the first few rows of the encoded dataset
print(data_encoded.head())


print(data_encoded.columns)

from sklearn.model_selection import train_test_split

# Select features and target variable
features = ['math score', 'reading score', 'writing score',
            'gender_female', 'gender_male',
            'race/ethnicity_group A', 'race/ethnicity_group B',
            'race/ethnicity_group C', 'race/ethnicity_group D', 'race/ethnicity_group E',
            'parental level of education_associate\'s degree',
            'parental level of education_bachelor\'s degree',
            'parental level of education_high school',
            'parental level of education_master\'s degree',
            'parental level of education_some college',
            'parental level of education_some high school']
target = ['lunch_free/reduced']

X = data_encoded[features]
y = data_encoded[target]

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets to verify the split
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

#Decison Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Predict lunch type for the testing data
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print()
# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
print()
# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)
print()
# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
print()
# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", roc_auc)
print()
# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


#Model 2: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy (Random Forest):", accuracy_rf)

# Predict lunch type for the testing data
y_pred_rf = rf_classifier.predict(X_test)

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(conf_matrix_rf)

# Precision
precision_rf = precision_score(y_test, y_pred_rf)
print("Precision:", precision_rf)

# Recall
recall_rf = recall_score(y_test, y_pred_rf)
print("Recall:", recall_rf)

# F1 Score
f1_rf = f1_score(y_test, y_pred_rf)
print("F1 Score:", f1_rf)

# ROC AUC Score
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
print("ROC AUC Score:", roc_auc_rf)

# Classification Report
class_report_rf = classification_report(y_test, y_pred_rf)
print("Classification Report:")
print(class_report_rf)
