import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (Ensure you have downloaded the CSV file)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names based on dataset description
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Read the dataset into a Pandas DataFrame
df = pd.read_csv(url, names=columns)

# Display basic info about the dataset
print(df.info())
print(df.describe())

# Create histograms for key medical features
features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

plt.figure(figsize=(12, 8))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

##############################################33

import numpy as np

# Identify columns where 0 is an unrealistic value
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Display original statistics
print("Original Statistics (with Zero Values):")
print(df[zero_columns].describe())

# Count zeros in each column
print("\nCount of Zero Values Before Handling Missing Data:")
print(df[zero_columns].eq(0).sum())

# Replace zeros with NaN
df_cleaned = df.copy()
df_cleaned[zero_columns] = df_cleaned[zero_columns].replace(0, np.nan)

# Display statistics after marking missing values
print("\nUpdated Statistics (After Replacing 0 with NaN):")
print(df_cleaned[zero_columns].describe())

# Count missing values after replacement
print("\nCount of Missing Values After Replacing Zeros with NaN:")
print(df_cleaned.isna().sum())

# Plot histograms before and after replacing 0 with NaN
plt.figure(figsize=(12, 8))

for i, feature in enumerate(zero_columns, 1):
    plt.subplot(3, 2, i)
    plt.hist(df[feature], bins=20, alpha=0.5, label="Before (With Zeros)", color="red")
    plt.hist(df_cleaned[feature], bins=20, alpha=0.5, label="After (NaN Replaced)", color="blue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()

##############################################3

# Perform groupby aggregation
grouped_stats = df_cleaned.groupby('Outcome')[zero_columns].agg(['mean', 'median', 'std'])

# Display the computed statistics
print("Summary Statistics by Diabetes Outcome:")
print(grouped_stats)

# Plot mean values for each group
# Extract mean values correctly

mean_values = grouped_stats.xs('mean', axis=1, level=1)

# Create a new figure
plt.figure(figsize=(12, 6))

# Plot the mean values as a bar chart
mean_values.T.plot(kind='bar', colormap="coolwarm", edgecolor='black')

# Add labels and title
plt.title("Comparison of Mean Medical Features by Diabetes Outcome")
plt.xlabel("Medical Features")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.legend(["No Diabetes (0)", "Diabetes (1)"])
plt.grid(axis='y')
plt.show()

####################################

import seaborn as sns

# Compute basic statistics
print("Basic Statistics:")
print(df_cleaned.describe())

# Compute correlation matrix
corr_matrix = df_cleaned.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.legend()

print("End of Correlation Statistics")


