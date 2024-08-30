#Load and Explore the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'C:/Users/ss748/OneDrive/Documents/Desktop/Data Source (sales_transactions).csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df_head = df.head()
print(df_head)

# Display summary information about the dataframe
df_info = df.info()
print(df_info)

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)
#Data Preprocessing and Cleaning
# Handling missing values
df['CustomerID'].fillna(-1, inplace=True)
df['TotalAmount'] = df['TotalAmount'].fillna(df['Quantity'] * df['PricePerUnit'])
df['PricePerUnit'] = df.groupby('ProductCategory')['PricePerUnit'].transform(lambda x: x.fillna(x.mean()))
df['PaymentMethod'].fillna('Unknown', inplace=True)
df['DiscountApplied'].fillna(0, inplace=True)

# Recalculate 'TotalAmount' to ensure consistency
df['TotalAmount'] = df['Quantity'] * df['PricePerUnit']

# Display the cleaned dataframe and any remaining missing values
df_cleaned_head = df.head()
remaining_missing_values = df.isnull().sum()
print(df_cleaned_head)
print(remaining_missing_values)
#Data Aggregation and Statistical Analysis
# Aggregating by 'ProductCategory'
aggregated_df = df.groupby('ProductCategory').agg({
    'Quantity': 'sum',
    'PricePerUnit': 'mean',
    'TotalAmount': 'sum',
    'DiscountApplied': 'mean'
}).reset_index()

# Display the aggregated dataframe
print(aggregated_df)

# Calculate mean values for only numeric columns
mean_values = df.mean(numeric_only = True)
print(mean_values)
median_values = df.median(numeric_only = True)
print(median_values)
mode_values = df.mode(numeric_only = True)
print(mode_values)
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
#Split the Data into Train and Test Sets
from sklearn.model_selection import train_test_split

# Assuming 'TargetVariable' is your target for prediction (replace with the actual column name)
print(df.columns)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Data Cleaning
# Convert 'TransactionDate' to datetime format
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y %H:%M', errors='coerce')

# Handle missing values by assigning directly to the columns
df['CustomerID'] = df['CustomerID'].fillna(-1)
df['TotalAmount'] = df['TotalAmount'].fillna(df['Quantity'] * df['PricePerUnit'])
df['PricePerUnit'] = df.groupby('ProductCategory')['PricePerUnit'].transform(lambda x: x.fillna(x.mean()))
df['PaymentMethod'] = df['PaymentMethod'].fillna('Unknown')
df['DiscountApplied'] = df['DiscountApplied'].fillna(0)

# Recalculate 'TotalAmount' to ensure consistency
df['TotalAmount'] = df['Quantity'] * df['PricePerUnit']

# Example 1: Predicting 'PricePerUnit' based on 'Quantity' and 'DiscountApplied'
X = df[['Quantity', 'DiscountApplied']]
y = df['PricePerUnit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Build and Evaluate the K-Nearest Neighbors (KNN) Model
# Building a KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predicting
y_pred = knn.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"KNN Model (Predicting PricePerUnit): MSE = {mse}, R^2 = {r2}")

# Model Evaluation
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"KNN Model (Predicting TotalAmount):")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Accuracy-like Metric for Regression
# A higher R² indicates a better fit, with 1 being a perfect fit.
accuracy = r2 * 100
print(f"Model Accuracy (R² as a percentage): {accuracy:.2f}%")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
# Example: Using 'Quantity', 'PricePerUnit', and 'TotalAmount' for customer segmentation
features = df[['Quantity', 'PricePerUnit', 'TotalAmount']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Adding cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Analyzing the resulting clusters
# Check if 'Cluster' column exists
if 'Cluster' in df.columns:
    # Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Perform the groupby operation only on numeric columns
    cluster_analysis = df.groupby('Cluster')[numeric_cols].mean()

    # Display the results
    print(cluster_analysis)
else:
    print("The 'Cluster' column does not exist in the DataFrame.")

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PricePerUnit', y='TotalAmount', hue='Cluster', data=df, palette='viridis')
plt.title('K-Means Clustering: Price vs Total Amount')
plt.xlabel('Price Per Unit')
plt.ylabel('Total Amount')
plt.show()
# Visualize the Prediction
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Amount')
plt.ylabel('Predicted Total Amount')
plt.title('Actual vs Predicted Total Amount')
plt.grid(True)
plt.show()
#Data Visualization
# 1. Sales Trends Over Time
plt.figure(figsize=(10, 6))
df.groupby(df['TransactionDate'].dt.date)['TotalAmount'].sum().plot(kind='line')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.show()

# 2. Distribution of Product Categories
plt.figure(figsize=(10, 6))
sns.barplot(x='ProductCategory', y='TotalAmount', data=df, estimator=sum, ci=None)
plt.title('Total Sales Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.show()

# 3. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['PricePerUnit'], bins=20, kde=True)
plt.title('Price Per Unit Distribution')
plt.xlabel('Price Per Unit')
plt.ylabel('Frequency')
plt.show()

# 4. Relationship Between Price and Quantity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PricePerUnit', y='Quantity', data=df)
plt.title('Price vs Quantity Sold')
plt.xlabel('Price Per Unit')
plt.ylabel('Quantity Sold')
plt.show()

# 5. Discount vs Sales
plt.figure(figsize=(10, 6))
sns.boxplot(x='DiscountApplied', y='TotalAmount', data=df)
plt.title('Impact of Discount on Sales')
plt.xlabel('Discount Applied')
plt.ylabel('Total Sales Amount')
plt.show()
