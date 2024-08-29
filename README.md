# Subramanya5-Mobile-Sales-Data-Analysis-and-Visualization

#Case Study: Analysis of Sales Transactions for Predicting Pricing Strategies
Overview The objective of this case study is to analyze sales transactions data to understand factors influencing product pricing and sales performance. By exploring and modeling the data, we aim to identify trends, key metrics, and relationships that can inform pricing strategies, discount policies, and sales predictions. The insights gained will help businesses optimize their pricing, improve customer satisfaction, and enhance overall sales performance.

$Dataset Description

TransactionID: Unique identifier for each transaction.
TransactionDate: Date and time of the transaction.
CustomerID: Unique identifier for each customer.
ProductCategory: Category of the product sold.
Quantity: Number of units sold in the transaction.
PricePerUnit: Price of each unit sold.
TotalAmount: Total monetary amount for the transaction.
PaymentMethod: Method used for payment.
DiscountApplied: Discount applied to the transaction.
Notebook Explanation

#Data Preprocessing:

Handling Missing Values: Missing values were addressed by filling with logical defaults, such as substituting missing PricePerUnit values with the mean price within the same ProductCategory.
Consistency Checks: The TotalAmount was recalculated where necessary to ensure accuracy.
Feature Engineering: Additional features were created to better capture relationships between existing features, like recalculating TotalAmount from Quantity and PricePerUnit.
Exploratory Data Analysis (EDA):

#Sales Trends: 
Analyzed sales trends over time to identify patterns and seasonal effects.
Category-wise Analysis: Explored sales performance by product category, including the impact of discounts on sales volume.
Correlation Analysis: Investigated relationships between different numerical variables, such as PricePerUnit, Quantity, and TotalAmount.
Modeling and Evaluation:

#K-Nearest Neighbors (KNN) Regression:
Built a KNN model to predict PricePerUnit based on Quantity and DiscountApplied.

#Model Evaluation:
Assessed the model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R².
Visualization: Plotted actual vs predicted values to visualize model performance.

#Usage

Pricing Strategies: The analysis helps businesses understand how different factors like discounts and quantities sold affect pricing, aiding in the development of more effective pricing strategies.
Sales Predictions: The predictive model can be used to estimate future prices, allowing businesses to set competitive prices and maximize revenue.
Customer Behavior Insights: Insights into how customers respond to different price points and discounts can be used to refine promotional campaigns and improve customer engagement.
