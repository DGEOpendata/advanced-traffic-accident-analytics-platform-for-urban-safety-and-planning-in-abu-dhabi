python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Load the dataset
file_path = 'Traffic_Accident-v3.0_AbuDhabi_2023.csv'
data = pd.read_csv(file_path)

# Data Exploration
print(data.head())

# Data Cleaning
# Remove rows with missing values
data.dropna(inplace=True)

# Feature Engineering
# Convert accident date to datetime
data['report_date'] = pd.to_datetime(data['report_date'])
data['month'] = data['report_date'].dt.month

# Accident Frequency by Month
monthly_accidents = data.groupby('month').size()
plt.figure(figsize=(10, 6))
monthly_accidents.plot(kind='bar')
plt.title('Monthly Accident Frequency in Abu Dhabi (2023)')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.show()

# Correlation Between Weather and Accidents
sns.boxplot(x='weather', y='number_of_accidents', data=data)
plt.title('Impact of Weather on Accident Frequency')
plt.show()

# Predictive Modeling
# Prepare data for linear regression
X = data[['month']]
y = data['number_of_accidents']

model = LinearRegression()
model.fit(X, y)

# Predict for a specific month
month_to_predict = [[7]]
predicted_accidents = model.predict(month_to_predict)
print(f"Predicted number of accidents for month {month_to_predict[0][0]}: {predicted_accidents[0]}")
