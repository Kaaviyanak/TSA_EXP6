# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18-05-2026
```
NAME: KAAVIYAN K
REG NO: 212224240066
```

### AIM:
To implement the Holt Winters Method Model using Python.


### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/content/Walmart_Sales.csv', parse_dates=['Date'], index_col='Date', dayfirst=True)
# Select 'Weekly_Sales' and rename it to 'Monthly_Sales' for clarity
# Then resample monthly sales data
data_monthly = data[['Weekly_Sales']].resample('MS').sum()
data_monthly.rename(columns={'Weekly_Sales': 'Monthly_Sales'}, inplace=True)
print(data_monthly.head())

# Plot monthly sales
data_monthly.plot(y='Monthly_Sales')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.title('Monthly Sales Data')
plt.show()

# Scale data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly[['Monthly_Sales']]).flatten(),
    index=data_monthly.index
)

# Check seasonality
seasonal_decompose(data_monthly['Monthly_Sales'], model='additive').plot()
plt.show()

# Train and test split
scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

# Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Predictions
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot evaluation
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(['Train Data', 'Predictions', 'Test Data'])
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.title('Visual Evaluation')
plt.show()

# RMSE
print(np.sqrt(mean_squared_error(test_data, test_predictions_add)))

# Final model
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Future prediction
final_predictions = final_model.forecast(steps=12)

# Plot future forecast
ax = data_monthly.plot(y='Monthly_Sales')
final_predictions.plot(ax=ax)

ax.legend(['Monthly Sales', 'Future Predictions'])
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.title('Prediction')
plt.show()
```
### OUTPUT:
SCALED DATA PLOT

<img width="1112" height="741" alt="image" src="https://github.com/user-attachments/assets/611b3c00-e301-44ac-83ad-10cc74cf10bd" />

DECOMPOSED PLOT
<img width="1128" height="583" alt="image" src="https://github.com/user-attachments/assets/6927e1f3-289e-4013-b175-edf063a47c92" />


TEST_PREDICTION

<img width="1121" height="575" alt="image" src="https://github.com/user-attachments/assets/f199c6dc-70bb-41c9-9fe4-36256b825125" />


FINAL_PREDICTION

<img width="1025" height="611" alt="image" src="https://github.com/user-attachments/assets/6a014c3d-3f21-4a1c-882c-9911f5b352e9" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
