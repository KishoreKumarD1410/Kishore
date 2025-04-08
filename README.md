#Wiseanalytics Sales Forecasting Assessment
import pandas as pd
import numpy as np


train = pd.read_csv("/content/train.csv", parse_dates=['date'])
test = pd.read_csv("/content/test.csv", parse_dates=['date'])
stores = pd.read_csv("/content/stores.csv")
oil = pd.read_csv("/content/oil.csv", parse_dates=['date'])
holidays = pd.read_csv("/content/holidays_events.csv", parse_dates=['date'])

oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear')

train_merged = train.merge(stores, on='store_nbr', how='left')

train_merged = train_merged.merge(oil, on='date', how='left')

#(merge with duplicates handled)
holidays['is_holiday'] = 1
holidays_simple = holidays[['date', 'is_holiday']].drop_duplicates()
train_merged = train_merged.merge(holidays_simple, on='date', how='left')
train_merged['is_holiday'] = train_merged['is_holiday'].fillna(0)

train_merged['earthquake'] = (train_merged['date'] == '2016-04-16').astype(int)

train_merged['day'] = train_merged['date'].dt.day
train_merged['is_payday'] = train_merged['day'].isin([15, 31]).astype(int)

train_merged['year'] = train_merged['date'].dt.year
train_merged['month'] = train_merged['date'].dt.month
train_merged['week'] = train_merged['date'].dt.isocalendar().week
train_merged['weekday'] = train_merged['date'].dt.weekday

train_merged.sort_values(by=['store_nbr', 'family', 'date'], inplace=True)

train_merged['sales_lag_1'] = train_merged.groupby(['store_nbr', 'family'])['sales'].shift(1)
train_merged['sales_lag_7'] = train_merged.groupby(['store_nbr', 'family'])['sales'].shift(7)
train_merged['rolling_mean_7'] = train_merged.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).mean()
train_merged['rolling_std_7'] = train_merged.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).std()

train_final = train_merged.dropna()

train_final.head()
import matplotlib.pyplot as plt
import seaborn as sns

daily_sales = train_final.groupby('date')['sales'].sum().reset_index()

# Plot overall sales trend
plt.figure(figsize=(14, 5))
sns.lineplot(data=daily_sales, x='date', y='sales')
plt.title("Total Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

#plot holidays vs non holidays
holiday_sales = train_final.groupby(['is_holiday'])['sales'].mean().reset_index()

sns.barplot(data=holiday_sales, x='is_holiday', y='sales')
plt.title("Average Sales: Holidays vs Non-Holidays")
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.ylabel("Average Sales")
plt.show()

#ovil vs price
sns.scatterplot(data=train_final, x='dcoilwtico', y='sales', alpha=0.3)
plt.title("Oil Price vs Sales")
plt.xlabel("Oil Price (WTI)")
plt.ylabel("Sales")
plt.show()

#plot by avg monthly sale

monthly_sales = train_final.groupby('month')['sales'].mean().reset_index()

sns.barplot(data=monthly_sales, x='month', y='sales', palette='viridis')
plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.show()

#cleaned data set
train_final.to_csv("cleaned_sales_data.csv", index=False)

#using XGBoost for forecasting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

features = [
    'store_nbr', 'family', 'onpromotion', 'cluster', 'dcoilwtico',
    'is_holiday', 'earthquake', 'is_payday',
    'year', 'month', 'week', 'weekday',
    'sales_lag_1', 'sales_lag_7', 'rolling_mean_7', 'rolling_std_7'
]

train_final['family'] = train_final['family'].astype('category').cat.codes

model_data = train_final.dropna(subset=features + ['sales'])

X = model_data[features]
y = model_data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

rmse, mape, r2

Day 2: Forecasting Model with XGBoost


features = [
    'store_nbr', 'family', 'onpromotion', 'cluster', 'dcoilwtico',
    'is_holiday', 'earthquake', 'is_payday',
    'year', 'month', 'week', 'weekday',
    'sales_lag_1', 'sales_lag_7', 'rolling_mean_7', 'rolling_std_7'
]
train_final['family'] = train_final['family'].astype('category').cat.codes

model_data = train_final.dropna(subset=features + ['sales'])

X = model_data[features]
y = model_data['sales']
#train and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Fit
xgb_model.fit(X_train, y_train)

#prediction value match
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Predict
y_pred = xgb_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 RMSE: {rmse:.2f}")
print(f"📉 MAPE: {mape:.2%}")
print(f"📈 R-squared: {r2:.4f}")

#plot predict vs actual
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title("Predicted vs Actual Sales (Sample)")
plt.xlabel("Sample")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

y_pred_naive = X_test['sales_lag_1']
rmse_naive = mean_squared_error(y_test, y_pred_naive)
mape_naive = mean_absolute_percentage_error(y_test, y_pred_naive)
r2_naive = r2_score(y_test, y_pred_naive)

print(f"Naive Model → RMSE: {rmse_naive:.2f}, MAPE: {mape_naive:.2%}, R²: {r2_naive:.4f}")

from statsmodels.tsa.arima.model import ARIMA

daily_sales = train_final.groupby('date')['sales'].sum()

arima_model = ARIMA(daily_sales, order=(5, 1, 0))
arima_result = arima_model.fit()

forecast_arima = arima_result.forecast(steps=15)

forecast_arima.plot(title='ARIMA Forecast: Next 15 Days')
plt.show()

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest →",
      f"RMSE: {mean_squared_error(y_test, y_pred_rf):.2f}, ",
      f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_rf):.2%}, ",
      f"R²: {r2_score(y_test, y_pred_rf):.4f}")
importances = rf_model.feature_importances_
feature_names = X_train.columns

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
plt.title("Feature Importance - Random Forest")
plt.show()


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(y.values.reshape(-1, 1))

def create_sequences(data, time_steps=10):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i+time_steps])
        ys.append(data[i+time_steps])
    return np.array(Xs), np.array(ys)

X_lstm, y_lstm = create_sequences(scaled_sales, 15)
X_train_lstm, X_test_lstm = X_lstm[:-300], X_lstm[-300:]
y_train_lstm, y_test_lstm = y_lstm[:-300], y_lstm[-300:]

# LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32)

# Predict
y_pred_lstm = model.predict(X_test_lstm)
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)

# Evaluate
print("LSTM →",
      f"RMSE: {mean_squared_error(y_test[-300:], y_pred_lstm_inv):.2f}")

