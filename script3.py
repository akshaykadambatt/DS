import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Simulate a dataset with missing, invalid, and outlier values
data = {
    'timestamp': pd.date_range(start='2024-01-01 00:00', periods=20, freq='H'),
    'temperature': [21, 22, np.nan, 20, 30, 45, 21, 23, -10, 22, 19, np.nan, 100, 20, 21, 22, 23, 25, 21, 22],
    'humidity': [50, 52, 55, np.nan, 60, 65, -5, 70, 75, 80, 85, 90, np.nan, 100, 50, 60, 70, 80, np.nan, 90],
    'CO2': [400, 450, 500, 600, 800, 3000, 900, 1000, 1100, 1500, 2000, 1800, 1600, 1400, 1000, 900, 800, 600, 500, 400],
    'fan_speed': [1000, 1050, 1100, np.nan, 1200, 1300, 1400, np.nan, 1500, 1600, 1700, 1750, 1800, 1850, np.nan, 1950, 2000, 2050, 2100, 2150],
    'power_consumption': [5, 5.5, 6, -3, 7.5, 8, 9, 9.5, 10, 11, 12, 11.5, 13, 14, -2, 15, 16, 17, 18, 19]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display raw data
print("Raw Data:")
print(df)

# ----- Data Cleaning -----
print("\nStarting Data Cleaning...")

# Handling Missing Values
df['temperature'].fillna(df['temperature'].mean(), inplace=True)
df['humidity'].interpolate(method='linear', inplace=True)
df['fan_speed'].fillna(method='ffill', inplace=True)  # Forward fill for fan speed

# Handle Invalid Data
df['humidity'] = df['humidity'].clip(lower=0, upper=100)  # Valid range for humidity is 0-100%
df['temperature'] = df['temperature'].clip(lower=0, upper=50)  # Cap temperature between 0 and 50 °C
df['power_consumption'] = df['power_consumption'].clip(lower=0)  # Power consumption can't be negative

# Remove Outliers using IQR for CO2 levels
Q1 = df['CO2'].quantile(0.25)
Q3 = df['CO2'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['CO2'] = np.where((df['CO2'] < lower_bound) | (df['CO2'] > upper_bound), df['CO2'].median(), df['CO2'])

# Display cleaned data
print("\nCleaned Data:")
print(df)

# ----- Feature Engineering -----
print("\nStarting Feature Engineering...")

# Add rolling mean for temperature
df['temp_rolling_mean'] = df['temperature'].rolling(window=3, min_periods=1).mean()

# Add time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Interaction term: Power consumption per fan speed unit
df['power_per_rpm'] = df['power_consumption'] / df['fan_speed']

# Display data with new features
print("\nData with New Features:")
print(df)

# ----- Descriptive Statistics -----
print("\nDescriptive Statistics:")
print(df.describe())

# Skewness and Kurtosis
print("\nSkewness:")
print(df.select_dtypes(include=[np.number]).skew())

print("\nKurtosis:")
print(df.select_dtypes(include=[np.number]).kurt())

# ----- Data Normalization -----
print("\nStarting Data Normalization...")

scaler = MinMaxScaler()
df[['temperature', 'humidity', 'CO2', 'fan_speed', 'power_consumption']] = scaler.fit_transform(df[['temperature', 'humidity', 'CO2', 'fan_speed', 'power_consumption']])

print("\nNormalized Data:")
print(df)

# ----- Data Visualization -----
print("\nStarting Data Visualization...")

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Time series plot for temperature and humidity
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['temperature'], label='Temperature (Normalized)', color='red')
plt.plot(df['timestamp'], df['humidity'], label='Humidity (Normalized)', color='blue')
plt.title('Temperature and Humidity Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Values (Normalized)')
plt.legend()
plt.show()

# Histogram for CO2 levels
plt.figure(figsize=(8, 6))
sns.histplot(df['CO2'], bins=10, kde=True)
plt.title('CO2 Levels Distribution')
plt.show()

# ----- Save cleaned and engineered data -----
df.to_csv('cleaned_hvac_data.csv', index=False)
print("\nCleaned and engineered data saved to cleaned_hvac_data.csv")
