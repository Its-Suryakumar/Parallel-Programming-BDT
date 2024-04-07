import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("sensor_data.csv", header=None, names=["SensorReading"])

# Extract the input feature (X) and create the target variable (y) from row indices
X = df[["SensorReading"]]
y = np.arange(len(df))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Function to fit the model and make predictions
def fit_model(X_train, y_train, X_test):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    return model.predict(X_test)

# Perform parallel processing using joblib
predictions = Parallel(n_jobs=-1)(delayed(fit_model)(X_train, y_train, X_test) for _ in range(10))

# Calculate mean squared error for each set of predictions
mse_values = [mean_squared_error(y_test, pred) for pred in predictions]

# Print mean squared error for each set of predictions
for i, mse in enumerate(mse_values):
    print(f"Mean Squared Error for set {i+1}: {mse}")

# Average mean squared error across all sets
average_mse = sum(mse_values) / len(mse_values)
print("Average Mean Squared Error:", average_mse)
