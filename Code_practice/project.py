import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv(
    "D:\\Learning\\Machine Learning\\Code_practice\\student_data.csv"
)  # Load your dataset
X = data[["Hour"]]  # Features
y = data["Score"]  # Target variable

model = LinearRegression()  # Create a linear regression model
model.fit(X, y)  # Fit the model to the data

predicted_score = model.predict(X)

# EValute
MAE = mean_absolute_error(y, predicted_score)
MSE = mean_squared_error(y, predicted_score)
RMSE = np.sqrt(MSE)

print("Mean Absolute Error (MAE):", MAE)
print("Mean Squared Error (MSE):", MSE)
print("Root Mean Squared Error (RMSE):", RMSE)

new_hour = float(input("Enter the number of hours studied: "))  # Get user input
new_score = model.predict([[new_hour]])  # Predict the score for the new input
print(
    f"Predicted score for studying {new_hour} hours: {new_score[0]:.2f}"
)  # Print the predicted score
