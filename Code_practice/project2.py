import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load dataset
data = pd.read_csv(
    "D:\\Learning\\Machine Learning\\Code_practice\\training_data_final.csv"
)  # Load your dataset

X = data[["sem_present_count"]]  # Features
y = data["semester_evaluation_gtu_mark"]  # Target variable

model = LinearRegression()
model.fit(X, y)
predicted_score = model.predict(X)


# Valid regression metrics
MAE = mean_absolute_error(y, predicted_score)
MSE = mean_squared_error(y, predicted_score)
RMSE = np.sqrt(MSE)
r2 = r2_score(y, predicted_score)

# Show result
print("Mean Absolute Error (MAE):", round(MAE, 2))
print("Mean Squared Error (MSE):", round(MSE, 2))
print("Root Mean Squared Error (RMSE):", round(RMSE, 2))
print("R-squared (R2):", round(r2, 2))


# histogram
# plt.figure(figsize=(10, 6))
# plt.hist(y, bins=20, alpha=0.5, label='Actual Scores', color='blue')
# plt.hist(predicted_score, bins=20, alpha=0.5, label='Predicted Scores', color='orange')
# plt.title('Distribution of Actual vs Predicted Scores')
# plt.xlabel('Scores')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Scores")
plt.plot(X, predicted_score, color="orange", linewidth=2, label="Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.xlabel("sem_present_count")
plt.ylabel("semester_evaluation_gtu_mark")
plt.legend()
plt.grid(True)
plt.show()
