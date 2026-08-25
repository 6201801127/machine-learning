import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# Classification metrics are used to evaluate the performance of classification models. Here are some commonly used metrics:
# True answers  [What actually happend]
y_true = [1, 0, 1, 1, 0, 1, 0]
# Model predictions [What the model predicted]
y_pred = [1, 0, 1, 0, 0, 1, 1]

# Evaluation metrics for classification
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("R2 Score:", r2_score(y_true, y_pred))

# Confusion matrix

y1_true = [1, 1, 0, 0, 1, 0, 1, 0]
y1_pred = [1, 0, 0, 0, 1, 1, 1, 0]

cm = confusion_matrix(y1_true, y1_pred)
print("Confusion Matrix:")
print(cm)


# MAE (MEAN ABSOLUTE ERROR) is a metric used to evaluate the performance of regression models. It measures the average absolute difference between the predicted values and the actual values. A lower MAE indicates better model performance.

# MSE (MEAN SQUARED ERROR) is another metric used to evaluate regression models. It calculates the average of the squared differences between the predicted values and the actual values. A lower MSE indicates better model performance.

# RMSE (ROOT MEAN SQUARED ERROR) is the square root of the MSE. It provides a measure of the average magnitude of the errors in the same units as the target variable. A lower RMSE indicates better model performance.

# real score
real_score = [90, 60, 80, 100]

# modal guess
predicted_score = [85, 70, 70, 95]

mae = mean_absolute_error(real_score, predicted_score)
mse = mean_squared_error(real_score, predicted_score)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
