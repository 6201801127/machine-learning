import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

data = {
    "Name": ["Ajay", "Vijay", "Ravi", "Suresh", "Anil"],
    "Age": [25, 30, 22, None, 35],
    "Salary": [50000, 60000, 45000, 70000, None],
}


df = pd.DataFrame(data)
# print(df)
# print(df.isnull().sum())  # Count of missing values in each column
drof_df = df.dropna()  # Drop rows with any missing values
# print(drof_df)
age_mean = df["Age"].mean()
salary_mean = df["Salary"].mean()
df["Age"] = df["Age"].fillna(age_mean)  # Fill missing Age with mean
df["Salary"] = df["Salary"].fillna(salary_mean)  # Fill missing Salary with mean
# print(df)


sample_df = pd.read_csv(
    "D:\\Learning\\Machine Learning\\Code_practice\\sample_student_data.csv"
)  # Read data from a CSV file
# print(sample_df.head())  # Display the first few rows of the DataFrame
# print(sample_df.tail())  # Display the last few rows of the DataFrame

df_label = sample_df.copy()

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
df_label["Gender_Encoded"] = label_encoder.fit_transform(df_label["gender"])
df_label["Passed_Encoded"] = label_encoder.fit_transform(df_label["passed"])
# print(sample_df_copy)  # Display the first few rows of the DataFrame with encoded columns

# one-hot encoding
df_label_encoder = pd.get_dummies(df_label, columns=["city"])
# print(df_label_encoder)  # Display the DataFrame with one-hot encoded columns for 'city'


data1 = {"StudyHours": [1, 2, 3, 4, 5], "TestScore": [40, 50, 60, 70, 80]}
# Feature Scaling
# 1. Standerd Scaler

df1 = pd.DataFrame(data1)
standers_scaler = StandardScaler()
standers_scaled = standers_scaler.fit_transform(df1)
print(
    pd.DataFrame(standers_scaled, columns=["StudyHours", "TestScore"])
)  # Display the standardized features

MinMax_Scaler = MinMaxScaler()
MinMax_scaled = MinMax_Scaler.fit_transform(df1)
print(
    pd.DataFrame(MinMax_scaled, columns=["StudyHours", "TestScore"])
)  # Display the min-max scaled features


# Train Test Split
X = df1[["StudyHours"]]
y = df1["TestScore"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)
