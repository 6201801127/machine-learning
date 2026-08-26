from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # type: ignore

# Linear regration
X = [[1], [2], [3], [4], [5]]
y = [40, 50, 65, 75, 90]
model = LinearRegression()
model.fit(X, y)
hours = float(input("Enter the number of hours studied: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted marks for studying {hours} hours: {predicted_marks[0]}")

# Classification
# 1. Logistics Regression 2. KNN (K-Nearest Neighbors) 3. Decision Tree 4. Random Forest 5. Support Vector Machine (SVM) 6. Naive Bayes

# Logistics regression
lx = [[1], [2], [3], [4], [5]]
ly = [0, 0, 1, 1, 1]
model = LogisticRegression()
model.fit(lx, ly)
hrs = float(input("Enter the number of hours studied: "))
result = model.predict([[hrs]])[0]
if result == 0:
    print(f"Predicted class for studying {hrs} hours: Not Passed")
else:
    print(f"Predicted class for studying {hrs} hours: Passed")


# KNN (K-Nearest Neighbors)
kx = [[180, 7], [200, 7.5], [250, 8], [300, 8.5], [330, 9], [350, 9.5]]
ky = [0, 0, 0, 1, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(kx, ky)
weight = float(input("Enter the weight of the fruit (in Grams): "))
height = float(input("Enter the height of the fruit (in CM): "))
result = model.predict([[weight, height]])[0]
if result == 0:
    print(f"Predicted class for the fruit with weight {weight} g and height {height} cm: Apple")
else:
    print(f"Predicted class for the fruit with weight {weight} g and height {height} cm: Orange")

# Decision Tree

dx = [[7, 2], [8, 3], [9, 8], [10, 9]]
dy = [0, 0, 1, 1]  # 0=apple, 1=orange
model = DecisionTreeClassifier()
model.fit(dx, dy)
size = float(input("Enter the size of the fruit (in CM): "))
shade = float(input("Enter the shade of the fruit (1-10): "))
result = model.predict([[size, shade]])[0]
if result == 0:
    print(f"Predicted class for the fruit with size {size} cm and shade {shade}: Apple")
else:
    print(f"Predicted class for the fruit with size {size} cm and shade {shade}: Orange")
