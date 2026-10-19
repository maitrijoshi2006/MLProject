# STUDENT PERFORMANCE ANALYSIS AND PREDICTION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# 1. Create dataset using Pandas
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7, 1, 8, 4, 6,
                   3, 9, 5, 2, 7, 8, 1, 6, 4, 9],
    "Attendance": [50, 60, 65, 70, 75, 80, 40, 95, 68, 85,
                   55, 98, 72, 45, 88, 92, 35, 82, 66, 99],
    "Marks": [30, 35, 42, 50, 60, 70, 25, 90, 45, 75,
              32, 95, 55, 28, 80, 85, 20, 72, 48, 98]
}

df = pd.DataFrame(data)

# Add pass/fail label
df["Result"] = np.where(df["Marks"] >= 40, 1, 0)

print("STUDENT DATASET")
print(df)

# 2. Statistical calculations
print("\nSTATISTICS")
print("Mean:", df["Marks"].mean())
print("Median:", df["Marks"].median())
print("Mode:", df["Marks"].mode().tolist())
print("Variance:", df["Marks"].var())
print("Standard Deviation:", df["Marks"].std())

# NumPy and Math examples
import math

arr = np.array([10, 20, 30, 40, 50])
print("\nNumPy array:", arr)
print("Dimensions:", arr.ndim)
print("Shape:", arr.shape)
print("Size:", arr.size)
print("Sum:", np.sum(arr))
print("Mean:", np.mean(arr))
print("Sorted:", np.sort(arr))
print("Square root of 25:", math.sqrt(25))
print("Floor of 4.8:", math.floor(4.8))
print("Ceil of 4.2:", math.ceil(4.2))

# 3. Save and reload dataset
df.to_csv("student_data.csv", index=False)
df = pd.read_csv("student_data.csv")
print("\nLoaded CSV dataset:")
print(df.head())

# 4. Data preprocessing
# Introduce a missing value for demonstration
df.loc[0, "Attendance"] = np.nan
df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].mean()
)

# Min-max normalization
scaler = MinMaxScaler()
df[["StudyHours", "Attendance", "Marks"]] = scaler.fit_transform(
    df[["StudyHours", "Attendance", "Marks"]]
)

print("\nNormalized data:")
print(df.head())

# 5. Data visualization
plt.figure(figsize=(7, 4))
plt.scatter(df["StudyHours"], df["Marks"])
plt.xlabel("Normalized Study Hours")
plt.ylabel("Normalized Marks")
plt.title("Study Hours vs Marks")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(df["Marks"], bins=5, edgecolor="black")
plt.xlabel("Normalized Marks")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Marks")
plt.show()

# 6. Machine learning using Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest (20)": RandomForestClassifier(
        n_estimators=20, random_state=42
    ),
    "Random Forest (50)": RandomForestClassifier(
        n_estimators=50, random_state=42
    ),
    "Random Forest (100)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "KNN (1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (7)": KNeighborsClassifier(n_neighbors=7),
    "KNN (11)": KNeighborsClassifier(n_neighbors=11),
    "SVM Linear": SVC(kernel="linear"),
    "SVM Polynomial": SVC(kernel="poly"),
    "SVM RBF": SVC(kernel="rbf")
}

print("\nIRIS CLASSIFICATION RESULTS")
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"{name}: {accuracy * 100:.2f}%")

# 7. PCA dimensionality reduction
standardizer = StandardScaler()
X_scaled = standardizer.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA explained variance ratio:",
      pca.explained_variance_ratio_)
print("Total variance retained:",
      pca.explained_variance_ratio_.sum())

plt.figure(figsize=(7, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.show()

# 8. Simple Linear Regression
hours = np.array(data["StudyHours"]).reshape(-1, 1)
marks = np.array(data["Marks"])

regression = LinearRegression()
regression.fit(hours, marks)
predicted_marks = regression.predict(hours)

print("\nLINEAR REGRESSION")
print("Slope:", regression.coef_[0])
print("Intercept:", regression.intercept_)
print("Predicted marks for 5 study hours:",
      regression.predict([[5]])[0])

plt.figure(figsize=(7, 4))
plt.scatter(hours, marks, label="Actual Marks")
plt.plot(hours, predicted_marks, color="red",
         label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()

print("\nProject completed successfully!")
