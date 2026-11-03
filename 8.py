import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
data = pd.read_csv("student_pass.csv")
X = data[["Hours_Studied", "Attendance"]]
y = data["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("Predicted probabilities:", y_prob)
print("Predicted classes:", y_pred)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
new_student = pd.DataFrame({ "Hours_Studied": [4], "Attendance": [72]})
new_scaled = scaler.transform(new_student)
probability = model.predict_proba(new_scaled)[0, 1]
prediction = model.predict(new_scaled)[0]
print("\nNew Student:", new_student.iloc[0].to_dict())
print("Probability of passing:", round(probability, 4))
print("Predicted class:", "Pass" if prediction == 1 else "Fail")
plt.scatter(data["Hours_Studied"], data["Attendance"], c=data["Pass"])
plt.xlabel("Hours Studied")
plt.ylabel("Attendance")
plt.title("Student Pass/Fail Dataset")
plt.show()