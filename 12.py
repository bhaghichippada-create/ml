import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("random_forest_data.csv")
X = data[["Age", "Salary", "Experience"]]
y = data["Purchased"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42, stratify=y)
rf = RandomForestClassifier( n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Actual Values: ", list(y_test))
print("Predicted Values: ", list(y_pred))
print("\nRandom Forest Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 