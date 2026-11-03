# Decision Tree using ID3-style criterion
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the Play Tennis dataset
data = pd.read_csv("play_tennis.csv")

X = data.drop(columns=["Play"])
y = data["Play"]

# Convert categorical attributes to dummy variables
X_encoded = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)

# ID3 uses entropy as the splitting criterion
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Predict test samples
y_pred = model.predict(X_test)

print("Decision Tree Rules:")
print(export_text(model, feature_names=list(X_encoded.columns)))

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Classify a new sample
new_sample = pd.DataFrame([{
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "Normal",
    "Wind": "Strong"
}])

new_encoded = pd.get_dummies(new_sample)
new_encoded = new_encoded.reindex(columns=X_encoded.columns, fill_value=0)

prediction = model.predict(new_encoded)

print("New Sample:", new_sample.iloc[0].to_dict())
print("Predicted class:", prediction[0])

# Display the decision tree
plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X_encoded.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("Decision Tree using ID3-style Entropy Criterion")
plt.show()