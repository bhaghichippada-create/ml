import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
train_data = pd.read_csv("naive_bayes_train.csv")
test_data = pd.read_csv("naive_bayes_test.csv")
features = ["Outlook", "Temperature", "Humidity", "Wind"]
X_train = train_data[features]
y_train = train_data["Play"]
X_test = test_data[features]
y_test = test_data["Play"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)
X_train_encoded = X_train_encoded.astype(int)
X_test_encoded = X_test_encoded.astype(int)
model = CategoricalNB()
model.fit(X_train_encoded, y_train) 
predictions = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, predictions)
result = pd.DataFrame({ "Actual": y_test.values, "Predicted": predictions})
print("Test Results:")
print(result)
print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")