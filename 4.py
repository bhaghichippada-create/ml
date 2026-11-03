import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Categorical_Data.csv")

print("Original Dataset")
print(dataset)

print("\nMissing Values")
print(dataset.isnull().sum())

imputer = SimpleImputer(strategy='mean')
dataset[['Age', 'Salary']] = imputer.fit_transform(dataset[['Age', 'Salary']])

print("\nDataset After Replacing Missing Values")
print(dataset)

X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 3])],
    remainder='passthrough'
)

X = ct.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded Features")
print(X)

print("\nEncoded Target")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\nScaled Training Data\n")
print(X_train)

print("\nScaled Testing Data\n")
print(X_test)