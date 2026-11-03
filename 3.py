import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read dataset
data = pd.read_csv("Employee_Data.csv")

# Check missing values
print(data.isnull().sum())

# Handle missing values
imp = SimpleImputer(strategy='mean')
data[['Age', 'Salary']] = imp.fit_transform(data[['Age', 'Salary']])

# Find outliers in Salary
Q1 = data['Salary'].quantile(0.25)
Q3 = data['Salary'].quantile(0.75)
IQR = Q3 - Q1

outliers = data[
    (data['Salary'] < Q1 - 1.5 * IQR) |
    (data['Salary'] > Q3 + 1.5 * IQR)
]

print("Outliers:")
print(outliers)

# Separate input and output
X = data.iloc[:, 0:3].values
y = data.iloc[:, 3].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("X_train:")
print(X_train)

print("X_test:")
print(X_test)