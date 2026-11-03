import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('Salary_Data_ML.csv')

# X = Independent variable
X = data[['YearsExperience']]

# y = Dependent variable
y = data['Salary']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Display intercept and slope
print(model.intercept_)
print(model.coef_[0])

# Predict test data
y_pred = model.predict(X_test)

# Compare Actual vs Predicted
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# Evaluate model
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Plot regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train), color='red')
plt.show()


# Exercise - 7:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('Salary_Data_ML.csv')

# X = Multiple independent variables
X = data[['YearsExperience', 'Age', 'EducationLevel']]

# y = Dependent variable
y = data['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)