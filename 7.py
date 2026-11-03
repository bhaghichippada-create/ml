import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv('Salary_Data_ML.csv')
X=data[['YearsExperience','Age','EducationLevel']]
y=data['Salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(pd.DataFrame({'Feature':X.columns,'Coefficient':model.coef_}))
y_pred=model.predict(X_test)
print(pd.DataFrame({'Actual':y_test.values,'Predicted':y_pred}))
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show() 