import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

# Read Iris CSV file
df = pd.read_csv("iris.csv")

# Display first 10 rows
print(df.head(10))

print("Shape:", df.shape)
print(df.columns)
print(df.info())
print(df)
print(df.dtypes)
print(df.describe())

# Check missing values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())

# Box plot
df.boxplot(figsize=(8, 5))
plt.show()

# Histogram
df.hist(figsize=(10, 8))
plt.show()

# Scatter plot
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Pair plot
sns.pairplot(df)
plt.show()

# Correlation
corr = df.corr(numeric_only=True)
print(corr)

# Correlation heatmap
corr = df.iloc[:, :-1].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Box plot using Seaborn
sns.boxplot(data=df)
plt.show()

# Count plot
sns.countplot(x='Species', data=df)
plt.show()