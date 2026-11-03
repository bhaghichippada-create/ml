import numpy as np
# Step 1: Create the data
x = np.array([10, 20, 30, 40, 50])
y = np.array([15, 25, 35, 45, 55])
# Probability values for Expected Value
p = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
# a. Expected Value
# Formula: E(X) = Σ (x × p)
expected_value = np.sum(x * p)
print("Expected Value =", expected_value)
# b. Mean
# Formula: Mean = Σx / n
mean = np.mean(x)
print("Mean =", mean)
# c. Standard Deviation
# Formula: σ = √( Σ(x-μ)² / n )
std = np.std(x)
print("Standard Deviation =", std)
# d. Variance
# Formula: σ² = Σ(x-μ)² / n
variance = np.var(x)
print("Variance =", variance)
# e. Covariance
# Formula:
# Cov(X,Y) = Σ[(Xi-X̄)(Yi-Ȳ)] / (n-1)
covariance = np.cov(x, y)[0, 1]
print("Covariance =", covariance)
# f. Covariance Matrix
# Matrix containing variances and covariance values.
#
cov_matrix = np.cov(x, y)
print("Covariance Matrix =")
print(cov_matrix)