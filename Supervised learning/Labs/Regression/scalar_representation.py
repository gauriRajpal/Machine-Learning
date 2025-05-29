from sklearn.preprocessing import StandardScaler
import numpy as np

# Example data
x=np.array([[10],[12],[14],[16],[18]])
print(f"Original dataset = {x}")

# To find mean of original training set
print(f"The mean of original dataset is {np.mean(x)}")

# To find standard deviation of original training set
print(f"The standard deviation of original dataset is {np.std(x)}")

# Create a scaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(x)

print(f"Scaled data set is {X_scaled}")

# To find mean of scaled training set
print(f"The mean of scaled dataset is {np.mean(X_scaled)}")

# To find standard deviation of scaled training set
print(f"The standard deviation of scaled dataset is {np.std(X_scaled)}")
