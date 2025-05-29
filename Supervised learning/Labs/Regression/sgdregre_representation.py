# To use both StandardScalar and SGDRegressor for linear regression

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train=np.array([[10],[12],[14],[16],[18]])
y_train=np.array([5,6,7,8,9])

scaler = StandardScaler()

# Fit and transform the data
x_norm = scaler.fit_transform(x_train)

# Train using SGD, but don’t try more than 1000 rounds of training
sgdr = SGDRegressor(max_iter=1000)

# Starts with random weights then goes through data point by point. 
# Updates the weights to reduce error via gradient descent
# Repeats untill either : loss stops improving or it hits max iterations 
sgdr.fit(x_norm, y_train)

# Shows a summary of the trained model
print(sgdr)


print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_norm)
# make a prediction using w,b. 
y_pred = np.dot(x_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")