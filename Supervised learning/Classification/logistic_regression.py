# Implementing the logistic regression model using scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Separate points by class
# It tells to seperate the points based on the output
class_0 = X[y == 0]
class_1 = X[y == 1]

# Plot

# plt.figure()->Creates a new figure window or canvas to draw  plots.
# figsize=(6, 6)->Specifies the size of the figure in inches — in this case, 6 inches wide by 6 inches tall.
plt.figure(figsize=(6, 6))

# class_0[:,0] would call the rows with column 0 and class_0[:,1] would call all the rows with column 1
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

# Add labels and legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot of X with Labels y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

lr_model = LogisticRegression()
lr_model.fit(X, y)

# Predictions can be made by calling the 'predict' function.
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)

#The accuracy of this model can be calculated by calling the 'score' function.
print("Accuracy on training set:", lr_model.score(X, y))