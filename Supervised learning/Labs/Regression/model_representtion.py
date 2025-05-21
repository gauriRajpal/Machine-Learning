#Goal is to  Learn to implement the model  𝑓𝑤,𝑏 for linear regression with one variable

#We would be using many notation 
# x=Training examples for the features values       Represented by x_train
# y= Training examples for the target values        Represented by y_train
# x(i),y(i)= ith Training example                   Represented by x_i,y_i
# m= Number of training examples                    Represented by m
# w= Parameter weight                               Represented by w
# b= Parameter bias                                 Represented by b
# fw,b(x(i))= The result of the model evaluation at  𝑥(𝑖) parameterized by  𝑤,𝑏 :  𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏
#                                                   Represented by f_wb


# TOOLS
# NumPy  -> Popular library for scientific calculation
# Matplotlib -> Popular library for plotting data


import numpy as np                              #For importing the library
import matplotlib.pyplot as plt


#This creates the x_train and y_train variables. The data is stored in one-dimensional NumPy arrays.
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# Now finding m. 
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")


# Accessing the items in the array
i = 1 
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# Now for plotting the data matplotlib library is used
# Using the 'scatter()' function in the matplotlib library
# The function arguments 'marker' and 'c' show the points as red crosses (the default is blue dots).

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title
plt.title("Housing Prices")

# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')

# Set the x-axis label
plt.xlabel('Size (1000 sqft)')

#Shows the plotted data
plt.show()


# Using the values of w and b
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")




def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


#We would call the compute_model_output function and plot  the data
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


# w and b does not represent the line which matches the data well 
# So we take turns to find the correct value of w and b
# In this case the value is 𝑤=200 and 𝑏=100