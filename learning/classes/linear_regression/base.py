# For your linear regression model, you'll be using scikit-learn's
# LinearRegression class. This class provides the function fit() to fit
# the model to your data.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_values, y_values)

# In the example above, the model variable is a linear regression model that
# has been fitted to the data x_values and y_values. Fitting the model means
# finding the best line that fits the training data. Let's make two predictions
# using the model's predict() function.

print(model.predict([ [127], [248] ]))
# ---> [[ 438.94308857, 127.14839521]]

# The model returned an array of predictions, one prediction for each input
# array. The first input, [127], got a prediction of 438.94308857. The second
# input, [248], got a prediction of 127.14839521. The reason for predicting on
# an array like [127] and not just 127, is because you can have a model that
# makes a prediction using multiple features. We'll go over using multiple
# variables in linear regression later in this lesson. For now, let's stick
# to a single value.