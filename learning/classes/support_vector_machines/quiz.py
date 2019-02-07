# from sklearn.svm import SVC
# model = SVC()
# model.fit(x_values, y_values)

# In the example above, the model variable is a support vector machine model
# that has been fitted to the data x_values and y_values. Fitting the model
# means finding the best boundary that fits the training data. Let's make two
# predictions using the model's predict() function.

# print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
# [[ 0., 1.]]

# The model returned an array of predictions, one prediction for each input
# array. The first input, [0.2, 0.8], got a prediction of 0.. The second input,
# [0.5, 0.4], got a prediction of 1..

# Hyperparameters
# When we define the model, we can specify the hyperparameters. As we've seen in
# this section, the most common ones are

# C: The C parameter.
# kernel: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
# degree: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
# gamma : If the kernel is rbf, this is the gamma parameter.
# For example, here we define a model with a polynomial kernel of degree 4, and a C parameter of 0.1.

# model = SVC(kernel='poly', degree=4, C=0.1)

# Goal: define a model that gives 100% accuracy on the sample dataset
# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=27, C=10)
model.fit(X, y)

# Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
