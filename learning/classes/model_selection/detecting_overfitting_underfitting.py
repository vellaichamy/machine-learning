# Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
#estimator = LogisticRegression()

### Decision Tree
#estimator = GradientBoostingClassifier()

### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)

### estimator, is the actual classifier we're using for the data, e.g., LogisticRegression() or GradientBoostingClassifier().
### X and y is our data, split into features and labels.
### train_sizes are the sizes of the chunks of data used to draw each point in the curve.
### train_scores are the training scores for the algorithm trained on each chunk of data.
### test_scores are the testing scores for the algorithm trained on each chunk of data.

### the function uses 3 fold cross validation
### training and testing SCORE is the opposite of ERROR; so the higher the error, the lower the score

# train_sizes, train_scores, test_scores = learning_curve(
#     estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))