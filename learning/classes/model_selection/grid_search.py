from sklearn.model_selection import GridSearchCV

# Here we pick what are the parameters we want to choose from, and form a
# dictionary. In this dictionary, the keys will be the names of the parameters,
# and the values will be the lists of possible values for each parameter.
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# We need to decide what metric we'll use to score each of the candidate models.
# In here, we'll use F1 Score.
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)

# get the best estimator
best_clf = grid_fit.best_estimator_

# Now you can use this estimator best_clf to make the predictions.
