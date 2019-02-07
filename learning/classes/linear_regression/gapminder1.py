import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
bmi = bmi_life_data[['BMI']] # note the double brackets
life_expectancy = bmi_life_data[['Life expectancy']]

print(bmi.head())
print(life_expectancy.head())

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi, life_expectancy)

# Make a prediction for BMI of 21.07931 using the model
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
# --> [[60.31564716]]
