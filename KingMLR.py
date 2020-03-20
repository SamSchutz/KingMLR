# Multiple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sstats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer
from yellowbrick.regressor import ResidualsPlot

# Importing and cleaning Kings home data
data = pd.read_csv('King County Homes.csv')
data = data.drop(labels='ID', axis=1)



# MODEL 1 - BASE Multiple Linear Regression model


# Split data in Training and Test set
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit MLR to Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test results
y_pred = regressor.predict(X_test)

score = r2_score(y_test, y_pred)
print(score)
# R^2 of .696

#Residuals
visualizer = ResidualsPlot(regressor)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

#Errors
print('Mean Absolute Error for model1:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error for model1:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error for model1:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# MODEL 2 - LOG TRANSFORMED PRICE


# Looking for normality issues; data is heavily right_skewed
data.hist(column='price', bins=100)

# Power transforming price
trans = data.copy()
trans['price'] = (np.log10(trans['price']))


# Split data in Training and Test set
X = trans.iloc[:, 1:]
y = trans.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit MLR to Training set
regressor2 = LinearRegression()
regressor2.fit(X_train, y_train)

# Predict test results
y_pred = regressor2.predict(X_test)

score2 = r2_score(y_test, y_pred)
print(score2)

# Residuals
visualizer2 = ResidualsPlot(regressor2)
visualizer2.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer2.score(X_test, y_test)  # Evaluate the model on the test data
visualizer2.show()                 # Finalize and render the figure

#Errors
print('Mean Absolute Error for model2:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error for model2:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error for model2:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# MODEL 3 - LOG PRICE TRANSFORM + BOX-COX




# Seaborn pairplot for determining skew of features to transform
#sns.pairplot(X_test)


# Need to transform sqft_lot15, long, sqft_above, sqft_living
trans2 = trans.copy()
f = sstats.boxcox(trans2['sqft_lot15'])
trans2['sqft_lot15'] = f[0]
f2 = sstats.boxcox(trans2['sqft_above'])
trans2['sqft_above'] = f2[0]
f3 = sstats.boxcox(trans2['sqft_living'])
trans2['sqft_living'] = f3[0]

# Split data in Training and Test set
X = trans2.iloc[:, 1:]
y = trans2.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit MLR to Training set
regressor3 = LinearRegression()
regressor3.fit(X_train, y_train)

# Predict test results
y_pred = regressor3.predict(X_test)
score3 = r2_score(y_test, y_pred)
print(score3)


# CHECK RESIDUALS
visualizer3 = ResidualsPlot(regressor3)
visualizer3.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer3.score(X_test, y_test)  # Evaluate the model on the test data
visualizer3.show()   # Finalize and render the figure

# Errors
print('Mean Absolute Error for model3:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error for model3:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error for model3:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
