# KingMLR
This is a Python data project using scikit-learn's Multiple Linear Regression models and feature power tranformations to predict home values in King County, WA. This project uses the following libraries: `numpy matplotlib pandas seaborn scipy scikit-learn yellowbrick`

Future ideas/plans include: using cross-validation to split the data instead of a traditional 80/20 split.

## Files 
- King County Homes.csv - (16187 X 20) dataset with house prices and predictors 
- KingMLR.py - A Multiple Linear Regression model with error metrics and visualizations
- Graphics folder - contains residual plots and pairplots of the dataset and models

## Base Mutliple Linear Regression Model
First lets see some of the descriptive statistics of the data.
```
              price      bedrooms  ...  sqft_living15     sqft_lot15
count  1.618700e+04  16187.000000  ...   16187.000000   16187.000000
mean   5.428020e+05      3.374374  ...    1989.630815   12677.886637
std    3.696331e+05      0.944437  ...     688.003602   27553.569520
min    7.500000e+04      0.000000  ...     399.000000     651.000000
25%    3.246235e+05      3.000000  ...    1490.000000    5100.000000
50%    4.510000e+05      3.000000  ...    1840.000000    7601.000000
75%    6.488760e+05      4.000000  ...    2370.000000   10080.000000
max    7.700000e+06     33.000000  ...    6210.000000  871200.000000
```

After reading in the data using the read_csv function in the pandas library, the first key step in using the data is spliting the dataset into the variable we want to predict vs. the predictors themselves. This is done using the `.iloc` function to seperate the code into X and y DataFrames. We can then instantiate the `train_test_split` function from scikit-learn to split the data into a training set and a test set to test how well our model preforms.
```Python
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

Now to actually create the MLR model using LinearRegression from scikit-learn's linear_model. With this first model, no changes will be made to the model itself--as we want to see how new MLR models compare to putting in the bare minimum effort.
```Python
# Fit MLR to Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test results
y_pred = regressor.predict(X_test)

# R^2 score of our model
score = metrics.r2_score(y_test, y_pred)
print(score)
```
```
r^2 = .696
```
Now, in order to avoid being lazy we will check the residuals of our model using `yellowbrick` and see if there are any glaring issues. In addition, let's check the Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error for the model.
```Python
#Residuals 
visualizer = ResidualsPlot(regressor)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

#Errors
print('Mean Absolute Error for model1:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error for model1:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error for model1:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```
![Residuals1](/Graphics/Model1resid.png)
```
Mean Absolute Error for model1: 129182.38876785948
Mean Squared Error for model1: 45594354996.200386
Root Mean Squared Error for model1: 213528.34705537432
```
WOW, that's pretty bad. Our RMSE is **HUGE** compared to our mean of price and our residuals are completely messed up. Looks like there is plenty to do here in terms of fixing up the data.

## Model 2: Multiple Linear Regression with variable transformations

Okay, so let's really dig into the data and see how messed up things are using histograms for examining the skew of 'price'.
```Python
data.hist(column='price', bins=100) 
```
![histogram](/Graphics/hist1.png)



