# KingMLR
This is a Python data project using scikit-learn's Multiple Linear Regression models and feature power tranformations to predict home values in King County, WA. This project uses the following libraries: `numpy matplotlib pandas seaborn scipy scikit-learn yellowbrick`

## Files 
- King County Homes.csv - (16187 X 20) dataset with house prices and predictors 
- King County Homes - Multiple Linear Regression.ipynb - where all of the analysis and model implementation is done
- Graphics folder - contains residual plots and pairplots of the dataset and models

## Looking at the data
First let's read in some of the data using `.head()`

```
	price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renovated	renovated	zipcode	lat	long	sqft_living15	sqft_lot15
0	221900	3	1.00	1180	5650	1.0	0	0	3	7	1180	0	1955	0	0	98178	47.5112	-122.257	1340	5650
1	538000	3	2.25	2570	7242	2.0	0	0	3	7	2170	400	1951	1991	1	98125	47.7210	-122.319	1690	7639
2	180000	2	1.00	770	10000	1.0	0	0	3	6	770	0	1933	0	0	98028	47.7379	-122.233	2720	8062
3	510000	3	2.00	1680	8080	1.0	0	0	3	8	1680	0	1987	0	0	98074	47.6168	-122.045	1800	7503
4	1230000	4	4.50	5420	101930	1.0	0	0	3	11	3890	1530	2001	0	0	98053	47.6561	-122.005	4760	101930

```
#### Descriptive Statistics
Then lets see some of the descriptive statistics of the data.
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
#### Pairplot

![pairplot](/Graphics/pairplot.png)

It looks like some variable have right-skewed distributions. Something we will fix with transformations down the road.

#### Removing Features
Because I'm not using any tree-based models--longitude, latitude, and zip would just confuse the model.
```Python
data = data.drop(labels=['long', 'lat', 'zipcode'], axis=1)
```

#### Checking for Null values
```Python
data.isnull().any()
```
```
price            False
bedrooms         False
bathrooms        False
sqft_living      False
sqft_lot         False
floors           False
waterfront       False
view             False
condition        False
grade            False
sqft_above       False
sqft_basement    False
yr_built         False
yr_renovated     False
renovated        False
sqft_living15    False
sqft_lot15       False
dtype: bool
```
#### Mean
And finally finding the mean value of 'price' to have a baseline for looking at how our errors stack up using `data['price'].mean()` which return the mean price in USD.

Histogram:
![hist](/Graphics/hist1.png)

```
542802.0316920986
```
## Creating the Dummy model

After reading in the data using the read_csv function in the pandas library, the first key step in using the data is spliting the dataset into the variable we want to predict vs. the predictors themselves. This is done using the `.iloc` function to seperate the code into X and y DataFrames. We can then instantiate the `train_test_split` function from scikit-learn to split the data into a training set and a test set to test how well our model preforms.
```Python
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

Now to actually create a model. First, we will use the DummyRegressor from scikit-learn. With this first model, we will be given the worst-case scenario in terms of Mean Absolute Error in dollars. Cross Validation of 5 scores will be used to show which "type" of Multiple Linear Regression we should use
```Python
dummyreg = DummyRegressor(strategy='mean')
scores = cross_val_score(dummyreg, X, y,
                         scoring=metrics.make_scorer(metrics.mean_absolute_error),
                         cv=5)
scores
```
```
array([237472.60747177, 238184.50756892, 229002.01176648, 238158.57318789,
       237013.63527984])
```
```Python
print("Average Mean Absolute Error: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
```
Average Mean Absolute Error: **235966.27 (+/- 7019.61)**

**Pretty horrible**, which is to be expected.

## Model 2: Scikit-Learn's LinearRegression()

```Python 
mlr1 = LinearRegression()
scores = cross_val_score(mlr1, X, y,
                         scoring=metrics.make_scorer(metrics.mean_absolute_error),
                         cv=5)
print("Average Mean Absolute Error: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
```
Average Mean Absolute Error: **141855.23 (+/- 8407.33)**

Residuals:

![resid1](/Graphics/resid1.png)

## Model 3: Scikit-Learn's Normalized LinearRegression()

```Python
lr2 = LinearRegression(normalize=True)
scores = cross_val_score(mlr2, X, y,
                         scoring=metrics.make_scorer(metrics.mean_absolute_error),
                         cv=5)
print("Average Mean Absolute Error: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

```
Average Mean Absolute Error: **141860.65 (+/- 8394.38)**

residuals:

![resid2](/Graphics/resid2.png)

Looks about the same to me. Nothing really has changed in the residuals either.

## Model 4: MLR with Recursive Feature Selection

```Python
m = feature_selection.RFECV(LinearRegression(normalize=True), cv=5)

m.fit(X, y)

m.support_

metrics.mean_absolute_error(y_test, m.predict(X_test))
```
```
array([ True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True])
```
It looks like here it only removed one feature: `sqft_lot` and return a mean absolute error of **235427.88012998947**. Pretty trash. Something might be wrong in the way I've set up the scoring, but it's not handling this data well.

## Model 5: MLR with log10(price) and Box-Cox transforms on skewed predictors

```Python
data3 = data2.copy()
f = sstats.boxcox(data3['sqft_lot15'])
data3['sqft_lot15'] = f[0]
f2 = sstats.boxcox(data3['sqft_above'])
data3['sqft_above'] = f2[0]
f3 = sstats.boxcox(data3['sqft_living'])
data3['sqft_living'] = f3[0]

X3 = data3.iloc[:, 1:]
y3 = data3.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.2,
                                                    random_state = 0)
mlr4 = LinearRegression()
mlr4.fit(X_train, y_train)

metrics.mean_absolute_error(10**y_test, 10**mlr4.predict(X_test))
```
Only transforming the price really messes up our residuals so let's not do that.
Log only residuals:
![resid4](/Graphics/resid4.png)

Residuals:
![resid3](/Graphics/resid3.png)

Gives us a Mean Absolute error of **130470.19222297132** and a similar residuals plot to the first two Linear Regression models. 

# Conclusion

Overall it seems that Multiple Linear Regression is not a good fit for this type of problem--especially when ensemble bagged and boosted methods exist. However, this was great practice working with statistical transformations and a general deep-dive into scikit-learn's LinearRegression() model.
