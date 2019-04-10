# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# scipy
from scipy.stats import norm

# reading the files
train = pd.read_csv("E:/Big_Mart/Train.csv")
test = pd.read_csv("E:/Big_Mart/Test.csv")

train.head()
test.head()

# check for shape
print(train.shape)
print(test.shape)

# knowing the statistical values of the data
train.describe()
test.describe()

# datatypes of the features
train.dtypes

# checking the null values
train.isnull().sum()
test.isnull().sum()

# checking the number of unique values
train.apply(lambda x: len(x.unique()))

# Exploratory data analysis
# ---------------------------------------------------------------------------------------
# Combine Datasets
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)
data.shape
data.describe()

# Unique values in every column
categorical_columns = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
categorical_columns = [x for x in categorical_columns if x not in \
                       ['Item_Identifier', 'Outlet_Identifier', 'source']]

# print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s' % col)
    print(train[col].value_counts())

# plot style
sb.set_style('darkgrid')

# Plot Graph
plt.figure(figsize=(10, 7))

# let's find the correlation between the features
cor = train.corr()
sb.heatmap(cor, vmax=0.8, square=True, annot=True)

# scatter plot matrix
sb.pairplot(train, hue='Outlet_Size', palette=['black', 'orange', 'blue'])
plt.plot()

# checking the counts of the outlets store with respect to their location
sb.countplot(data=train, x=train.Outlet_Type, hue=train.Outlet_Location_Type)

# More data visualisations
sb.distplot(train['Item_MRP'])
box_outlet_sales = sb.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=train)
box_outlettype_sales = sb.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=train)
box_itemtype_sales = sb.boxplot(x='Item_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=train)
box_visibility_outlet = sb.boxplot(x='Outlet_Identifier', y='Item_Visibility', data=train)

# -------------------------------------------------------------------------------------------------------

# Data Cleaning

# checking shapes of dataset
print(data.shape)

sb.distplot(train.Item_MRP, fit=norm)

# Missing Values
data.isnull().sum()
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.Item_Weight = data.groupby('Outlet_Type')['Item_Weight'].apply(lambda x: x.fillna(x.mean()))
data.Item_Weight.isnull().sum()

# filling the null values with the mean value
data.Item_Weight.fillna(0, inplace=True)

# checking the null replacement
data.Item_Weight.isnull().sum()

# Impute missing values of outlet_size by mode
from scipy.stats import mode

data.Outlet_Size.value_counts(dropna=False)

# General mode of all outlet size irrespective of outlet type
data['Outlet_Size'].mode()

# Mode per outlet type
missing_value = data['Outlet_Size'].isnull()
data.loc[missing_value, 'Outlet_Size'] = data.loc[missing_value, 'Outlet_Type'].apply(lambda x: mode(x))
print(sum(data['Outlet_Size'].isnull()))

# checking null remaining
data.Outlet_Size.isnull().sum()

# checking the missing Values for Item_weight and Outlet_size
data.isnull().sum()

# ---------------------------------------------------------------------------------------------------------

# Features
# check wheather to combine SuperMarket 2 & 3 on outlet sales basis
data.pivot_table(value='Item_Outlet_Sales', index='Outlet_Type')

# Change year to age
data.Outlet_Establishment_Year = [2013 - s for s in data.Outlet_Establishment_Year]
type(data.Outlet_Establishment_Year)
data['Outlet_Establishment_Year'].describe()

# visibilty in practical can't be 0, hence we replace the 0 values with the average ones
data.Item_Visibility.replace(0, data.Item_Visibility.mean(), inplace=True)
data['Item_Visibility'].describe()

# grab from first 2 characters of ID
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

# rename them
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable',
                                                             'DR': 'Drinks'})
data['Item_Type_Combined'].value_counts()

# combining Item_Fat_Content
sb.countplot(x='Item_Fat_Content', data=data)

fat = {"Item_Fat_Content": {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}}
data.replace(fat, inplace=True)

# taking non-consumable as seperate catergory in low_fat
data.loc[data['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

# checking frequency
data.Item_Fat_Content.value_counts(dropna=False)
data.Item_Type.value_counts(dropna=False)

# compress the classes based on their similar product
variety = {"Item_Type": {"Fruits and Vegetables": "Breads", 'Household': 'Breads', 'Dairy': 'Breads',
                         'Baking Goods': 'Breads', 'Health and Hygiene': 'Breads', 'Breakfast': 'Breads',
                         'Soft Drinks': 'Others', 'Canned': 'Others', 'Hard Drinks': 'Others', 'Frozen Foods': 'Meat',
                         'Starchy Foods': 'Meat', 'Seafood': 'Meat'}}

data.replace(variety, inplace=True)
data.Item_Type.value_counts()

# importing the label encoder from sklearn
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

# creating new variable for Outlet_Identifier
data['Outlet'] = label.fit_transform(data['Outlet_Identifier'])
features = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
label = LabelEncoder()
for i in features:
    data[i] = label.fit_transform(data[i])

# getting dummies of the data
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type',
                                     'Outlet_Size', 'Outlet_Type', 'Item_Type_Combined', 'Outlet'])
data.dtypes

# drop the columns which have been converted to different types
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

# divide into test and train
train = data.loc[data['source'] == "train"]
test = data.loc[data['source'] == "test"]

# drop unnecessary columns
train.drop(['source'], axis=1, inplace=True)
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)

# Export files as modified versions:
train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)

# Mean based
mean_sales = train['Item_Outlet_Sales'].mean()

# Define dataframe with IDs for submission:
df1 = test[['Item_Identifier', 'Outlet_Identifier']]
df1['Item_Outlet_Sales'] = mean_sales

# export submission file
df1.to_csv("algorithm.csv", index=False)

# making the target
target = 'Item_Outlet_Sales'
IDcolumn = ['Item_Identifier', 'Outlet_Identifier']

from sklearn import cross_validation, metrics


def modelfit(algorithm, xtrain, xtest, predictors, target, IDcolumn, filename):
    # Fit the algorithm on the data
    algorithm.fit(xtrain[predictors], xtrain[target])

    # Predict training set:
    xtrain_predictions = algorithm.predict(xtrain[predictors])

    # Perform cross-validation:
    cv_score = cross_validation.cross_val_score(algorithm, xtrain[predictors], xtrain[target], cv=20,
                                                scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print
    "\nModel Report"
    print
    "RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(xtrain[target].values, xtrain_predictions))
    print
    "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Predict on testing data:
    xtest[target] = algorithm.predict(xtest[predictors])

    # Export submission file:
    IDcolumn.append(target)
    submission = pd.DataFrame({x: xtest[x] for x in IDcolumn})
    submission.to_csv(filename, index=False)


# -----------------------------------------------------------------------------------------------------------------------------------------------

# Linear Regression Model
from sklearn.linear_model import LinearRegression

predictors = [x for x in train.columns if x not in [target] + IDcolumn]

# print predictors
reg = LinearRegression(normalize=True)
modelfit(reg, train, test, predictors, target, IDcolumn, 'Lin_reg.csv')
coef1 = pd.Series(reg.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor

predictors = [x for x in train.columns if x not in [target] + IDcolumn]

# print predictors
tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(tree, train, test, predictors, target, IDcolumn, 'decision_tree.csv')
coef2 = pd.Series(tree.feature_importances_, predictors).sort_values(ascending=False)
coef2.plot(kind='bar', title='Feature Importances')

# again checking prediction by changing top variables
predictors = ['Item_MRP', 'Outlet_Type_0', 'Outlet_5', 'Item_Weight']

tree1 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(tree1, train, test, predictors, target, IDcolumn, 'decision_tree1.csv')
coef3 = pd.Series(tree1.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Random Forest
from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target] + IDcolumn]

# print predictors
random = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
modelfit(random, train, test, predictors, target, IDcolumn, 'rf.csv')
coef4 = pd.Series(random.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

# again checking prediction by changing top 4 variables
predictors = [x for x in train.columns if x not in [target] + IDcolumn]

random1 = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=100, n_jobs=4)
modelfit(random1, train, test, predictors, target, IDcolumn, 'rf1.csv')
coef5 = pd.Series(random.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

# ---------------------------------------------------------------------------------------------------------------------------------------------