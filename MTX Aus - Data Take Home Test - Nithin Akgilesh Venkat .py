#!/usr/bin/env python
# coding: utf-8

#     Author: Nithin Akgilesh Venkat
#     Date: 27.10.2021
#     Topic: Cats life expectancy prediction

# **Import required libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn import preprocessing, linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')


# **Load Dataset**

# In[2]:


df = pd.read_csv("cats.csv" , sep=',', index_col =0)
df.head()


# In[3]:


df


#     General overall view of the dataset

# In[4]:


# data overview
print ('Rows     : ', df.shape[0])
print ('Columns  : ', df.shape[1])
print ('\nFeatures : ', df.columns.tolist())
print ('\nMissing values :  ', df.isnull().sum().values.sum())
print ('\nUnique values :  \n', df.nunique())

df.info()
df.isnull().sum()


# **Data Prepocessing**

#     we can see two columns have missing values. lets do a quick data technique for filling them.

# In[5]:


# Converting the feature names to standard format for further processing
orig_cols = list(df.columns) 
new_cols = [] 
for col in orig_cols:     
    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower()) 

df.columns = new_cols


# In[6]:


df.describe()


#     The above describe() function shows us some info that needs to be fixed in the dataset before any data analysis.
#     1. Fill missing values using mean/mode inputers
#     2. number_vet_visits cannot be -1 
#     3. weight cannot be 0 for any cat
# 
#     The above issues can be fixed using filling values with mean/mode distributioon depending on the data distribution.

#     Outliers are unusual values in your dataset, and they can distort statistical analyses and violate their assumptions.Outliers increase the variability in your data, which decreases statistical power. Consequently, excluding outliers can cause your results to become statistically significant.

# In[7]:



# Remove the outliers using the interquartile range (IQR).
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]



# Print the dimensions of the cleaned dataset.
df.shape


#     Visualising data feature distribution to get an overall view
#     Box plots gives us the data distibution of individual features

# In[8]:


sns.boxplot(df.age_at_death)


#     For symmetric data distribution, we can use the mean value for inputing missing/incorrect values.
#     
#     Similar process will be carried out for other features as well

# In[9]:


df['age_at_death'].fillna(value=df['age_at_death'].mean(), inplace=True)


# In[10]:


sns.boxplot(df.hair_length)


#     When the data is skewed, it is good to consider using mode values for replacing the missing values

# In[11]:


df['hair_length'] = df['hair_length'].fillna(df['hair_length'].mode()[0])


#     Final check to confirm if the data is clean for further analysis

# In[12]:


df.info()
df.isnull().sum()


# In[13]:


sns.boxplot(df.number_of_vet_visits)


#     no# of vet visits cannot be -1 because last visit date provided the neccessary details
#     so assigning mean column in place of -1

# In[14]:


df["number_of_vet_visits"].replace({-1: (df['number_of_vet_visits'].mean()), 0: (df['number_of_vet_visits'].mean())}, inplace=True)


# In[15]:


sns.boxplot(df.weight)


#     weight cannot be 0 for a cat, especially when its old. So filling them with mode value as the data is skewed

# In[16]:


df["weight"].replace({0: (df['weight'].mean())}, inplace=True)


# In[17]:


df.describe()


# In[18]:


# analysing the data types of features
df.dtypes


# In[19]:


# Discard the metadata (breed, vet visits).
df = df.drop(['breed' ,'date_of_last_vet_visit'], axis=1)


# In[20]:


df


# **Data Exploration**

# In[21]:


# creating box plots

# Create a dictionary of columns representing the features of the dataset.
col_dict = {'age_at_death':1,'hair_length':2,'height':3,'number_of_vet_visits':4,'weight':5}

# Visualize the data for each feature using box plots.
plt.figure(figsize=(18,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(df[variable],whis=1.5)
                     plt.title(variable)

plt.show()
df.shape


#     A heatmap is a graphical representation where individual values of a matrix are represented as colors. A heatmap is very useful in visualizing the concentration of values between two dimensions of a matrix. This helps in finding patterns and gives a perspective of depth.

# In[22]:


# Plot heatmap to visualize the correlations.
plt.figure(figsize = (10, 8))
# sns.heatmap(df.corr(), annot = True)
sns.heatmap(df.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.title('Correlation between different features');


#     Since the no. of samples is less in the dataset, there seems to be less correlation between features 
# 

#     Scatter plots' primary uses are to observe and show relationships between two numeric variables. Relationships between variables can be described in many ways: positive or negative, strong or weak, linear or nonlinear. A scatter plot can also be useful for identifying other patterns in data.

# In[23]:



# Create a vector containing all the features of the dataset.
all_col = ['hair_length','height','number_of_vet_visits','weight']

plt.figure(figsize=(15,30))

# Plot each feature in function of the target variable (age_at_death) using scatter plots.
for i in range(len(all_col)):
    plt.subplot(7,3,i+1)
    plt.scatter(df[all_col[i]], df['age_at_death'])
    plt.xlabel(all_col[i])
    plt.ylabel('age_at_death')

plt.show()


# There is no evidence of any tight corelation in data, porobably the sample size could be an issue.(random scatter of the points)

# **Data Analysis**

# In[24]:


df_copy = df.copy()


# In[25]:


# Y label of the model will be age_at_death column. So we seperate it from the dataframe.
y = df['age_at_death']
df = df.drop(labels='age_at_death', axis=1)
df


# In[26]:


y = y.to_numpy(dtype='float64')
y


# In[27]:


# Dividing the dataset into train, validation and test sets. %80 training set, %10 validation set, %10 test set.

x_train, x_test, y_train, y_test = model_selection.train_test_split(df, y, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = model_selection.train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(f'X_train shape -->{x_train.shape}')
print(f'X_valid shape -->{x_valid.shape}')
print(f'X_test shape -->{x_test.shape}')
print(f'y_train shape -->{y_train.shape}')
print(f'y_valid shape -->{y_valid.shape}')
print(f'y_test shape -->{y_test.shape}')


# Standardization the features using Sklearn. Fitted the train set (because we only want to use these values) and transformed all sets.

# In[28]:



scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)


# **Model 1: Linear Regression**
# 
#     Using Linear Regression as the first model. Model is trained on the train set and made predictions for validation set. Mean Squared Error is used to compute the loss.

# In[29]:



model = linear_model.LinearRegression()
model.fit(X=x_train, y=y_train)
val_hat = model.predict(x_valid)
metrics.mean_squared_error(y_valid, val_hat)


# **Model 2: Poisson Regressor**
# 
#     Using Poisson Regressor as the second model. Defined the hyper-parameter alpha as 0.1 and maximum iterations is 100 (default setting). 

# In[30]:


poisson = linear_model.PoissonRegressor(alpha=1, max_iter=100)
poisson.fit(X=x_train, y=y_train)
poi_hat = poisson.predict(x_valid)
metrics.mean_squared_error(y_valid, poi_hat)


# 
# **Model 3: Gamma Regressor**
# 
#     Using Gamma Regressor as the last model. Used the same parameters as the Poisson Regressor. The Mean Squared Error is lower, so we will use the Gamma Regressor on our test set.

# In[31]:


gamma = linear_model.GammaRegressor(alpha=1, max_iter=100)
gamma.fit(x_train, y_train)
gamma_val = gamma.predict(x_valid)
metrics.mean_squared_error(y_valid, gamma_val)


# Testing with Gamma progessor on test dataset

# In[32]:


gamma_test = gamma.predict(x_test)
print(f'r2score --> {metrics.r2_score(y_test, gamma_test)}')


# **Analysis:** 
# 
#     The negative R-squared value means that the prediction tends to be less accurate that the average value of the data set over time.
# 
#     Correlation coefficient is low which indicates less relationship between the variables (random scatter of the points)
# 
# **Solution:**
# 
#     When more variables(features) are added, r-squared values typically increases.

# **Extended analysis on correlation using OLS regression**

#     When working with small datasets in general, your best bet is resampling. 
#     In other words, this would involve finding the mean and standard deviation of the samples 
#     in your existing data, and then generating random data that would conform to the same mean and standard deviation.

# In[33]:



mu=np.mean(df.hair_length)
mu


# In[34]:


stdev=np.std(df.hair_length)
stdev


# In[35]:


s_hair_length = np.random.normal(mu, stdev, 2000)
s_hair_length


# In[36]:


mu=np.mean(df.number_of_vet_visits)
stdev=np.std(df.number_of_vet_visits)

s_number_of_vet_visits = np.random.normal(mu, stdev, 2000)
s_number_of_vet_visits


# In[37]:


mu=np.mean(df.weight)
stdev=np.std(df.weight)

s_weight = np.random.normal(mu, stdev, 2000)
s_weight


# In[38]:


mu=np.mean(df.height)
stdev=np.std(df.height)

s_height = np.random.normal(mu, stdev, 2000)
s_height



# In[39]:


mu=np.mean(df_copy.age_at_death)
stdev=np.std(df_copy.age_at_death)

s_age_at_death = np.random.normal(mu, stdev, 2000)
s_age_at_death




# In[40]:


# creating new dataframe with resampled variables
df_2000 = pd.DataFrame(zip(s_age_at_death,s_hair_length,s_height, s_number_of_vet_visits,s_weight), columns=['age_at_death','hair_length','height', 'number_of_vet_visits','weight'])
print(df_2000)


# In[41]:


# Plot heatmap to visualize the correlations.
plt.figure(figsize = (10, 8))
# sns.heatmap(df.corr(), annot = True)
sns.heatmap(df_2000.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.title('Correlation between different features');


#     We still dont notice any significant correlation between features

# **Ordinary least squares (OLS) regression is a statistical method of analysis that estimates the relationship between one or more independent variables and a dependent variable**

# In[42]:


# OLS regression model
import statsmodels.api as sm

Y = df_2000['age_at_death']
X = df_2000['hair_length']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results_hair_length = model.fit()
results_hair_length.params


# In[43]:


Y = df_2000['age_at_death']
X = df_2000['number_of_vet_visits']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results_number_of_vet_visits = model.fit()
results_number_of_vet_visits.params


# In[44]:


Y = df_2000['age_at_death']
X = df_2000['height']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results_height = model.fit()
results_height.params


# In[45]:


Y = df_2000['age_at_death']
X = df_2000['weight']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results_weight = model.fit()
results_weight.params


# In[46]:


results_hair_length.tvalues


# **Analysis:**
# 
#     Ordinary least squares (OLS) regression analysis shows us the relationship between 'hair_length','height','number_of_vet_visits','weight' features/variables show very low/independent of the target feature age_at_death.
# 
# **Solution:**
#     
#     Will need more varibales to predict the life expectancy of cats. These variables provided shows low correlation/dependency to the target feature.

# End of document
# 
# Refrence: Stackoverflow
