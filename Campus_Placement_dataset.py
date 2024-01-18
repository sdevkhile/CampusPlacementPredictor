#!/usr/bin/env python
# coding: utf-8

# # Campus Placement Data
# 

# ### Objective 
# 
# The primary objective of this project is to develop a classification model to predict whether a student will be placed or not in a campus placement scenario.
# 
# 
# ### Dataset Description
# 
# The dataset contains placement data of students in a XYZ campus. Key features include secondary,higher secondary school percentages, specialization, degree specialization, degree type, work experience, and salary offers to placed students. The target variable is binary, indicating whether a student is placed or not.
# 
# 
# ###  Model building using Logistic Regression  
# 
# In this project, we employ the Logistic Regression algorithm for binary classification. Logistic Regression is well-suited for scenarios where the outcome variable has two classes. The model utilizes the logistic function (sigmoid) to map input features to a probability range of [0, 1].
# 
# 
# ### Outline 
# 
# 1. Importing the Libraries
# 2. Reading the Dataset
# 3. Data pre-processing and EDA
# 4. Splitting the Dependent and Independent Variables
# 5. Splitting the dataset into Train and Test
# 6. Model Building (Without handling Imbalanced data)
# 7. HyperParameter Tuning using GridSearchCV
# 8. Model Building after balancing the data using SMOTE 
# 9. Conclusion
# 

# ### Importing libraries
# 

# In[1]:


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style='whitegrid')

from warnings import filterwarnings
filterwarnings('ignore')


# ### Reading the Dataset

# In[2]:


df = pd.read_csv(r"C:\Users\abhis\Downloads\Placement_Data_Full_Class.csv")


# In[3]:


df.head()


# ### Basic Information about the dataset

# In[4]:


df.shape


# In[5]:


print('*'*30 + ' No. of rows in the dataset are :', df.shape[0], '*'*30)
print('*'*30 + ' No. of columns in the dataset are :', df.shape[1], '*'*28)


# In[6]:


df.info()


# In[7]:


print(df.columns.to_list())


# ### Check for missing values

# In[8]:


df.isnull().sum()  

# Salary column has 67 missing values for students who did not get the placed.


# In[9]:


df[(df['salary'].isnull()) & (df['status'] == 'Placed')]


# ### Checking for duplicate records

# In[10]:


df.duplicated().sum()


# ### Statistical Analysis

# In[11]:


df.describe().transpose().style


# In[12]:


#Removing the sl_no from the dataframe

df.drop('sl_no', axis=1, inplace=True) 


# ### Exploratory Data Analysis

# In[13]:


# Checking correlation between the variables using heatmap

sns.heatmap(df.corr(), annot=True)
plt.show()


# ### Checking for outliers 

# In[14]:


Numerical_cols = df.select_dtypes(exclude='object')
Categorical_cols = df.select_dtypes(include = 'object')


# In[15]:


Numerical_cols


# In[16]:


Categorical_cols


# In[17]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,18))
fig.subplots_adjust(hspace=0.5)
sns.boxplot(df['ssc_p'], ax=axes[0,0]).set_title('SSC Percentage', fontweight='bold', fontsize=14)
sns.boxplot(df['hsc_p'], ax=axes[0,1]).set_title('HSC Percentage',fontweight='bold',fontsize=14)
sns.boxplot(df['degree_p'], ax=axes[1,0]).set_title('Degree Percentage',fontweight='bold',fontsize=14)
sns.boxplot(df['etest_p'], ax=axes[1,1]).set_title('Employability test Percentage',fontweight='bold',fontsize=14)
sns.boxplot(df['mba_p'], ax=axes[2,0]).set_title('MBA Percentage',fontweight='bold',fontsize=14)
sns.boxplot(df['salary'], ax=axes[2,1]).set_title('Salary for students that got placed',fontweight='bold',fontsize=14)
plt.show()


# In[18]:


df.hist(figsize=(20,18), bins=20, color='grey')
plt.show()


# In[19]:


# Adding the status variable in numerical format for better visualization

df['Status'] = df['status'].replace({'Placed': 1, 'Not Placed' : 0})


# In[20]:


plt.figure(figsize=(12,10))
sns.pairplot(data=df, hue='Status')
plt.show()


# #### Gender Distribution

# In[21]:


df['gender'].value_counts().plot(kind='pie', autopct='%0.1f%%', explode=(0.05,0.05), colors=['limegreen','lightcoral'])
plt.show()


# In[22]:


plt.figure(figsize=(8,6))
ax = sns.countplot(df['gender'], hue=df['status'], palette=['skyblue','green'])
for i in ax.containers:
    ax.bar_label(i)
plt.title('Placed Vs Not-Placed students by gender') 
plt.show()    


# In[23]:


# Student Distribution across all categories

fig, axes = plt.subplots(nrows= 2, ncols=2, figsize=(14,12))
ax=sns.countplot(df['ssc_b'], ax=axes[0,0])
for i in ax.containers:
    ax.bar_label(i)
ax1=sns.countplot(df['hsc_b'], ax=axes[0,1])
for i in ax1.containers:
    ax1.bar_label(i)
ax2=sns.countplot(df['hsc_s'], ax=axes[1,0])
for i in ax2.containers:
    ax2.bar_label(i)
ax3=sns.countplot(df['degree_t'], ax=axes[1,1])
for i in ax3.containers:
    ax3.bar_label(i)


# In[24]:


df['specialisation'].value_counts().plot(kind='pie', autopct='%0.1f%%', colors=['teal','skyblue'])
plt.show()


# In[25]:


df['status'].value_counts().plot(kind='pie', autopct='%0.1f%%', colors=['lightgreen','orange'])
plt.show()

# 69% students got placed while 31% did not get the placement.


# In[26]:


# Relationship between MBA percentage and salary:

sns.scatterplot(x=df['salary'],y=df['mba_p'])
plt.show()


# In[27]:


ax = sns.countplot(df['workex'], hue=df['status'], palette=['skyblue','indigo'])
for i in ax.containers:
    ax.bar_label(i)


# In[28]:


plt.pie(df['specialisation'].value_counts(), labels=df['specialisation'].value_counts().index, autopct='%0.1f%%', 
        explode=(0.02,0.02), colors=['teal','gold'])
plt.show()


# In[29]:


ax = sns.countplot(df['specialisation'], hue=df['status'], palette=['wheat','indigo'])
for i in ax.containers:
    ax.bar_label(i)
    


# In[30]:


data = pd.crosstab(df['specialisation'], df['status'])
data['Total'] = data['Placed']+ data['Not Placed']
data['% of students placed in each specialisation'] = data['Placed']*100/data['Total']
data


# In[31]:


pd.crosstab(index=[df['degree_t'], df['specialisation']], columns=df['status'])


# ### Encoding the categorical variables

# In[32]:


from sklearn.preprocessing import LabelEncoder


# In[33]:


le = LabelEncoder()


# In[34]:


columns = Categorical_cols.columns
columns


# In[35]:


df['gender'] = le.fit_transform(df['gender'])
df['ssc_b'] = le.fit_transform(df['ssc_b'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['hsc_s'] = le.fit_transform(df['hsc_s'])
df['degree_t'] = le.fit_transform(df['degree_t'])
df['workex'] = le.fit_transform(df['workex'])
df['specialisation'] = le.fit_transform(df['specialisation'])


# **Dropping Salary column as the students who did not get placed have the salary value as Null. This will create bias while model building as it is representing similar information as the Target variable 'status'***

# In[36]:


new_df = df.drop(['salary','status'], axis=1)
new_df.head()


# In[37]:


new_df.isnull().sum()


# ### Splitting the dataset into Train and test

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X = new_df.iloc[:,:-1]


# In[40]:


y = new_df.iloc[:,-1]


# In[41]:


X.head()


# In[42]:


y.head()


# In[43]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=101)


# In[44]:


x_train.shape, x_test.shape


# ### Case 1 : Model Building without handling the Imbalance in data
# 

# ## Model Building

# ### Logistic Regression without Hyperparameter Tuning

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


log_it = LogisticRegression(random_state=32)


# In[47]:


log_it.fit(x_train,y_train)


# In[48]:


y_pred_train = log_it.predict(x_train)
y_pred_test = log_it.predict(x_test)


# In[49]:


# Evaluating the model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[50]:


accuracy_score(y_test,y_pred_test) # Accuracy of Test data


# In[51]:


accuracy_score(y_train,y_pred_train) # Accuracy of Train data


# In[52]:


# Evaluating Test data 

print(classification_report(y_test,y_pred_test))


# In[53]:


confusion_matrix(y_test,y_pred_test)


# In[54]:


# Evaluating Train data 

print(classification_report(y_train, y_pred_train))


# In[55]:


confusion_matrix(y_train,y_pred_train)


# ### Logistic Regression with Hyperparameter Tuning using GridSearchCV

# In[56]:


from sklearn.model_selection import GridSearchCV


# In[57]:


# Specifying the parameters that we want to Hypertune

parameters = {'penalty': ['l1','l2','elasticnet'], 'C': [1,2,3,5,10,20,30,50], 'max_iter': [100,200,300]}


# In[58]:


log_it_grid = GridSearchCV(log_it, param_grid=parameters, scoring = 'accuracy', cv=10)


# In[59]:


log_it_grid.fit(x_train,y_train)


# In[60]:


print(log_it_grid.best_params_)


# In[61]:


y_pred_grid_test = log_it_grid.predict(x_test)


# In[62]:


y_pred_grid_train = log_it_grid.predict(x_train)


# In[63]:


accuracy_score(y_test,y_pred_grid_test) # Accuracy score of test data


# In[64]:


accuracy_score(y_train,y_pred_grid_train) # Accuracy score of train data


# In[65]:


# Evaluating Test data 

print(classification_report(y_test,y_pred_grid_test))


# In[66]:


confusion_matrix(y_test,y_pred_grid_test)


# In[67]:


# Evaluating Train data 

print(classification_report(y_train, y_pred_grid_train))


# In[68]:


confusion_matrix(y_train,y_pred_grid_train)


# ### Plotting Area Under Receiver Operating Curve (AUROC)

# ***AUC (Area Under the Curve)***: A metric that represents overall performance of a binary classification model based on the area under its ROC curve.
# 
# ***ROC Curve (Receiver Operating Characteristic Curve)***: It is a graphical plot illustrating the trade-off between True Positive Rate and False Positive Rate at various classification thresholds.
# 
# ***True Positive Rate (Sensitivity / tpr)***: Proportion of actual positives correctly identified by the model.
# 
# ***False Positive Rate (fpr)*** : The model incorrectly classifies the proportion of actual negatives as positives.
# 

# In[69]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[70]:


roc_auc_score(y_test,y_pred_test)


# In[71]:


roc_auc_score(y_train,y_pred_train)


# In[72]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred_test)


# In[73]:


plt.figure(figsize=(8,6))
plt.plot(fpr,tpr, color='indigo', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics (ROC) Curve')
plt.show()


# ***AUC represents the degree or measure of separability. AUC score of the model is coming out to be 0.88 which is quite good, as higher the area under the curve better the model is at distinguishing the target class.***

# ### Case 2 : Model Building after handling Imbalance data

# In[74]:


# Handling class Imbalance with SMOTE (Synthetic Minorty Over-sampling Technique)

from imblearn.over_sampling import SMOTE


# In[75]:


smote = SMOTE()


# In[76]:


y.value_counts()


# In[77]:


x_smote,y_smote = smote.fit_resample(X,y)


# In[78]:


y_smote.value_counts()


# In[79]:


# Splitting the data to train and test

xtrain,xtest,ytrain,ytest=train_test_split(x_smote.values,y_smote,test_size=0.2,random_state=101)


# In[80]:


log_it.fit(xtrain,ytrain)


# In[81]:


ys_pred_test = log_it.predict(xtest)
ys_pred_train= log_it.predict(xtrain)


# In[82]:


accuracy_score(ytest,ys_pred_test) #Accuracy score of Test data


# In[83]:


accuracy_score(ytrain,ys_pred_train) #Accuracy score of Train data


# In[84]:


# Evaluating Test data

print(classification_report(ytest,ys_pred_test))


# In[85]:


confusion_matrix(ytest,ys_pred_test)


# In[86]:


# Evaluating Train data

print(classification_report(ytrain,ys_pred_train))


# In[87]:


confusion_matrix(ytrain,ys_pred_train)


# ## Conclusion
# 
# ### Initial Model Evaluation
# 
# 1. **Without Handling Imbalance in Data:**
#    - The accuracy of the Logistic Regression model, without addressing class imbalance, is approximately 88% for both the Train      and Test dataset.
#    - This initial assessment provides a baseline understanding of the model's performance on the original, imbalanced dataset.
# 
# ### Hyperparameter Tuning
# 
# 2. **Post Hyperparameter Tuning:**
#    - After hyperparameter tuning the model, we observed that the accuracy remains consistent, hovering around 88% for both          Train and Test datasets.
#    - While hyperparameter tuning fine-tunes the model, it does not lead to a substantial improvement in accuracy in this case.
# 
# ### Impact of Handling Imbalanced Data
# 
# 3. **Handling Imbalanced Data:**
#    - There is change in the  performance of the model when addressing class imbalance using SMOTE
#    - The train accuracy increases to approximately 89%, indicating better capturing of patterns in the majority and minority        classes.
#    - The test accuracy is around 85% which is quite stable. 
# 
# ### Generalization without Overfitting or Underfitting
# 
# 4. **Absence of Overfitting or Underfitting:**
#    - Notably, throughout these model iterations, there is neither the case of overfitting nor underfitting.
#    - The model demonstrates consistent and reliable performance across both training and testing datasets.
# 

# In[ ]:




