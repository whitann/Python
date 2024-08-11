#!/usr/bin/env python
# coding: utf-8

# In[560]:


# import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


# In[562]:


#import file from mapped working directory
os.chdir('C:/Users/Whit/Documents/Data')
medical_df = pd.read_csv('medical_clean.csv',dtype={'locationid':np.int64})


# In[564]:


# view file data
medical_df.info()


# In[566]:


#detect duplicates
medical_df.duplicated()
print(medical_df.duplicated().value_counts())


# In[568]:


#View missing data
medical_df.isna().sum()


# In[570]:


#review for outliers
plt.boxplot(medical_df['Children'])
plt.show()


# In[572]:


#Determine range for outliers
UQ = medical_df['Children'].describe().loc['75%']
LQ = medical_df['Children'].describe().loc['25%']
IQR = UQ - LQ
OU= 1.5*IQR + UQ
OL=LQ-1.5*IQR
print (OU)
print (OL)


# In[574]:


#Drop Outliers
medical_df.drop(medical_df[(medical_df['Children'] >OU)].index, inplace=True)
medical_df.drop(medical_df[(medical_df['Children'] <OL)].index, inplace=True)
plt.boxplot(medical_df['Children'])
plt.show()


# In[576]:


#review for outliers
plt.boxplot(medical_df['Age'])
plt.show()


# In[578]:


#review for outliers
plt.boxplot(medical_df['Income'])
plt.show()


# In[580]:


#Determine range for outliers
UQ = medical_df['Income'].describe().loc['75%']
LQ = medical_df['Income'].describe().loc['25%']
IQR = UQ - LQ
OU= 1.5*IQR + UQ
OL=LQ-1.5*IQR
print (OU)
print (OL)


# In[582]:


#Drop Outliers
medical_df.drop(medical_df[(medical_df['Income'] >OU)].index, inplace=True)
medical_df.drop(medical_df[(medical_df['Income'] <OL)].index, inplace=True)

#review
plt.boxplot(medical_df['Income'])
plt.show()


# In[584]:


#Review for outliers
plt.boxplot(medical_df['VitD_levels'])
plt.show()


# In[586]:


#Determine range for outliers
UQ = medical_df['VitD_levels'].describe().loc['75%']
LQ = medical_df['VitD_levels'].describe().loc['25%']
IQR = UQ - LQ
OU= 1.5*IQR + UQ
OL=LQ-1.5*IQR
print (OU)
print (OL)


# In[588]:


#Drop Outliers
medical_df.drop(medical_df[(medical_df['VitD_levels'] >OU)].index, inplace=True)
medical_df.drop(medical_df[(medical_df['VitD_levels'] <OL)].index, inplace=True)
plt.boxplot(medical_df['VitD_levels'])
plt.show()


# In[590]:


#Review for outliers
plt.boxplot(medical_df['vitD_supp'])
plt.show()


# In[592]:


#Determine range for outliers
UQ = medical_df['vitD_supp'].describe().loc['75%']
LQ = medical_df['vitD_supp'].describe().loc['25%']
IQR = UQ - LQ
OU= 1.5*IQR + UQ
OL=LQ-1.5*IQR
print (OU)
print (OL)


# In[594]:


#Drop Outliers
medical_df.drop(medical_df[(medical_df['vitD_supp'] >OU)].index, inplace=True)
medical_df.drop(medical_df[(medical_df['vitD_supp'] <OL)].index, inplace=True)
plt.boxplot(medical_df['vitD_supp'])
plt.show()


# In[596]:


plt.boxplot(medical_df['Initial_days'])
plt.show()


# In[598]:


medical_df['ReAdmis'].describe()


# In[600]:


medical_df['Age'].describe()


# In[602]:


medical_df['Income'].describe()


# In[604]:


medical_df['VitD_levels'].describe()


# In[606]:


medical_df['Initial_days'].describe()


# In[608]:


medical_df['Children'].describe()


# In[610]:


medical_df['vitD_supp'].describe()


# In[612]:


medical_df['Complication_risk'].describe()


# In[614]:


medical_df['Marital'].describe()


# In[616]:


medical_df['Gender'].describe()


# In[618]:


medical_df['Soft_drink'].describe()


# In[620]:


medical_df['HighBlood'].describe()


# In[622]:


medical_df['Stroke'].describe()


# In[624]:


medical_df['Arthritis'].describe()


# In[626]:


medical_df['Diabetes'].describe()


# In[628]:


medical_df['Hyperlipidemia'].describe()


# In[630]:


medical_df['BackPain'].describe()


# In[632]:


medical_df['Anxiety'].describe()


# In[634]:


medical_df['Overweight'].describe()


# In[636]:


medical_df['Allergic_rhinitis'].describe()


# In[638]:


medical_df['Reflux_esophagitis'].describe()


# In[640]:


medical_df['Asthma'].describe()


# In[642]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['ReAdmis'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of ReAdmis')


# In[644]:


#Univariate visualization
plt.boxplot(medical_df['Children'])
plt.title("Univariate Visualization of Children")
plt.show()


# In[646]:


#Univariate visualization
plt.boxplot(medical_df['Age'])
plt.title("Univariate Visualization of Age")
plt.show()


# In[648]:


#Univariate visualization
plt.boxplot(medical_df['Income'])
plt.title("Univariate Visualization of Income")
plt.show()


# In[650]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['Marital'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Marital')


# In[652]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['Gender'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Gender')


# In[654]:


#Univariate visualization
plt.boxplot(medical_df['VitD_levels'])
plt.title("Univariate Visualization of VitD_levels")
plt.show()


# In[656]:


#Univariate visualization
plt.boxplot(medical_df['vitD_supp'])
plt.title("Univariate Visualization of vitD_supp")
plt.show()


# In[658]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['Soft_drink'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Soft_drink')


# In[660]:


fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['HighBlood'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of High BP')


# In[662]:


fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Stroke'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Stroke')


# In[664]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['Complication_risk'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Complication_risk')


# In[666]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Arthritis'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Arthritis')


# In[668]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Diabetes'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Diabetes')


# In[670]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Hyperlipidemia'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of HLD')


# In[672]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['BackPain'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of BackPain')


# In[674]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Anxiety'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Anx')


# In[676]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
medical_df['Overweight'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Overweight')


# In[678]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Allergic_rhinitis'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Allergic Rhinitis')


# In[680]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Reflux_esophagitis'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of GERD')


# In[682]:


#Univariate visualization
fig = plt.figure(figsize=(10,10)) 
plt.subplot2grid(fig_dims, (0, 0))
medical_df['Asthma'].value_counts().plot(kind='bar', 
                                     title='Univariate Visualization of Asthma')


# In[684]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Children", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[686]:


#Bivariate visualization
sns.stripplot(x="Age", y="ReAdmis", data=medical_df)


# In[688]:


#Bivariate visualization
sns.stripplot(x="Income", y="ReAdmis", data=medical_df)


# In[690]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Marital", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[692]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Gender", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[694]:


#Bivariate visualization
sns.stripplot(x="VitD_levels", y="ReAdmis", data=medical_df)


# In[696]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="vitD_supp", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[698]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Soft_drink", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[700]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="HighBlood", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[702]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Stroke", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[704]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Complication_risk", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[706]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Arthritis", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[708]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Diabetes", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[710]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Hyperlipidemia", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[712]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="BackPain", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[714]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Anxiety", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[716]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Overweight", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[718]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Allergic_rhinitis", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[720]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Reflux_esophagitis", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[722]:


#Bivariate visualization
sns.histplot(binwidth=0.5, x="Asthma", hue="ReAdmis", data=medical_df, stat="count", multiple="stack")


# In[724]:


#Bivariate visualization
sns.stripplot(x="Initial_days", y="ReAdmis", data=medical_df)


# In[726]:


#Map complication risk
scale_mapper = {"Low":1, "Medium":2, "High":3}
medical_df["Risk"] = medical_df["Complication_risk"].replace(scale_mapper)
print(medical_df["Risk"])


# In[728]:


#One Hot Encoding for Categorical Columns
dummy_df = pd.get_dummies(medical_df, columns = ['ReAdmis', 'Marital', 'Gender','Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma'], dtype=int)
dummy_df.info()


# In[730]:


#Drop extra columns
medical_df = dummy_df.drop(['ReAdmis_No', 'Marital_Never Married', 'Gender_Male','Soft_drink_No', 'HighBlood_No', 'Stroke_No', 'Overweight_No', 'Arthritis_No','Diabetes_No', 'Hyperlipidemia_No', 'BackPain_No', 'Anxiety_No', 'Allergic_rhinitis_No', 'Reflux_esophagitis_No', 'Asthma_No'], axis=1)
medical_df.info()


# In[732]:


#Cleaned dataset
medical_df.to_csv('clean_medical_D208_logistic.csv')


# In[734]:


#Initial logistic regression model
x =  medical_df[['Children', 'Age', 'Income', 'VitD_levels', 'vitD_supp', 'Risk', 'Initial_days', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Gender_Nonbinary', 'Soft_drink_Yes', 'HighBlood_Yes', 'Stroke_Yes', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Hyperlipidemia_Yes', 'BackPain_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
y = medical_df['ReAdmis_Yes'].values
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())


# In[736]:


#Heatmap
heatmap_df = medical_df[['ReAdmis_Yes', 'Age', 'Income', 'VitD_levels', 'Risk', 'Initial_days', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(heatmap_df.corr(), annot=True)
plt.show()


# In[738]:


#VIF
X = medical_df[['Age', 'Income', 'VitD_levels', 'Risk', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
vif_score=pd.DataFrame()
vif_score["feature"]=X.columns
vif_score['VIF']=[variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_score['VIF']=round(vif_score['VIF'],2)
vif_score=vif_score.sort_values(by="VIF", ascending = False)
print(vif_score)


# In[740]:


#VIF verification
X = medical_df[['Age', 'Income', 'Risk', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
vif_score=pd.DataFrame()
vif_score["feature"]=X.columns
vif_score['VIF']=[variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_score['VIF']=round(vif_score['VIF'],2)
vif_score=vif_score.sort_values(by="VIF", ascending = False)
print(vif_score)


# In[742]:


#Reduced logistic regression model
x =  medical_df[['Age', 'Income', 'Risk', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
y = medical_df['ReAdmis_Yes'].values
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())


# In[744]:


#Further reduced logistic regression model
x =  medical_df[['Income', 'Risk', 'Marital_Divorced', 'Gender_Female', 'Asthma_Yes']]
y = medical_df['ReAdmis_Yes'].values
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())


# In[746]:


#Initial logistic regression model score
x =  medical_df[['Children', 'Age', 'Income', 'VitD_levels', 'vitD_supp', 'Risk', 'Initial_days', 'Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Gender_Nonbinary', 'Soft_drink_Yes', 'HighBlood_Yes', 'Stroke_Yes', 'Overweight_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Hyperlipidemia_Yes', 'BackPain_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes']]
y = medical_df['ReAdmis_Yes'].values
model=LogisticRegression(solver='liblinear', random_state=0)
model.fit(x,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='12', random_state=0, solver = 'liblinear', tol=0.0001, verbose=0, warm_start=False)
model.score(x,y)


# In[748]:


#Reduced logistic regression model
x =  medical_df[['Income', 'Risk', 'Marital_Divorced', 'Gender_Female', 'Asthma_Yes']]
y = medical_df['ReAdmis_Yes'].values
model=LogisticRegression(solver='liblinear', random_state=0)
model.fit(x,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='12', random_state=0, solver = 'liblinear', tol=0.0001, verbose=0, warm_start=False)
model.predict(x)
model.score(x,y)


# In[750]:


#model intercept for final equation
model.intercept_


# In[752]:


#Confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=77)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# In[754]:


#Calculate Accuracy
Negative = matrix[0]
Positive = matrix [1]
TN = Negative[0]
FN = Negative[1]
FP = Positive[0]
TP = Positive[1]
Tr = TP+TN
Total = TP+TN+FP+FN
print("Accuracy is:", Tr/Total)


# In[ ]:




