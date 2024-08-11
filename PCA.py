#!/usr/bin/env python
# coding: utf-8

# In[101]:


# import packages
import pandas as pd
import numpy as np
import os
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


# In[102]:


#import file from mapped working directory
os.chdir('C:/Users/Whit/Documents/Data')
medical_df = pd.read_csv('medical_raw_data.csv',dtype={'locationid':np.int64})


# In[103]:


#check that data was imported
medical_df.head(5)


# In[104]:


# view data type
medical_df.info()


# In[105]:


#detect duplicates
medical_df.duplicated()
print(medical_df.duplicated().value_counts())


# In[106]:


#review unique data
for col in medical_df:
    print(medical_df[col].unique())


# In[107]:


#View missing data
medical_df.isna().sum()


# In[108]:


#visualize missing data
msno.matrix(medical_df, fontsize = 12, labels=True)


# In[109]:


#view Age column
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(medical_df['Age'])
plt.show 


# In[110]:


#This numerical data is bimodal, so we should replace missing values with the median
medical_df['Age'].fillna(medical_df['Age'].median(), inplace=True)


# In[111]:


#view Income column
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(medical_df['Income'])
plt.show


# In[112]:


#This numerical data has a skew to the right, so we should replace missing values with the median
medical_df['Income'].fillna(medical_df['Income'].median(), inplace=True)


# In[113]:


#For categorical data we should replace missing values with the mode
medical_df['Soft_drink']=medical_df['Soft_drink'].fillna(medical_df['Soft_drink'].mode()[0])
medical_df['Soft_drink'].isna().sum()


# In[114]:


#For categorical data we should replace missing values with the mode
medical_df['Overweight']=medical_df['Overweight'].fillna(medical_df['Overweight'].mode()[0])
medical_df['Overweight'].isna().sum()


# In[115]:


#For categorical data we should replace missing values with the mode
medical_df['Anxiety']=medical_df['Anxiety'].fillna(medical_df['Anxiety'].mode()[0])
medical_df['Anxiety'].isna().sum()


# In[116]:


#view Initial_days histogram
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(medical_df['Initial_days'])
plt.show


# In[117]:


#Change this to float datatype so that it can be plotted against Intial_days
medical_df=medical_df.replace({'ReAdmis':{'Yes':1, 'No': 0}})
medical_df['ReAdmis']=medical_df['ReAdmis'].astype('int')
medical_df['ReAdmis'].head(10000)


# In[118]:


#create a scatterplot
medical_df.plot(x='Initial_days', y='ReAdmis', kind='scatter')


# In[119]:


medical_df['ReAdmis'].value_counts()


# In[120]:


#find the median based off readmission status
medical_df.groupby('ReAdmis')['Initial_days'].median()


# In[121]:


#from datascience
swaps = {
    1: 64.276773,
    0: 10.360185
}

# create series of substitution values
substitutions = medical_df.apply(
    lambda row: swaps[row["ReAdmis"]],
    axis=1)


# In[122]:


# fill NA in Initial_days column with substitutions
medical_df["Initial_days"] = medical_df["Initial_days"].fillna(substitutions)


# In[123]:


# verify that all null values have been replaced
medical_df.isna().sum()


# In[124]:


#review Age for outliers
medical_df.boxplot(['Age'])
medical_df['Age'].describe()


# In[125]:


#review Income for outliers
medical_df.boxplot(['Income'])


# In[126]:


medical_df['Income'].describe()


# In[127]:


UQ = 54075.235000
LQ = 19450.7925
IQR = UQ - LQ
UW= 1.5*IQR + UQ
print (UW)


# In[128]:


#remove outliers
medical_df.drop(medical_df[(medical_df['Income'] >106011.89875)].index, inplace=True)
medical_df.boxplot(['Income'])


# In[129]:


#review VitD_levels for outliers
medical_df.boxplot(['VitD_levels'])


# In[130]:


#the values below 30 are likely ng/ml as this would be very low for nmol/L, formula from medscape
medical_df.loc[medical_df["VitD_levels"] < 30, "VitD_levels"] = medical_df["VitD_levels"] * 2.5


# In[131]:


#review boxplot again
medical_df.boxplot(['VitD_levels'])
medical_df['VitD_levels'].describe()


# In[132]:


UQ = 48.653617
LQ = 41.274550
IQR = UQ - LQ
UW= 1.5*IQR + UQ
LW= LQ-(1.5*IQR)
print (UW)
print (LW)


# In[133]:


#drop outliers
medical_df.drop(medical_df[(medical_df['VitD_levels'] >59.722175)].index, inplace=True)
medical_df.drop(medical_df[(medical_df['VitD_levels'] <30.2059495)].index, inplace=True)
medical_df.boxplot(['VitD_levels'])


# In[134]:


#review Initial_days for outliers
medical_df.boxplot(['Initial_days'])


# In[135]:


#Save clean data
medical_df.to_csv('medical_data_clean.csv')


# In[136]:


#import file from mapped working directory
clean_df = pd.read_csv('medical_data_clean.csv',dtype={'locationid':np.int64})


# In[137]:


clean_df.head()


# In[138]:


clean_df['Overweight']=clean_df['Overweight'].astype('int')


# In[139]:


clean_df=clean_df.replace({'HighBlood':{'Yes':1, 'No': 0}})
clean_df['HighBlood']=clean_df['HighBlood'].astype('int')


# In[140]:


clean_df=clean_df.replace({'Stroke':{'Yes':1, 'No': 0}})
clean_df['Stroke']=clean_df['Stroke'].astype('int')


# In[141]:


clean_df=clean_df.replace({'Arthritis':{'Yes':1, 'No': 0}})
clean_df['Arthritis']=clean_df['Arthritis'].astype('int')


# In[142]:


clean_df=clean_df.replace({'Diabetes':{'Yes':1, 'No': 0}})
clean_df['Diabetes']=clean_df['Diabetes'].astype('int')


# In[143]:


clean_df=clean_df.replace({'Hyperlipidemia':{'Yes':1, 'No': 0}})
clean_df['Hyperlipidemia']=clean_df['Hyperlipidemia'].astype('int')


# In[144]:


clean_df=clean_df.replace({'BackPain':{'Yes':1, 'No': 0}})
clean_df['BackPain']=clean_df['BackPain'].astype('int')


# In[145]:


clean_df=clean_df.replace({'Allergic_rhinitis':{'Yes':1, 'No': 0}})
clean_df['Allergic_rhinitis']=clean_df['Allergic_rhinitis'].astype('int')


# In[146]:


clean_df=clean_df.replace({'Reflux_esophagitis':{'Yes':1, 'No': 0}})
clean_df['Reflux_esophagitis']=clean_df['Reflux_esophagitis'].astype('int')


# In[147]:


clean_df=clean_df.replace({'Asthma':{'Yes':1, 'No': 0}})
clean_df['Asthma']=clean_df['Asthma'].astype('int')


# In[148]:


#Seperate the columns we need
analysis_df = clean_df[['Age', 'VitD_levels', 'Income', 'Initial_days', 'Overweight', 'HighBlood', 'Stroke', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma']]
analysis_df.head()


# In[149]:


#normalize the data
analysis_normalized=(analysis_df-analysis_df.mean())/analysis_df.std()


# In[150]:


#add components
pca = PCA(n_components=analysis_df.shape[1])
pca


# In[151]:


#save normalized data as a variable
pca.fit(analysis_normalized)
analysis_pca = pd.DataFrame(pca.transform(analysis_normalized),     
     columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8', 'PC9','PC10','PC11','PC12', 'PC13', 'PC14'])


# In[152]:


#loadings
loadings = pd.DataFrame(pca.components_.T,
     columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8', 'PC9','PC10','PC11','PC12', 'PC13', 'PC14'],
     index=analysis_df.columns)
loadings


# In[153]:


#scree plot
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show()


# In[154]:


cov_matrix = np.dot(analysis_normalized.T, analysis_normalized) / analysis_df.shape[0]
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for
eigenvector in pca.components_]


# In[155]:


#eigen plot
plt.plot(eigenvalues)
plt.plot ([0,13],[1,1], color='g', linestyle='-', linewidth=1)
plt.xlabel('number of components')
plt.ylabel('eigenvalue')
plt.show() 


# In[ ]:




